# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from accelerate import Accelerator
from torch.cuda.amp import autocast
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_original_cwd
from pytorch3d.renderer.cameras import PerspectiveCameras

from gluefactory.models.extractors.superpoint_open import SuperPoint
from gluefactory.models.extractors.sift import SIFT

from vggsfm.datasets.imc import IMCDataset

from vggsfm.two_view_geo.estimate_preliminary import estimate_preliminary_cameras
from vggsfm.utils.utils import (
    set_seed_and_print,
    farthest_point_sampling,
    calculate_index_mappings,
    switch_tensor_order,
)
from vggsfm.utils.metric import camera_to_rel_deg, calculate_auc, calculate_auc_np


@torch.no_grad()
@torch.autocast(device_type='cuda', dtype=torch.bfloat16)
def run_one_scene(model, images, crop_params=None, query_frame_num=3, return_in_pt3d=False):
    """
    images have been normalized to the range [0, 1] instead of [0, 255]
    """
    batch_num, frame_num, image_dim, height, width = images.shape
    device = images.device
    reshaped_image = images.reshape(batch_num * frame_num, image_dim, height, width)

    camera_predictor = model.camera_predictor
    track_predictor = model.track_predictor
    triangulator = model.triangulator

    # Find the query frames
    # First use DINO to find the most common frame among all the input frames
    # i.e., the one has highest (average) cosine similarity to all others
    # Then use farthest_point_sampling to find the next ones
    # The number of query frames is determined by query_frame_num
    query_frame_indexes = find_query_frame_indexes(reshaped_image, camera_predictor, query_frame_num)

    # Prepare the methods to extract query points
    superpoint = SuperPoint({"nms_radius": 4, "force_num_keypoints": True}).cuda().eval()
    sift = SIFT({}).cuda().eval()

    # Prepare image feature maps for tracker
    fmaps_for_tracker = track_predictor.process_images_to_fmaps(images)

    pred_track_list = []
    pred_vis_list = []
    pred_score_list = []

    for query_index in query_frame_indexes:
        # Find query_points at the query frame
        query_points = get_query_points(superpoint, sift, images[:, query_index])

        # Switch so that query_index frame stays at the first frame
        # This largely simplifies the code structure of tracker
        new_order = calculate_index_mappings(query_index, frame_num, device=device)
        images_feed, fmaps_feed = switch_tensor_order([images, fmaps_for_tracker], new_order)

        # Feed into track predictor
        fine_pred_track, _, pred_vis, pred_score = track_predictor(images_feed, query_points, fmaps=fmaps_feed)

        # Switch back the predictions
        fine_pred_track, pred_vis, pred_score = switch_tensor_order([fine_pred_track, pred_vis, pred_score], new_order)

        # Append predictions for different queries
        pred_track_list.append(fine_pred_track)
        pred_vis_list.append(pred_vis)
        pred_score_list.append(pred_score)

    pred_track = torch.cat(pred_track_list, dim=2)
    pred_vis = torch.cat(pred_vis_list, dim=2)
    pred_score = torch.cat(pred_score_list, dim=2)

    # If necessary, force all the predictions at the padding areas as non-visible
    if crop_params is not None:
        boundaries = crop_params[:, :, -4:-2].abs().to(device)
        boundaries = torch.cat([boundaries, reshaped_image.shape[-1] - boundaries], dim=-1)
        hvis = torch.logical_and(
            pred_track[..., 1] >= boundaries[:, :, 1:2], pred_track[..., 1] <= boundaries[:, :, 3:4]
        )
        wvis = torch.logical_and(
            pred_track[..., 0] >= boundaries[:, :, 0:1], pred_track[..., 0] <= boundaries[:, :, 2:3]
        )
        force_vis = torch.logical_and(hvis, wvis)
        pred_vis = pred_vis * force_vis.float()

    # Estimate preliminary_cameras by recovering fundamental/essential/homography matrix from 2D matches
    # By default, we use fundamental matrix estimation with 7p/8p+LORANSAC
    # All the operations are batched and differentiable (if necessary)
    preliminary_cameras, preliminary_dict = estimate_preliminary_cameras(
        pred_track, pred_vis, width, height, tracks_score=pred_score, loopresidual=True
    )

    pose_predictions = camera_predictor(
        reshaped_image,
        batch_size=batch_num,
        preliminary_cameras=preliminary_cameras
    )

    # Conduct Triangulation and Bundle Adjustment

    # If we want to keep the result in the format of COLMAP,
    # please set return_in_pt3d = False,
    # and get the rot and trans by
    # BA_cameras.R, BA_cameras.T
    BA_cameras, _, _, _, _ = triangulator(
        pose_predictions["pred_cameras"],
        pred_track,
        pred_vis,
        images,
        preliminary_dict,
        pred_score=pred_score,
        return_in_pt3d=return_in_pt3d,
    )

    return BA_cameras


def get_query_points(superpoint, sift, query_image, max_query_num=4096):
    # Run superpoint and sift on the target frame
    # Feel free to modify for your own

    pred_sp = superpoint({"image": query_image})["keypoints"]
    pred_sift = sift({"image": query_image})["keypoints"]

    query_points = torch.cat([pred_sp, pred_sift], dim=1)

    if query_points.shape[1] > max_query_num:
        random_point_indices = torch.randperm(query_points.shape[1])[:max_query_num]
        query_points = query_points[:, random_point_indices, :]

    return query_points


def find_query_frame_indexes(reshaped_image, camera_predictor, query_frame_num, image_size=336):
    # Downsample image to image_size x image_size
    # because we found it is unnecessary to use high resolution
    rgbs = F.interpolate(reshaped_image, (image_size, image_size), mode="bilinear", align_corners=True)
    rgbs = camera_predictor._resnet_normalize_image(rgbs)

    # Get the image features (patch level)
    frame_feat = camera_predictor.backbone(rgbs, is_training=True)
    frame_feat = frame_feat["x_norm_patchtokens"]
    frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)

    # Compute the similiarty matrix
    frame_feat_norm = frame_feat_norm.permute(1, 0, 2)
    similarity_matrix = torch.bmm(frame_feat_norm, frame_feat_norm.transpose(-1, -2))
    similarity_matrix = similarity_matrix.mean(dim=0)
    distance_matrix = 1 - similarity_matrix.clone()

    # Ignore self-pairing
    similarity_matrix.fill_diagonal_(0)

    similarity_sum = similarity_matrix.sum(dim=1)

    # Find the most common frame
    most_common_frame_index = torch.argmax(similarity_sum).item()

    # Conduct FPS sampling
    # Starting from the most_common_frame_index,
    # try to find the farthest frame,
    # then the farthest to the last found frame
    # (frames are not allowed to be found twice)
    fps_idx = farthest_point_sampling(distance_matrix, query_frame_num, most_common_frame_index)

    return fps_idx


def get_model(cfg):
    OmegaConf.set_struct(cfg, False)
    accelerator = Accelerator(even_batches=False, device_placement=False)

    # Print configuration and accelerator state
    accelerator.print("Model Config:", OmegaConf.to_yaml(cfg), accelerator.state)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Set seed
    set_seed_and_print(cfg.seed)

    # Model instantiation
    model = instantiate(cfg.MODEL, _recursive_=False, cfg=cfg)

    device = accelerator.device
    model = model.to(device)

    # Accelerator setup
    model = accelerator.prepare(model)

    if cfg.resume_ckpt:
        # Reload model
        checkpoint = torch.load(cfg.resume_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        accelerator.print(f"Successfully resumed from {cfg.resume_ckpt}")