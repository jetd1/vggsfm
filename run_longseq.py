# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

from visdom import Visdom

import pycolmap

import torch
from torch.cuda.amp import autocast

import hydra
from omegaconf import DictConfig, OmegaConf

from vggsfm.datasets.imagefolder_loader import ImageFolderLoader
from vggsfm.utils.utils import segment_tensor_with_overlap_circular
from vggsfm.utils2.run_utils import run_one_scene, get_test_model


@hydra.main(config_path="vggsfm/cfgs/", config_name="longseq", version_base="1.1")
def test_fn(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    model, device = get_test_model(cfg)

    # Prepare test dataset
    test_dataset = ImageFolderLoader(
        SEQ_DIR=cfg.image_dir,
        img_size=cfg.image_size,
        normalize_cameras=False,
        load_gt=cfg.load_gt,
        cfg=cfg,
    )

    # if cfg.visualize:
    #     viz = Visdom()

    sequence_list = test_dataset.sequence_list
    assert len(sequence_list) == 1, "Only one sequence is allowed at a time"

    seq_name = sequence_list[0]
    print("*" * 20 + f" Testing on Scene {seq_name} " + "*" * 20)

    batch, image_paths = test_dataset.get_data(sequence_name=seq_name, return_path=True)
    all_images = batch["image"].to(device).unsqueeze(0)
    all_crop_params = batch["crop_params"].to(device).unsqueeze(0)

    segment_len = cfg.segment_len
    overlap_len = cfg.overlap_len

    images_segments = segment_tensor_with_overlap_circular(all_images, segment_len, overlap_len)
    crop_params_segments = segment_tensor_with_overlap_circular(all_crop_params, segment_len, overlap_len)
    print(f"Segmenting into {len(images_segments)} segments, {segment_len=}, {overlap_len=}")

    reconstructions = []
    for i, (images, crop_params) in enumerate(zip(images_segments, crop_params_segments)):
        print(f"Processing segment {i+1}/{len(images_segments)}")

        with torch.no_grad(), autocast(enabled=cfg.use_bf16, dtype=torch.bfloat16):
            predictions = run_one_scene(
                model,
                images,
                crop_params=crop_params,
                query_frame_num=cfg.query_frame_num,
                return_in_pt3d=cfg.return_in_pt3d,
                max_ransac_iters=cfg.max_ransac_iters,
            )

            reconstruction: pycolmap.Reconstruction = predictions["reconstruction"]
            output_path = os.path.join(cfg.output_path, f"{i}")
            os.makedirs(output_path, exist_ok=True)
            reconstruction.write(output_path)
            reconstruction.write_text(output_path)

    # if cfg.visualize:
    #     pcl = Pointclouds(points=predictions["points3D"][None])
    #     visual_dict = {"scenes": {"points": pcl, "cameras": pred_cameras}}
    #     fig = plot_scene(visual_dict, camera_scale=0.05)
    #     viz.plotlyplot(fig, env=f"demo_visual", win="3D")

    return True


if __name__ == "__main__":
    with torch.no_grad():
        test_fn()
