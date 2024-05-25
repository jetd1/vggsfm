# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from visdom import Visdom

import torch
from torch.cuda.amp import autocast

import hydra
from omegaconf import DictConfig, OmegaConf

from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.renderer.cameras import PerspectiveCameras

from vggsfm.datasets.sequence_loader import SequenceLoader
from vggsfm.utils.metric import camera_to_rel_deg, calculate_auc_np
from vggsfm.utils2.run_utils import run_one_scene, get_test_model


@hydra.main(config_path="vggsfm/cfgs/", config_name="demo", version_base="1.1")
def test_fn(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    model, device = get_test_model(cfg)

    # Prepare test dataset
    test_dataset = SequenceLoader(
        SEQ_DIR=cfg.SEQ_DIR, img_size=cfg.image_size, normalize_cameras=False, load_gt=cfg.load_gt, cfg=cfg
    )

    error_dict = {"rError": [], "tError": []}

    if cfg.visualize:
        viz = Visdom()

    sequence_list = test_dataset.sequence_list

    for seq_name in sequence_list:
        print("*" * 50 + f" Testing on Scene {seq_name} " + "*" * 50)

        # Load the data
        batch, image_paths = test_dataset.get_data(sequence_name=seq_name, return_path=True)

        # Send to GPU
        images = batch["image"].to(device)
        crop_params = batch["crop_params"].to(device)

        if cfg.load_gt:
            translation = batch["T"].to(device)
            rotation = batch["R"].to(device)
            fl = batch["fl"].to(device)
            pp = batch["pp"].to(device)

            # Prepare gt cameras
            gt_cameras = PerspectiveCameras(
                focal_length=fl.reshape(-1, 2),
                principal_point=pp.reshape(-1, 2),
                R=rotation.reshape(-1, 3, 3),
                T=translation.reshape(-1, 3),
                device=device,
            )

        # Unsqueeze to have batch size = 1
        images = images.unsqueeze(0)
        crop_params = crop_params.unsqueeze(0)

        batch_size = len(images)

        with torch.no_grad(), autocast(enabled=cfg.use_bf16, dtype=torch.bfloat16):
            predictions = run_one_scene(
                model,
                images,
                crop_params=crop_params,
                query_frame_num=cfg.query_frame_num,
                return_in_pt3d=cfg.return_in_pt3d,
                max_ransac_iters=cfg.max_ransac_iters,
            )

        # Export prediction as colmap format
        reconstruction_pycolmap = predictions["reconstruction"]
        output_path = os.path.join(seq_name, "output")
        os.makedirs(output_path, exist_ok=True)
        reconstruction_pycolmap.write(output_path)

        with open(os.path.join(output_path, "file_order.txt"), "w") as file:
            for s in image_paths:
                file.write(s + "\n")  # Write each string with a newline

        pred_cameras = predictions["pred_cameras"]

        if cfg.visualize:
            pcl = Pointclouds(points=predictions["points3D"][None])
            visual_dict = {"scenes": {"points": pcl, "cameras": pred_cameras}}
            fig = plot_scene(visual_dict, camera_scale=0.05)
            viz.plotlyplot(fig, env=f"demo_visual", win="3D")

        # For more details about error computation,
        # You can refer to IMC benchmark
        # https://github.com/ubc-vision/image-matching-benchmark/blob/master/utils/pack_helper.py

        if cfg.load_gt:
            # Compute the error
            rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(pred_cameras, gt_cameras, device, batch_size)

            print(f"    --  Mean Rot   Error (Deg) for this scene: {rel_rangle_deg.mean():10.2f}")
            print(f"    --  Mean Trans Error (Deg) for this scene: {rel_tangle_deg.mean():10.2f}")

            error_dict["rError"].extend(rel_rangle_deg.cpu().numpy())
            error_dict["tError"].extend(rel_tangle_deg.cpu().numpy())

    if cfg.load_gt:
        rError = np.array(error_dict["rError"])
        tError = np.array(error_dict["tError"])

        # you can choose either calculate_auc/calculate_auc_np, they lead to the same result
        Auc_30, normalized_histogram = calculate_auc_np(rError, tError, max_threshold=30)
        Auc_3 = np.mean(np.cumsum(normalized_histogram[:3]))
        Auc_5 = np.mean(np.cumsum(normalized_histogram[:5]))
        Auc_10 = np.mean(np.cumsum(normalized_histogram[:10]))

        print(f"Testing Done")

        for _ in range(5):
            print("-" * 100)

        print("On the IMC dataset")
        print(f"Auc_3  (%): {Auc_3 * 100}")
        print(f"Auc_5  (%): {Auc_5 * 100}")
        print(f"Auc_10 (%): {Auc_10 * 100}")
        print(f"Auc_30 (%): {Auc_30 * 100}")

        for _ in range(5):
            print("-" * 100)

    return True


if __name__ == "__main__":
    with torch.no_grad():
        test_fn()
