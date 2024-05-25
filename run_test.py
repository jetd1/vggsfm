# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
from torch.cuda.amp import autocast

import hydra
from omegaconf import DictConfig, OmegaConf

from pytorch3d.renderer.cameras import PerspectiveCameras

from vggsfm.datasets.imc import IMCDataset
from vggsfm.utils.metric import camera_to_rel_deg, calculate_auc_np
from vggsfm.utils2.run_utils import run_one_scene, get_test_model


@hydra.main(config_path="vggsfm/cfgs/", config_name="test", version_base="1.1")
def test_fn(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    model, device = get_test_model(cfg)

    # Prepare test dataset
    test_dataset = IMCDataset(IMC_DIR=cfg.IMC_DIR, split="test", img_size=1024, normalize_cameras=False, cfg=cfg)

    error_dict = {"rError": [], "tError": []}

    sequence_list = test_dataset.sequence_list

    for seq_name in sequence_list:
        print("*" * 50 + f" Testing on Scene {seq_name} " + "*" * 50)

        # Load the data
        batch = test_dataset.get_data(sequence_name=seq_name)

        # Send to GPU
        images = batch["image"].to(device)
        translation = batch["T"].to(device)
        rotation = batch["R"].to(device)
        fl = batch["fl"].to(device)
        pp = batch["pp"].to(device)
        crop_params = batch["crop_params"].to(device)

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
            )

        pred_cameras = predictions["pred_cameras"]

        # For more details about error computation,
        # You can refer to IMC benchmark
        # https://github.com/ubc-vision/image-matching-benchmark/blob/master/utils/pack_helper.py

        # Compute the error
        rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(pred_cameras, gt_cameras, device, batch_size)

        print(f"    --  Mean Rot   Error (Deg) for this scene: {rel_rangle_deg.mean():10.2f}")
        print(f"    --  Mean Trans Error (Deg) for this scene: {rel_tangle_deg.mean():10.2f}")

        error_dict["rError"].extend(rel_rangle_deg.cpu().numpy())
        error_dict["tError"].extend(rel_tangle_deg.cpu().numpy())

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
