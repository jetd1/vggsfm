# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import random
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from accelerate.utils import set_seed as accelerate_set_seed, PrecisionType


import torch
import torch.nn as nn
import torch.nn.functional as F


from .metric import closed_form_inverse


def calculate_index_mappings(query_index, S, device=None):
    """
    Construct an order that we can switch [query_index] and [0]
    so that the content of query_index would be placed at [0]
    """
    new_order = torch.arange(S)
    new_order[0] = query_index
    new_order[query_index] = 0
    if device is not None:
        new_order = new_order.to(device)
    return new_order


def switch_tensor_order(tensors, order, dim=1):
    """
    Switch the tensor among the specific dimension
    """
    return [torch.index_select(tensor, dim, order) for tensor in tensors]


def set_seed_and_print(seed):
    accelerate_set_seed(seed, device_specific=True)
    print(f"----------Seed is set to {np.random.get_state()[1][0]} now----------")


def transform_camera_relative_to_first(pred_cameras, batch_size):
    pred_se3 = pred_cameras.get_world_to_view_transform().get_matrix()
    rel_transform = closed_form_inverse(pred_se3[0:1, :, :])
    rel_transform = rel_transform.expand(batch_size, -1, -1)

    pred_se3_rel = torch.bmm(rel_transform, pred_se3)
    pred_se3_rel[..., :3, 3] = 0.0
    pred_se3_rel[..., 3, 3] = 1.0

    pred_cameras.R = pred_se3_rel[:, :3, :3].clone()
    pred_cameras.T = pred_se3_rel[:, 3, :3].clone()
    return pred_cameras


def farthest_point_sampling(distance_matrix, num_samples, most_common_frame_index=0):
    # Number of points
    distance_matrix = distance_matrix.clamp(min=0)

    N = distance_matrix.size(0)

    # Initialize
    # Start from the first point (arbitrary choice)
    selected_indices = [most_common_frame_index]
    # Track the minimum distances to the selected set
    check_distances = distance_matrix[selected_indices]

    while len(selected_indices) < num_samples:
        # Find the farthest point from the current set of selected points
        farthest_point = torch.argmax(check_distances)
        selected_indices.append(farthest_point.item())

        check_distances = distance_matrix[farthest_point]
        # the ones already selected would not selected any more
        check_distances[selected_indices] = 0

        # Break the loop if all points have been selected
        if len(selected_indices) == N:
            break

    return selected_indices


def run_in_subbatch(model, batch, subbatch_size=32):
    outputs = []

    # Get the total number of samples in the batch
    batch_size = batch.size(0)

    # Process the batch in subbatches
    for i in range(0, batch_size, subbatch_size):
        subbatch = batch[i:i+subbatch_size]
        subbatch_outputs = model(subbatch)
        outputs.append(subbatch_outputs)

    return torch.cat(outputs, dim=0)


def seed_all_random_engines(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def segment_tensor_with_overlap_circular(
        tensor: torch.Tensor,
        segment_len: int,
        overlap_len: int
) -> list:
    # Get the shape of the input tensor
    b, v, *rest = tensor.shape

    # Calculate the number of segments
    num_segments = (v + overlap_len) // (segment_len - overlap_len)

    # Initialize an empty list to store the segmented tensors
    segmented_tensors = []

    # Iterate over the segments
    for i in range(num_segments):
        start_idx = i * (segment_len - overlap_len)
        end_idx = start_idx + segment_len

        # Check if the end index exceeds the tensor length
        if end_idx > v:
            # Perform circular padding for the last segment
            pad_len = end_idx - v
            padded_tenor = torch.cat((tensor, tensor[:, :pad_len, ...]), dim=1)
            segment = padded_tenor[:, start_idx:end_idx, ...]
        else:
            # Slice the tensor to get the current segment
            segment = tensor[:, start_idx:end_idx, ...]

        # Append the segment to the list of segmented tensors
        segmented_tensors.append(segment)

    return segmented_tensors
