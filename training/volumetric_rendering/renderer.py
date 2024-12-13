# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""

import math
import torch
import torch.nn as nn

from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.volumetric_rendering import math_utils
from torch_utils.ops import grid_sample_gradfix_ as grid_sample_gradfix

grid_sample_gradfix.enabled = True

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def normalize_coordinates_batch(coords):
    min_coords = coords.min(dim=1, keepdim=True)[0]  # Find the min for each batch across all 3 dimensions
    max_coords = coords.max(dim=1, keepdim=True)[0]  # Find the max for each batch across all 3 dimensions
    
    # Apply the linear transformation to map coordinates to [-1, 1]
    normalized_coords = 2 * (coords - min_coords) / (max_coords - min_coords) - 1
    return normalized_coords

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape 
    _, M, _ = coordinates.shape 

    plane_features = plane_features.view(N*n_planes, C, H, W)




    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)

    output_features = grid_sample_gradfix.grid_sample(plane_features, projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features


class ImportanceRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.plane_axes = generate_planes()
        self.instance_norm = nn.InstanceNorm2d(512, affine=True)


    def run_model(self, planes,sample_coordinates, sample_directions, options):
   
        self.plane_axes = self.plane_axes.to('cuda')
        sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])

        sampled_features = sampled_features.mean(1)

        return sampled_features 
    
    def run_model_no_mean(self, planes,sample_coordinates, sample_directions, options):
   
        self.plane_axes = self.plane_axes.to('cuda')

        # Apply InstanceNorm to each plane independently
        normalized_planes = []
        for i in range(planes.shape[1]):  # Loop over planes
            plane = planes[:, i]  # Select the i-th plane
            normalized_plane = self.instance_norm(plane)  # Apply InstanceNorm
            normalized_planes.append(normalized_plane)

        # Stack the normalized planes back along the plane dimension
        normalized_planes = torch.stack(normalized_planes, dim=1)  

        sampled_features = sample_from_planes(self.plane_axes, normalized_planes, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])
    
        concatenated_features = sampled_features.permute(0, 2, 1, 3).reshape(planes.size(0), 20000, -1)

        return concatenated_features

