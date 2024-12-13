# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone

from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim 
        self.c_dim=c_dim 
        self.w_dim=w_dim 
        self.img_resolution=img_resolution 
        self.img_channels=img_channels 
        self.renderer = ImportanceRenderer()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=self.img_resolution, img_channels = 3*self.img_channels, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
    
        self._last_planes = None
    
    def mapping(self, z, truncation_psi=1, truncation_cutoff=None, update_emas=False):

        return self.backbone.mapping(z,  truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def sample_mixed(self, coordinates, directions, z, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)

        planes = planes.view(len(planes), 3, self.img_channels, planes.shape[-2], planes.shape[-1])
     
        return self.renderer.run_model(planes, coordinates, directions, self.rendering_kwargs)

    def sample_mixed_no_mean(self, coordinates, directions, z, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, self.img_channels, planes.shape[-2], planes.shape[-1])

        return self.renderer.run_model_no_mean(planes, coordinates, directions, self.rendering_kwargs)


from training.networks_stylegan2 import FullyConnectedLayer
