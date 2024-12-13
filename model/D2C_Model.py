import torch
import torch.nn as nn
from models.cbndec import CbnDecoder

from models.coordsenc import CoordsEncoder
from models.MLP import MLP


import dnnlib
import legacy
import torch
import numpy as np
from torch_utils import misc
from training.triplane import TriPlaneGenerator


class Design2Cloth_Model(nn.Module):
    def __init__(self,ckpt, latent_size):
        super(Design2Cloth_Model, self).__init__()
        encoder = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)

        encoder.classifier = nn.Sequential(*[nn.Linear(1280, 256), nn.ReLU(), nn.Linear(256, latent_size), nn.ReLU()])


        encoder.load_state_dict(ckpt["encoder"])
        self.encoder = encoder 

        shape_encoder = MLP(input_dim=10, hidden_dim=64, output_dim=32)
        shape_encoder.load_state_dict(ckpt['shape_encoder'])
        self.shape_encoder = shape_encoder
        

        self.coords_encoder = CoordsEncoder()
        decoder_hidden_dim =512
        decoder_num_hidden_layers = 8
        decoder = CbnDecoder(
                        self.coords_encoder.out_dim,
                        32, 
                        decoder_hidden_dim,
                        decoder_num_hidden_layers,
                    )
        decoder.load_state_dict(ckpt["decoder"])
        self.decoder = decoder 


        rendering_kwargs = {'depth_resolution': 64, 'depth_resolution_importance': 64, 'ray_start': 2.35, 'ray_end': 2.6, 'box_warp': 1.6, 'white_back': True, 'avg_camera_radius': 1.7, 'avg_camera_pivot': [0, 0, 0], 'image_resolution': 128, 'disparity_space_sampling': False, 'clamp_mode': 'softplus', 'superresolution_module': 'training.superresolution.SuperresolutionHybrid2X', 'c_gen_conditioning_zero': True, 'c_scale': 1.0, 'superresolution_noise_mode': 'none', 'density_reg': 0.25, 'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0, 'sr_antialias': True}

        G = TriPlaneGenerator(z_dim=96, c_dim=0,  w_dim=512, img_resolution=256,img_channels=32, rendering_kwargs = rendering_kwargs)

        G.load_state_dict(ckpt["G"])
        self.G = G

        
    def forward(self, mask, shape): 
        latent_space = self.encoder(mask)
        latent_shape = self.shape_encoder(shape)
        latent_codes = torch.cat([latent_space, latent_shape] , -1)
        
        return latent_codes



