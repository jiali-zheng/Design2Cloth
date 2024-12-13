import torch 
import torch.nn as nn 
import numpy as np 
import trimesh 
from torch import Tensor
from pathlib import Path
from meshudf.meshudf import get_mesh_from_udf, GridFiller
from models.D2C_Model import Design2Cloth_Model
from torchvision import transforms
import open3d as o3d
import os
import cv2

CHECKPOINT_PATH = './ckpt/D2C_ckpt.pt'
mask_pth = './sample_data/trousers.png'
shape_pth = './sample_data/trousers_shape.npy'

udf_max_dist   = 0.05
latent_size    = 64

ckpt = torch.load(CHECKPOINT_PATH)

model = Design2Cloth_Model(ckpt,latent_size).cuda()
model.eval()

# preprocessing function for loaded mask
mask_preprocess = transforms.Compose([
                                transforms.Resize(224, antialias=True),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])

# load mask
mask = cv2.imread(mask_pth)/255
masks_A = torch.tensor(mask,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
masks_C = mask_preprocess(masks_A)

# load shape information
shape_C = torch.tensor(np.load(shape_pth))

#encode mask and shape to latent space
with torch.no_grad():
    latent_codes_C = model(masks_C.cuda(), shape_C.cuda())
    


def udf_func(c: Tensor) -> Tensor:
    coords_encoded = model.coords_encoder.encode(c.unsqueeze(0))  
    projected_w = latent_codes_C

    transformed_ray_directions_expanded = torch.zeros((c.size(0), c.size(1), 3))
    transformed_ray_directions_expanded[..., -1] = -1

    sampled_features = model.G.sample_mixed(c.unsqueeze(0), transformed_ray_directions_expanded, projected_w, truncation_psi=1, noise_mode='const')
    p = model.decoder(coords_encoded, sampled_features)

    p = torch.sigmoid(p)
    p = (1 - p) * udf_max_dist
    return p

v, t = get_mesh_from_udf(
    udf_func,
    coords_range=(-1, 1),
    max_dist=udf_max_dist,
    N=512,
    max_batch=2**10,
    differentiable=False,
)


# Save the output mesh
o3d_mesh = o3d.geometry.TriangleMesh(
    vertices=o3d.utility.Vector3dVector(v.detach().cpu().numpy()),
    triangles=o3d.utility.Vector3iVector(t.cpu().numpy())
)
o3d_mesh = o3d_mesh.filter_smooth_simple(number_of_iterations=2)  # Adjust iterations
smoothed_trimesh = trimesh.Trimesh(
    vertices=np.asarray(o3d_mesh.vertices),
    faces=np.asarray(o3d_mesh.triangles)
)

directory_path = "./test_output/"
os.makedirs(directory_path, exist_ok=True)
smoothed_trimesh.export(directory_path+'output.obj')
