import PIL
import math
import numpy as np
import torch
import cv2
import torch.nn as nn
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from omnidata.midas.dpt_depth import OmnidataDepthModel, OmnidataNormalModel

from fileio import read_pfm, write_pfm
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# just randomly set fov
fov = 90
stereo_path = Path('stereo')
stereo_path.mkdir(exist_ok=True, parents=True)
(stereo_path / 'raw').mkdir(exist_ok=True, parents=True)
(stereo_path / 'output').mkdir(exist_ok=True, parents=True)
(stereo_path / 'mask').mkdir(exist_ok=True, parents=True)
(stereo_path / 'inpainted').mkdir(exist_ok=True, parents=True)
(stereo_path / 'inpainted-mask').mkdir(exist_ok=True, parents=True)

if not (stereo_path / 'bedroom.png').exists():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    prompt = f"a photo of a bedroom with fov of {fov} degrees"
    bedroom_image = pipe(prompt).images[0]  
    bedroom_image.save(stereo_path / 'bedroom.png')
else:
    bedroom_image = PIL.Image.open(stereo_path / 'bedroom.png')

H, W = bedroom_image.height, bedroom_image.width
bedroom_tensor = transforms.functional.to_tensor(bedroom_image)

# depth_model = OmnidataDepthModel().eval().cuda()
# trans = transforms.ToTensor()
# norm_trans = transforms.Normalize(mean=0.5, std=0.5)
# def compute_depth(pil_image):
#     with torch.no_grad():
#         cropped = trans(pil_image)
#         normalized = norm_trans(cropped)
#         depth = depth_model(normalized[None].cuda()).clamp(0, 1).view(512, 512)
#     return depth

if not (stereo_path / 'bedroom.pfm').exists():
    depth = compute_depth(bedroom_image)
    write_pfm(stereo_path / 'bedroom.pfm', depth)
else:
    depth = read_pfm(stereo_path / 'bedroom.pfm').view(H, W)

# perform projection
cx = 256
cy = 256
fx = math.atan(fov * math.pi / 180.0 / 2) * cx
fy = math.atan(fov * math.pi / 180.0 / 2) * cy
K = torch.tensor([
    fx, 0, cx,
    0, fy, cy,
    0, 0, 1
]).view(3, 3)

# generate poses for rendering
ref_pose = torch.eye(4)
mean_dist = depth.mean()
# assume front = z+
center_of_orbit = torch.zeros(3)
center_of_orbit[2] = mean_dist



# create rays
H, W = depth.shape[:2]
O = 0.5
f = K[0, 0]

x_coords = torch.linspace(O, W - 1 + O, W)
y_coords = torch.linspace(O, H - 1 + O, H)

# HxW grids
y_grid_coords, x_grid_coords = torch.meshgrid([y_coords, x_coords])

# HxWx3
h_coords = torch.stack([x_grid_coords, y_grid_coords, torch.ones_like(x_grid_coords)], -1)
rays = h_coords @ K.inverse().T
points = rays * depth[..., None]

# warp points based on target pose
h_points = torch.cat([
    points, 
    torch.ones_like(points[..., :1])
], -1)

H = 256
W = 256
cx = 128
cy = 128
sfx = math.atan(fov  * 2 * math.pi / 180.0 / 2) * cx
sfy = math.atan(fov  * 2 * math.pi / 180.0 / 2) * cy
src_K = torch.tensor([
    sfx, 0, cx,
    0, sfy, cy,
    0, 0, 1
]).view(3, 3)




theta = 10 * math.pi / 180
# phi = 0 * math.pi / 180

def sample_E(theta, phi):
    init_pos = torch.tensor([ 0, math.sin(theta) * mean_dist, 0 ])


    # rotate around z axis by phi
    z_rot = torch.tensor([
        math.cos(phi), -math.sin(phi), 0,
        math.sin(phi), math.cos(phi), 0,
        0, 0, 1
    ]).view(3, 3)


    target_pos = z_rot @ init_pos

    z = center_of_orbit - target_pos
    z = nn.functional.normalize(z, p=2, dim=0)
    # front = torch.tensor([0, 0, 1])
    up = torch.tensor([0, 1, 0.0])
    right = up.cross(z)
    up = z.cross(right)

    target_R = torch.eye(3)
    target_R[0] = right
    target_R[1] = up
    target_R[2] = z

    src_E = torch.eye(4)
    src_E[:3, :3] = target_R
    src_E[:3, 3] = -target_R @ target_pos
    return src_E

pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
h_points = h_points.view(-1, 4)
ref_colors = bedroom_tensor.view(3, -1).T

phis = torch.arange(0, 60) / 60.0 * math.pi * 2
src_Es = []
for i, phi in tqdm(enumerate(phis), total=len(phis)):
    src_E = sample_E(theta, phi)
    src_Es.append(src_E)

    src_img_hp = (h_points @ src_E.T)[:, :3] @ src_K.T
    src_img_p = (src_img_hp[:, :2] / src_img_hp[:, 2:])
    src_img_id = (src_img_p - 0.5).long()
    src_img_valids = (src_img_id >= 0).all(-1) & (src_img_id[:, 0] < H)  & (src_img_id[:, 1] < W)
    src_flat_ids = (src_img_id[:, 0] + src_img_id[:, 1] * W)[src_img_valids]

    ref_flat_rgb = ref_colors[src_img_valids]

    # src_img_p must be colored with bedroom image
    src_image = torch.zeros(H, W, 3).view(-1, 3)
    src_mask_image = torch.zeros(H, W, 3).view(-1, 3)
    src_image.scatter_(0, index=src_flat_ids.view(-1, 1).repeat(1, 3), src=ref_flat_rgb)
    src_mask_image.scatter_(0, index=src_flat_ids.view(-1, 1).repeat(1, 3), src=torch.ones_like(ref_flat_rgb))

    src_image = src_image.view(H, W, 3).permute(2, 0, 1)
    src_mask_image = src_mask_image.view(H, W, 3).permute(2, 0, 1)

    src_pil_image = transforms.functional.to_pil_image(src_image)
    src_pil_image = src_pil_image.resize((512, 512), PIL.Image.BILINEAR)
    src_pil_image.save(stereo_path / 'raw' / f'bedroom_{i}_raw.png')

    src_pil_mask_image = transforms.functional.to_pil_image(1 - src_mask_image)
    src_pil_mask_image = src_pil_mask_image.resize((512, 512), PIL.Image.NEAREST)
    src_pil_mask_image.save(stereo_path / 'mask' / f'bedroom_{i}_mask.png')

    src_np_image = np.array(src_pil_image)
    src_np_mask = np.array(src_pil_mask_image)

    inpainted = cv2.inpaint(src_np_image, src_np_mask[..., :1],5,cv2.INPAINT_TELEA)
    pil_inpainted = PIL.Image.fromarray(inpainted)
    pil_inpainted.save(stereo_path / 'inpainted' / f'bedroom_{i}.png')

    # np_mask_image = np.array(src_pil_mask_image)
    # # Taking a matrix of size 5 as the kernel
    # kernel = np.ones((5, 5), np.uint8)
    # np_mask_image = cv2.erode(np_mask_image, kernel, iterations=1)
    # src_pil_mask_image = PIL.Image.fromarray((np_mask_image > 0).astype(np.uint8) * 255)
    # src_pil_mask_image.save(stereo_path / 'inpainted-mask' / f'bedroom_{i}_mask.png')

    # # # src_pil_mask_image = transforms.functional.to_pil_image(1 - src_mask_image)
    # # src_pil_mask_image = src_pil_mask_image.resize((512, 512), PIL.Image.NEAREST)
    # # src_pil_mask_image.save(stereo_path / 'mask' / f'bedroom_{i}_mask.png')


    # prompt = f"a photo of a bedroom with fov of {fov} degrees"
    # output_image = pipe(prompt=prompt, image=pil_inpainted, mask_image=src_pil_mask_image).images[0]
    # output_image.save(stereo_path / 'output' / f'bedroom_{i}.png')
    # output_tensor = trans(output_image)

    # new_depth = compute_depth(output_image).cpu()
    # src_rays = h_coords @ src_K.inverse().T
    # ray_mask = torch.from_numpy((np_mask_image > 0)[..., 0].astype(np.uint8))
    # src_masked_rays = src_rays[ray_mask]
    # masked_new_depth = new_depth[ray_mask]
    # masked_output_color = output_tensor.permute(1, 2, 0)[ray_mask]

    # new_points = src_masked_rays* masked_new_depth[..., None]

    # # warp points based on target pose
    # new_h_points = (src_E.inverse() @ torch.cat([
    #     new_points, 
    #     torch.ones_like(new_points[..., :1])
    # ], -1).T).T

    # h_points = torch.cat([h_points, new_h_points])
    # ref_colors = torch.cat([ref_colors, masked_output_color])
torch.save(torch.stack(src_Es), 'poses.pth')