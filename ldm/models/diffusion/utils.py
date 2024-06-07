import os
import yaml
import math
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory




def save_image_3d(tensor, slice_idx, file_name):
    '''
    tensor: [bs, c, h, w, 1]
    '''
    image_num = len(slice_idx)
    tensor = tensor[0, slice_idx, ...].permute(0, 3, 1, 2).cpu().data  # [c, 1, h, w]
    image_grid = vutils.make_grid(tensor, nrow=image_num, padding=0, normalize=True, scale_each=True)
    vutils.save_image(image_grid, file_name, nrow=1)



def map_coordinates(input, coordinates):
    ''' PyTorch version of scipy.ndimage.interpolation.map_coordinates
    input: (B, H, W, C)
    coordinates: (2, ...)
    '''
    bs, h, w, c = input.size()

    def _coordinates_pad_wrap(h, w, coordinates):
        coordinates[0] = coordinates[0] % h
        coordinates[1] = coordinates[1] % w
        return coordinates

    co_floor = torch.floor(coordinates).long()
    co_ceil = torch.ceil(coordinates).long()
    d1 = (coordinates[1] - co_floor[1].float())
    d2 = (coordinates[0] - co_floor[0].float())
    co_floor = _coordinates_pad_wrap(h, w, co_floor)
    co_ceil = _coordinates_pad_wrap(h, w, co_ceil)

    f00 = input[:, co_floor[0], co_floor[1], :]
    f10 = input[:, co_floor[0], co_ceil[1], :]
    f01 = input[:, co_ceil[0], co_floor[1], :]
    f11 = input[:, co_ceil[0], co_ceil[1], :]
    d1 = d1[None, :, :, None].expand(bs, -1, -1, c)
    d2 = d2[None, :, :, None].expand(bs, -1, -1, c)

    fx1 = f00 + d1 * (f10 - f00)
    fx2 = f01 + d1 * (f11 - f01)
    
    return fx1 + d2 * (fx2 - fx1)


def ct_parallel_project_2d(img, theta):
	bs, h, w, c = img.size()

	# (y, x)=(i, j): [0, w] -> [-0.5, 0.5]
	y, x = torch.meshgrid([torch.arange(h, dtype=torch.float32) / h - 0.5,
							torch.arange(w, dtype=torch.float32) / w - 0.5])

	# Rotation transform matrix: simulate parallel projection rays
	x_rot = x * torch.cos(theta) - y * torch.sin(theta)
	y_rot = x * torch.sin(theta) + y * torch.cos(theta)

	# Reverse back to index [0, w]
	x_rot = (x_rot + 0.5) * w
	y_rot = (y_rot + 0.5) * h

	# Resample (x, y) index of the pixel on the projection ray-theta
	sample_coords = torch.stack([y_rot, x_rot], dim=0).cuda()  # [2, h, w]
	img_resampled = map_coordinates(img, sample_coords) # [b, h, w, c]

	# Compute integral projections along rays
	proj = torch.mean(img_resampled, dim=1, keepdim=True) # [b, 1, w, c]

	return proj


def ct_parallel_project_2d_batch(img, thetas):
    '''
    img: input tensor [B, H, W, C]
    thetas: list of projection angles
    '''
    projs = []
    for theta in thetas:
      proj = ct_parallel_project_2d(img, theta)
      projs.append(proj)
    projs = torch.cat(projs, dim=1)  # [b, num, w, c]

    return projs