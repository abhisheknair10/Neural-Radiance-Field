from importlib.metadata import PathDistribution
import os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from NeuralRadianceField.model import NeRFModel
import NeuralRadianceField.nerf_helpers as nerf_helpers


def main():
    device = 'cpu'

    # download data if not present
    if not os.path.exists('lego_data/lego_data_update.npz'):
        os.system('wget https://www.cs.cornell.edu/courses/cs5670/2023sp/projects/pa5/lego_data_update.npz -P lego_data/')

    # load data
    data = np.load('lego_data/lego_data_update.npz')
    intrinsics = torch.from_numpy(data['intrinsics'])
    images = torch.from_numpy(data['images'])
    tform_cam2world = torch.from_numpy(data['poses'])

    # get image dimensions
    img_height, img_width = images[0].shape[:2]

    # initialize NeRF model
    model = NeRFModel(
        num_hidden_layers=2,
        hidden_layer_size=256,
        rgb_channels=3,
        sigma_channels=1
    )

    # compute properties of rays
    ray_dir, ray_origin = nerf_helpers.compute_rays(
        height=img_height,
        width=img_width,
        intrinsics=intrinsics,
        tform_cam2world=tform_cam2world[0],
        device=device
    )
    
    # march along ray and sample points
    samples, depth = nerf_helpers.sample_ray(
        ray_dir=ray_dir,
        ray_origin=ray_origin,
        near=0.0,
        far=4.0,
        num_samples=8,
        device=device
    )

    # positional encoding
    samples = nerf_helpers.positional_encoding(
        tensor=samples,
        num_encoding_functions=8,
        include_input=True
    )

    print(samples.shape)
    print(depth.shape)


if __name__ == '__main__':
    main()