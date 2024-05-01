import os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from NeuralRadianceField.model import NeRFModel
import NeuralRadianceField.nerf_helpers as nerf_helpers


def load_data():
    # download data if not present
    if not os.path.exists('lego_data/lego_data_update.npz'):
        os.system('wget https://www.cs.cornell.edu/courses/cs5670/2023sp/projects/pa5/lego_data_update.npz -P lego_data/')

    # load data
    data = np.load('lego_data/lego_data_update.npz')
    intrinsics = torch.from_numpy(data['intrinsics'])
    images = torch.from_numpy(data['images'])
    poses = torch.from_numpy(data['poses'])

    return intrinsics, images, poses

def train():
    # load data
    intrinsics, images, poses = load_data()

    # set parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_height, img_width = images[0].shape[:2]

    # initialize NeRF model
    model = NeRFModel(num_hidden_layers=6, hidden_layer_size=256, in_channels=51).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    intrinsics, images, poses = intrinsics.to(device), images.to(device), poses.to(device)

    for epoch in range(10000):
        for i in range(len(images)):
            optimizer.zero_grad()

            # compute properties of rays
            ray_dir, ray_origin = nerf_helpers.compute_rays(
                height=img_height,
                width=img_width,
                intrinsics=intrinsics,
                tform_cam2world=poses[i],
                device=device
            )
            
            # march along ray and sample points
            samples, depth = nerf_helpers.sample_ray(
                ray_dir=ray_dir,
                ray_origin=ray_origin,
                near=0.5,
                far=2.0,
                num_samples=64,
                device=device
            )

            # positional encoding
            samples = nerf_helpers.positional_encoding(
                tensor=samples,
                num_encoding_functions=8
            )

            # forward pass
            rgb_samples, sigma_samples = model(samples)
            color, depth = nerf_helpers.compute_ray_properties(
                rgb_samples=rgb_samples,
                sigma_samples=sigma_samples,
                depth=depth,
                device=device
            )

            loss = F.mse_loss(color, images[i])
            loss.backward()
            optimizer.step()

            # plot image
            if i % 10 == 0:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                
                axes[0].imshow(color.cpu().detach().numpy())
                axes[0].set_title('Predicted Image')
                
                axes[1].imshow(images[i].cpu().detach().numpy())
                axes[1].set_title('Image')
                
                axes[2].imshow(depth.cpu().detach().numpy())
                axes[2].set_title('Depth')

                plt.title(f'Epoch: {epoch}, Image: {i}')
                plt.show()


if __name__ == '__main__':
    train()