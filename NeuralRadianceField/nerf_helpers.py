from itertools import accumulate
from re import I
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_rays(height, width, intrinsics, tform_cam2world, device='cpu'):
    """
    Description:
        Compute the direction for rays corresponding to each pixel in the image based on the 
        camera intrinsics and the camera poses. This if performed by creating a normalized grid 
        of pixel coordinates, changing their basis to camera coordinates, and then transforming 
        the camera coordinates to world coordinates using the camera poses. With each transformation, 
        we are calculating the ray properties based on its respective frame of reference.
    Args:
        height: (int) height of the image
        width: (int) width of the image
        intrinsics: (torch.Tensor) camera intrinsics matrix
        tform_cam2world: (torch.Tensor) camera poses
        device: (str) device to use (default: 'cpu')
    Returns:
        ray_dir: (torch.Tensor) ray directions
        ray_origin: (torch.Tensor) ray origins
    """

    assert intrinsics.shape[-2:] == (3, 3), "intrinsics must be a Nx3x3 matrix"
    assert tform_cam2world.shape[-2:] == (4, 4), "tform_cam2world must be a Nx4x4 matrix"

    # move intrinsics and tform_cam2world to device
    intrinsics = intrinsics.to(device)
    tform_cam2world = tform_cam2world.to(device)

    # create pixel grid
    # y: (height, width), x: (height, width)
    y, x = torch.meshgrid(
        torch.arange(height),
        torch.arange(width),
        indexing='ij'
    )

    # normalize pixel coordinates
    # y: (height, width), x: (height, width)
    y = y.float() / height
    x = x.float() / width

    # pixel coordinates
    # pixel_coords: (height, width, 3)
    pixel_coords = torch.stack([x, y, torch.ones_like(x)], dim=-1).float().to(device)

    # project pixel coordinates to camera coordinates
    # camera_coords: (3, height, width)
    camera_coords = torch.matmul(
        torch.inverse(intrinsics), 
        pixel_coords.permute(2, 0, 1).reshape(3, -1)
    ).reshape(3, height, width)

    # calculate ray directions
    # ray_dir: (height, width, 3)
    ray_dir = torch.matmul(
        tform_cam2world[:3, :3],
        camera_coords.reshape(3, -1)
    ).reshape(3, height, width).permute(1, 2, 0)
    
    # calculate ray origins
    # ray_origin: (height, width, 3)
    ray_origin = tform_cam2world[:3, 3].view(1, 1, 3).expand(height, width, 3)

    return ray_dir, ray_origin


def sample_ray(ray_dir, ray_origin, near, far, num_samples, device='cpu'):
    """
    Description:
        March along the ray from the near plane to the far plane and sample points along the ray.
        This is done by using the equation: p = o + td, where p is the point along the ray, o is the
        ray origin, t is the distance along the ray, and d is the ray direction. We sample points
        along the ray by dividing the distance between the near and far plane into num_samples
        intervals and sampling points at each interval.
    Args:
        ray_dir: (torch.Tensor) ray directions
        ray_origin: (torch.Tensor) ray origins
        near: (float) near plane inclusive
        far: (float) far plane exclusive
        num_samples: (int) number of samples
    Returns:
        samples: (torch.Tensor) sampled 3D points along the ray
        depth: (torch.Tensor) sampled depth values along the ray
    """

    # move ray_dir and ray_origin to device
    ray_dir = ray_dir.to(device)
    ray_origin = ray_origin.to(device)

    # calculate depth samples
    # depth: (num_samples)
    i_vector = torch.arange(0, num_samples, device=device)
    interval = (far - near) / num_samples
    depth = near + (interval * i_vector)

    # create extra dimension sampling points along each ray
    # each ray from a pixel receives a new set of sampled points, as opposed to each color channel
    ray_origin = ray_origin.unsqueeze(-2)
    ray_dir = ray_dir.unsqueeze(-2)

    # calculate 3d points along the ray
    # samples: (num_samples, 3)
    samples = ray_origin + depth.unsqueeze(-1) * ray_dir

    return samples, depth


def positional_encoding(tensor, num_encoding_functions=1, include_input=True):
    """
    Description:
        Compute positionally encoded tensor. This is performed by extending the tensor with
        sinusoidal functions of increasing frequencies mapping the tensor to a higher dimensional
        space. This is used to provide the model with information about the absolute position of
        the input tensor.
    Args:
        tensor: (torch.Tensor) input tensor
        num_encoding_functions: (int) number of encoding functions
        include_input: (bool) include the input tensor in the output tensor 
    Returns:
        output: (torch.Tensor) positionally encoded tensor
    """

    output = []
    if include_input:
        output.append(tensor)

    for i in range(0, num_encoding_functions):
        l_scaled = tensor * (2.0 ** i)

        sin_enc = torch.sin(l_scaled)
        cos_enc = torch.cos(l_scaled)

        fin_enc = torch.cat([sin_enc, cos_enc], dim=-1)
        output.append(fin_enc)

    # positionally encoded samples
    # output: (num_samples, (3 * 2 * num_encoding_functions) + (3 or 0))
    return torch.cat(output, dim=-1)


def compute_weights(sigma, depth, device='cpu'):
    """
    Description:
        Compute the weights for the volume rendering integral. This is performed by calculating the
        transmittance of the ray at each sample point along the ray. The transmittance is calculated
        as the product of the accumulated sigma values along the ray.
    Args:
        sigma: (torch.Tensor) volume density values at each sample point along the ray
        depth: (torch.Tensor) sampled depth values along the ray
        device: (str) device to use (default: 'cpu')
    Returns:
        weights: (torch.Tensor) weights for the volume rendering integral
    """

    # move sigma to device
    sigma = sigma.to(device)
    depth = depth.to(device)

    # calculate delta, the difference in depth values
    # delta: (num_samples)
    delta = torch.cat([
        depth[..., 1:] - depth[..., :-1], 
        torch.tensor([1e10], device=device).expand(depth[..., -1].unsqueeze(-1).shape)
    ], dim=-1)
    
    # calculate accumulated transmittance
    # accumulated_transmittance: (num_samples)
    accumulated_transmittance = torch.exp(-torch.cumprod(sigma * delta, dim=-1))

    # calculate normalized weights
    # weights: (num_samples)
    weights = accumulated_transmittance * (1 - torch.exp(-sigma * delta))

    return weights


def compute_ray_properties(rgb_samples, sigma_samples, depth, device='cpu'):
    """
    Description:
        Compute the pixel color by performing the volume rendering integral. This is done by
        calculating the weighted sum of the RGB values at each sample point along the ray.
    Args:
        rgb_samples: (torch.Tensor) RGB values at each sample point along the ray
        sigma_samples: (torch.Tensor) volume density values at each sample point along the ray
        depth: (torch.Tensor) sampled depth values along the ray
        device: (str) device to use (default: 'cpu')
    Returns:
        color: (torch.Tensor) pixel color
        depth: (torch.Tensor) pixel depth
    """

    # compute weights
    # weights: (num_samples)
    weights = compute_weights(sigma_samples, depth).to(device)

    # calculate pixel color
    # color: (3)
    color = torch.sum(weights.unsqueeze(-1) * rgb_samples, dim=2)

    # calculate pixel depth
    # depth: (1)
    depth = torch.sum(weights * depth, dim=2)

    return color, depth