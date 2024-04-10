import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRFModel(nn.Module):
    def __init__(self, num_hidden_layers=8, hidden_layer_size=256, rgb_channels=3, sigma_channels=1):
        super(NeRFModel, self).__init__()

        self.rgb_channels = rgb_channels
        self.sigma_channels = sigma_channels

        self.input_layer = nn.Linear(5, hidden_layer_size)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_layer_size, hidden_layer_size) for _ in range(num_hidden_layers)]
        )

        self.output_layer = nn.Linear(hidden_layer_size, rgb_channels + sigma_channels)

    def forward(self, x):
        """
        Args:
            x: (ray_samples, (x, y, z, $\theta$, $\phi$))
        Returns:
            rgb: (ray_samples, (r, g, b))
            sigma: (ray_samples, $\sigma$)
        """
        
        x = F.relu(self.input_layer(x))

        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        x = self.output_layer(x)

        # x (shape): (ray_samples, 4)
        rgb, sigma = torch.split(x, [self.rgb_channels, self.sigma_channels], dim=-1)

        return rgb, sigma