import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRFModel(nn.Module):
    def __init__(self, num_hidden_layers=8, hidden_layer_size=256, in_channels=5):
        super(NeRFModel, self).__init__()

        self.in_channels = in_channels

        self.input_layer = nn.Linear(in_channels, hidden_layer_size)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_layer_size, hidden_layer_size) for _ in range(num_hidden_layers)]
        )

        self.output_layer = nn.Linear(hidden_layer_size, 4)

    def forward(self, x):
        """
        Args:
            x: (ray_samples, (x, y, z, $\theta$, $\phi$))
        Returns:
            rgb: (ray_samples, (r, g, b))
            sigma: (ray_samples, $\sigma$)
        """
        
        x = F.relu(self.input_layer(x))
        mi = x

        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # with skip connection
        x = self.output_layer(x + mi)

        # x (shape): (ray_samples, 4)
        rgb = torch.sigmoid(x[..., :3])
        sigma = torch.relu(x[..., 3])

        return rgb, sigma