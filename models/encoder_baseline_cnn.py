import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNPhospheneEncoder(nn.Module):
    """
    Simple CNN encoder that maps an image to 256 phosphene intensities.
    """

    def __init__(self, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        self.output_dim = grid_size * grid_size

        # simple convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )

        # connecting to 256 outputs
        self.fc = nn.Linear(128, self.output_dim)

    def forward(self, x):
        """
        x: (B, 3, H, W)
        """
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # output the strength for each phosphene in [0, 1]
        x = torch.sigmoid(x)

        return x
