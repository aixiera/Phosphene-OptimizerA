import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseMaskedPhospheneEncoder(nn.Module):
    def __init__(self, input_channels=1, phosphene_dim=256):
        super(SparseMaskedPhospheneEncoder, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),  # 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Intensity head: global feature → vector
        self.fc_intensity = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, phosphene_dim)
        )

        # Mask head: spatial-aware 1x1 conv → 16x16 map → flatten
        self.mask_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),       # (B, 1, 16, 16)
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.backbone(x)                    # Shared CNN backbone: (B, 128, 16, 16)
        intensity = self.fc_intensity(feat)        # Global encoding vector (B, 256)
        mask = self.mask_head(feat)                # Spatially structured mask (B, 1, 16, 16)
        mask = mask.view(mask.size(0), -1)         # Flatten mask to match phosphene layout
        return intensity, mask


if __name__ == "__main__":
    model = SparseMaskedPhospheneEncoder()
    dummy_input = torch.randn(1, 1, 256, 256)
    intensity, mask = model(dummy_input)
    print("intensity:", intensity.shape)
    print("mask:", mask.shape)
    print("effective phosphene usage:", mask.detach().numpy().mean())
