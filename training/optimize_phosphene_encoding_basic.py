import os
import cv2
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn

from models.cnn_encoder import CNNPhospheneEncoder

'''
This file trains a neural network to learn how to optimally allocate a fixed number of phosphenes (256)
so that the resulting perceptual image preserves as much visual structure as possible.
'''

# Dataset
class ImageFolderDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.images = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".png"))
        ]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = self.transform(img)
        return img

# Differentiable Phosphene Renderer
class DifferentiablePhospheneRenderer(nn.Module):
    def __init__(self, grid_size=16, sigma=6):
        super().__init__()
        self.grid_size = grid_size

        kernel_size = 11
        ax = torch.arange(kernel_size) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        self.register_buffer(
            "gaussian_kernel",
            kernel.view(1, 1, kernel_size, kernel_size)
        )

    def forward(self, code):
        x = code.view(-1, 1, self.grid_size, self.grid_size)
        x = F.interpolate(x, size=(256, 256), mode="nearest")
        x = F.conv2d(
            x,
            self.gaussian_kernel,
            padding=self.gaussian_kernel.shape[-1] // 2
        )
        return x

# Training + Visualization
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESULT_DIR = "results/train_encoder"
    EPOCH_DIR = os.path.join(RESULT_DIR, "epoch_10")
    os.makedirs(EPOCH_DIR, exist_ok=True)

    dataset = ImageFolderDataset("data/train/fanzhendong")
    assert len(dataset) > 0, "No training images found"
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    encoder = CNNPhospheneEncoder().to(device)
    renderer = DifferentiablePhospheneRenderer().to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    encoder.train()
    renderer.eval()

    loss_history = []

    # Training Loop
    for epoch in range(10):
        epoch_loss = 0.0

        for images in loader:
            images = images.to(device)

            phosphene_code = encoder(images)
            rendered = renderer(phosphene_code)

            gt = torch.mean(images, dim=1, keepdim=True)
            loss = criterion(rendered, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    # Save loss curve
    with open(os.path.join(RESULT_DIR, "loss_curve.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for i, l in enumerate(loss_history):
            writer.writerow([i + 1, l])

    plt.figure()
    plt.plot(range(1, 11), loss_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve (Phosphene Encoder)")
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, "loss_curve.png"), dpi=200)
    plt.close()

    # Visualization on one fixed sample
    encoder.eval()
    with torch.no_grad():
        sample = dataset[0].unsqueeze(0).to(device)
        code = encoder(sample)
        rendered = renderer(code)
        original = torch.mean(sample, dim=1, keepdim=True)

    plt.imsave(
        os.path.join(EPOCH_DIR, "original.png"),
        original.squeeze().cpu().numpy(),
        cmap="gray"
    )

    plt.imsave(
        os.path.join(EPOCH_DIR, "phosphene.png"),
        rendered.squeeze().cpu().numpy(),
        cmap="gray"
    )

    #heatmap
    #The phosphene activation heatmap represents the learned allocation of limited stimulation resources across the visual field.
    #Each element corresponds to the intensity of a single phosphene, revealing 
    #how the encoder prioritizes spatial regions to preserve structural information under severe resolution constraints.
    heatmap = code.view(16, 16).cpu().numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(heatmap, cmap="inferno")
    plt.colorbar()
    plt.title("Phosphene Activation (16*16)")
    plt.axis("off")
    plt.savefig(os.path.join(EPOCH_DIR, "heatmap.png"), dpi=200)
    plt.close()

    #side-by-side comparison
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(original.squeeze().cpu(), cmap="gray")
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(rendered.squeeze().cpu(), cmap="gray")
    axs[1].set_title("Learned Phosphene")
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(EPOCH_DIR, "comparison.png"), dpi=200)
    plt.close()
    os.makedirs("models", exist_ok=True)
    torch.save(
        encoder.state_dict(),
        "models/optimize_encoder_basic.pth"
    )


if __name__ == "__main__":
    train()
    




'''
Output using this in the terminal instead of simply running the file: 
python -m training.optimize_phosphene_encoding_basic
'''