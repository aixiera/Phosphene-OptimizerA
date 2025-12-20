import os
from typing import List, Tuple
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader, has_file_allowed_extension


class ImageFileDataset(Dataset):
    """Load all images under a directory, regardless of class subfolders.
    This class is a lightweight fallback for when ``ImageFolder`` cannot find
    images (e.g., when data is not organized by class). It mirrors the basic
    interface of torchvision datasets and returns a dummy label of 0 for every
    sample.
    """

    def __init__(self, root: str, transform=None, extensions=IMG_EXTENSIONS):
        self.root = root
        self.transform = transform
        self.extensions = extensions
        self.samples = self._find_images()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No image files found in {root}. Supported extensions: {', '.join(self.extensions)}"
            )

    def _find_images(self):
        return _list_image_files(self.root, self.extensions)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        image = default_loader(path)
        if self.transform:
            image = self.transform(image)
        return image, 0


def build_dataloader(data_dir: str, image_size: Tuple[int, int] = (256, 256), batch_size: int = 8) -> DataLoader:
    """Create a DataLoader for a directory of images.

    The loader prefers ``ImageFolder`` for class-structured data but falls back
    to a simple file list when no classes contain images. When no supported
    images are present at all, the function raises with a clear, actionable
    message.
    """
    if not os.path.isdir(data_dir):
        raise RuntimeError(f"Data directory does not exist: {data_dir}")

    available_images = _list_image_files(data_dir)
    if not available_images:
        raise RuntimeError(
            "No image files found. Please place .jpg, .png, or similar files under "
            f"{data_dir} (nested folders are fine)."
        )

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]
    )

    try:
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        if len(dataset) == 0:
            raise RuntimeError("ImageFolder found no images; falling back to plain image loader.")
    except (FileNotFoundError, RuntimeError):
        dataset = ImageFileDataset(root=data_dir, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())


def _list_image_files(root: str, extensions=IMG_EXTENSIONS) -> List[str]:
    image_paths: List[str] = []
    for base, _, files in os.walk(root):
        for fname in files:
            if has_file_allowed_extension(fname, extensions):
                image_paths.append(os.path.join(base, fname))
    return sorted(image_paths)


class SparseMaskedAutoencoder(nn.Module):
    """A lightweight encoder-decoder with a masking step in the latent space."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        if self.training and mask_ratio > 0:
            mask = (torch.rand_like(latent) > mask_ratio).float()
            latent = latent * mask
        reconstruction = self.decoder(latent)
        return reconstruction, latent


def train(data_dir: str = "data/train/fanzhendong", epochs: int = 50, lr: float = 1e-3, l1_weight: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_dataloader(data_dir)

    model = SparseMaskedAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs("results/train_encoder_sparse_masked", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, _ in loader:
            images = images.to(device)
            recon, latent = model(images)

            reconstruction_loss = criterion(recon, images)
            sparsity_loss = l1_weight * latent.abs().mean()
            loss = reconstruction_loss + sparsity_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f} (recon + sparsity)")

    torch.save(model.state_dict(), "models/sparse_masked_encoder.pth")


if __name__ == "__main__":
    start_time = time.time()
    train()
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

'''
Output using this in the terminal instead of simply running the file: 
python -m training.train_encoder_sparse_masked
'''