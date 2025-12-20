import os
import time
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import tqdm

from models.encoder_sparse_masked import SparseMaskedPhospheneEncoder
from training.learn_encoder_baseline import render_phosphenes_torch
from training.train_encoder_sparse_masked import build_dataloader


def main():
    # Configuration
    data_dir = "data/train/fanzhendong"  # path to your image dataset
    output_dir = "results/SelectiveSparseActivation"
    batch_size = 8
    model_path = "models/encoder_sparse_masked.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rendered"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "mask_heatmap"), exist_ok=True)

    # Load trained model
    model = SparseMaskedPhospheneEncoder().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load image data (grayscale)
    dataloader = build_dataloader(
        data_dir, batch_size=batch_size
    )

    # Start generation
    start_time = time.time()
    with torch.no_grad():
        for idx, (images, _) in enumerate(
            tqdm.tqdm(dataloader, desc="Generating images")
        ):
            images = images.to(device)
            intensity, mask = model(images)
            encoded = intensity * mask
            rendered = render_phosphenes_torch(encoded)  # (B, 1, 256, 256)

            for b in range(images.size(0)):
                i = idx * batch_size + b
                save_image(
                    images[b], os.path.join(output_dir, "original", f"{i:04d}.png")
                )
                save_image(
                    rendered[b], os.path.join(output_dir, "rendered", f"{i:04d}.png")
                )

                # Save resized mask heatmap (from 16x16 to 256x256)
                m = mask[b].view(1, 16, 16)
                m_resized = F.interpolate(
                    m.unsqueeze(0), size=(256, 256), mode="nearest"
                )
                save_image(
                    m_resized, os.path.join(output_dir, "mask_heatmap", f"{i:04d}.png")
                )

    elapsed = time.time() - start_time
    print(f"Done. Total generation time: {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()

'''
Output using this in the terminal instead of simply running the file: 
python -m evaluation.generate_sparse_images
'''