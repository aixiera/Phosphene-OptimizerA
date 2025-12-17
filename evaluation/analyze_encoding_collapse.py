import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from models.cnn_encoder import CNNPhospheneEncoder
from training.optimize_phosphene_encoding_basic import DifferentiablePhospheneRenderer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data/train/fanzhendong"
RESULT_DIR = "results/Learned_versus_naive_phosphene/analysis_encoding_collapse"
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------- Load model ----------------
encoder = CNNPhospheneEncoder().to(DEVICE)
encoder.load_state_dict(
    torch.load("models/optimize_encoder_basic.pth", map_location=DEVICE)
)
encoder.eval()

renderer = DifferentiablePhospheneRenderer().to(DEVICE)
renderer.eval()

transform = transforms.ToTensor()

codes = []
heatmaps = []

# ---------------- Extract codes ----------------
for fname in os.listdir(DATA_DIR):
    if not fname.lower().endswith((".jpg", ".png")):
        continue

    img = cv2.imread(os.path.join(DATA_DIR, fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        code = encoder(tensor).squeeze().cpu().numpy()

    codes.append(code)
    heatmaps.append(code.reshape(16, 16))

codes = np.stack(codes)           # (N, 256)
heatmaps = np.stack(heatmaps)     # (N, 16, 16)

# ---------------- Statistics ----------------
mean_code = codes.mean(axis=0)
var_code = codes.var(axis=0)
global_variance = var_code.mean()

# Save statistics
with open(os.path.join(RESULT_DIR, "code_statistics.txt"), "w") as f:
    f.write(f"Number of samples: {codes.shape[0]}\n")
    f.write(f"Mean variance across phosphenes: {global_variance:.6f}\n")
    f.write(f"Max variance: {var_code.max():.6f}\n")
    f.write(f"Min variance: {var_code.min():.6f}\n")

# ---------------- Variance plot ----------------
plt.figure(figsize=(6, 3))
plt.plot(var_code)
plt.title("Per-phosphene variance across samples")
plt.xlabel("Phosphene index")
plt.ylabel("Variance")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "code_variance.png"), dpi=200)
plt.close()

# ---------------- Heatmap visualization ----------------
sample_indices = np.linspace(0, len(heatmaps) - 1, 5, dtype=int)

fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for ax, idx in zip(axes, sample_indices):
    ax.imshow(heatmaps[idx], cmap="inferno")
    ax.axis("off")
fig.suptitle("Learned phosphene heatmaps across different inputs")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "heatmap_samples.png"), dpi=200)
plt.close()

print("Encoding collapse analysis completed.")

'''
Output using this in the terminal instead of simply running the file: 
python -m evaluation.analyze_encoding_collapse
'''