import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from models.encoder_baseline_cnn import CNNPhospheneEncoder
from training.learn_encoder_baseline import DifferentiablePhospheneRenderer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIR = "data/train/fanzhendong"
OUTPUT_DIR = "results/Learned_versus_naive_phosphene"
os.makedirs(OUTPUT_DIR, exist_ok=True)
for sub in ["original", "naive", "learned"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)
# Load model
encoder = CNNPhospheneEncoder().to(DEVICE)
encoder.load_state_dict(torch.load("models/learn_encoder_baseline.pth", map_location=DEVICE))
encoder.eval()

renderer = DifferentiablePhospheneRenderer().to(DEVICE)
renderer.eval()
transform = transforms.ToTensor()

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith((".jpg", ".png")):
        continue

    img = cv2.imread(os.path.join(INPUT_DIR, fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        code = encoder(tensor)
        learned = renderer(code).squeeze().cpu().numpy()

    # Original grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Naive phosphene (downsample + blur)
    naive = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    naive = cv2.resize(naive, (256, 256), interpolation=cv2.INTER_NEAREST)
    naive = cv2.GaussianBlur(naive, (11, 11), 6)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "original", fname), gray)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "naive", fname), naive)
    learned_img = np.clip(learned, 0, 1)          # Normalize to [0, 1] and turn to uint8 for opencv
    learned_img = (learned_img * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "learned", fname), learned_img)

print(code.view(16,16)[0:3, 0:3])

'''
Output using this in the terminal instead of simply running the file: 
python -m evaluation.generate_phosphene_images_basic
'''