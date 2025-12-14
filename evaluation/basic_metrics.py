import cv2
import numpy as np
from baselines.downsample_baseline import downsample_baseline
from simulator.phosphene_simulator import render_phosphene

def mse(img1, img2):
    """
    Compute Mean Squared Error between two images.
    Images must have the same shape.
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return np.mean((img1 - img2) ** 2)


if __name__ == "__main__":
    # -------- Load images --------
    image_path = "E:/Semi Study/CLC 12/Normal Pictures/Fanzhendong vs zhangben quarter final.jpg"

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original = cv2.resize(original, (256, 256), interpolation=cv2.INTER_AREA)
    if original is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    original_rgb = cv2.imread(image_path)
    baseline = downsample_baseline(original_rgb)
    phosphene = render_phosphene(original_rgb)

    assert original.shape == baseline.shape == phosphene.shape

    # -------- Compute metrics --------
    mse_baseline = mse(original, baseline)
    mse_phosphene = mse(original, phosphene)

    # -------- Print results --------
    print("=== Day 4: Basic Metrics ===")
    print(f"MSE (Downsample Baseline): {mse_baseline:.2f}")
    print(f"MSE (Phosphene Simulator): {mse_phosphene:.2f}")

    # -------- Save results --------
    with open("results/day4_metrics.txt", "w") as f:
        f.write("Day 4 Metrics\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"MSE (Downsample): {mse_baseline:.2f}\n")
        f.write(f"MSE (Phosphene): {mse_phosphene:.2f}\n")

'''
Output using this in the terminal instead of simply running the file: 
python -m evaluation.basic_metrics
'''
