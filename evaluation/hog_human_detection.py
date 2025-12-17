import os
import cv2
import csv

INPUT_BASE = "results/Learned_versus_naive_phosphene"
OUTPUT_BASE = "results/Learned_versus_naive_phosphene"

METHODS = ["original", "naive", "learned"]

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

summary = []

for method in METHODS:
    input_dir = os.path.join(INPUT_BASE, method)
    output_dir = os.path.join(OUTPUT_BASE, f"detection_{method}")
    os.makedirs(output_dir, exist_ok=True)

    total = 0
    detected = 0

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".jpg", ".png")):
            continue

        img = cv2.imread(os.path.join(input_dir, fname))
        rects, _ = hog.detectMultiScale(img)

        total += 1
        if len(rects) > 0:
            detected += 1

        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imwrite(os.path.join(output_dir, fname), img)

    rate = detected / total if total > 0 else 0
    summary.append([method, total, detected, rate])

# Save summary
with open(os.path.join(OUTPUT_BASE, "detection_summary.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Method", "Total Images", "Detected", "Detection Rate"])
    for row in summary:
        writer.writerow(row)

print("Detection summary saved.")

'''
Output using this in the terminal instead of simply running the file: 
python -m evaluation.hog_human_detection
'''