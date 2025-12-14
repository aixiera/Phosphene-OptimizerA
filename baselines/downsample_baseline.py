import cv2
import numpy as np


def downsample_baseline(image, grid_size=16, output_size=256):
    """
    Simple downsampling baseline:
    resize -> resize back
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    small = cv2.resize(
        gray,
        (grid_size, grid_size),
        interpolation=cv2.INTER_AREA
    )

    restored = cv2.resize(
        small,
        (output_size, output_size),
        interpolation=cv2.INTER_NEAREST
    )

    return restored


if __name__ == "__main__":
    image_path = "E:/Semi Study/CLC 12/Normal Pictures/Fanzhendong vs zhangben quarter final.jpg"
    image = cv2.imread(image_path)

    baseline_img = downsample_baseline(image)

    cv2.imshow("Downsample Baseline", baseline_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()