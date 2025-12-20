import cv2
import numpy as np



def render_phosphene(
    image,
    grid_size=16,
    output_size=256,
    sigma=6
):
    """
    Render a phosphene image from an input RGB image SYSTEMATICALLY, which is why it is called renderer :)
    Args:
        image (np.ndarray): Input RGB image (H, W, 3)
        grid_size (int): Phosphene grid size (grid_size x grid_size)
        output_size (int): Output image size (output_size x output_size)
        sigma (int): Gaussian blur strength for each phosphene
    Returns:
        phosphene_img (np.ndarray): Rendered phosphene image (output_size x output_size)
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to phosphene grid
    small = cv2.resize(
        gray,
        (grid_size, grid_size),
        interpolation=cv2.INTER_AREA
    )

    # Normalize intensity
    small = small.astype(np.float32) / 255.0

    # Upsample back to output resolution
    phosphene = cv2.resize(
        small,
        (output_size, output_size),
        interpolation=cv2.INTER_NEAREST
    )

    # Apply Gaussian blur to simulate phosphene spread
    phosphene = cv2.GaussianBlur(
        phosphene,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )
    phosphene = np.clip(phosphene * 255, 0, 255).astype(np.uint8)
    return phosphene


if __name__ == "__main__":
    # -------- Demo usage --------

    # Change this path to any image you have
    image_path = "E:/Semi Study/CLC 12/Normal Pictures/Fanzhendong vs zhangben quarter final.jpg"

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    phosphene_img = render_phosphene(image)

    cv2.imshow("Original Image", image)
    cv2.imshow("Phosphene Image (256)", phosphene_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
Update this with this commands:
git add .
git commit -m "Add minimal 256-phosphene simulator"
git push
'''
