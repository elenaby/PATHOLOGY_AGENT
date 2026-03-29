from smolagents import tool
import cv2

@tool
def segment_image(image_path: str) -> str:
    """
    Segment the image and return a binary mask path.

    Args:
        image_path (str): Path to the input image.

    Returns:
        str: Path to the generated mask image.
    """

    output_mask = "outputs/mask.png"

    img = cv2.imread(image_path)
    print("Image loaded:", img is not None)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    cv2.imwrite(output_mask, mask)
    

    return output_mask