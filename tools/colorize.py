from smolagents import tool
import cv2
import numpy as np

@tool
def color_instances(mask_path: str) -> str:
    """
    Color each connected component in the mask.

    Args:
        mask_path (str): Path to the binary mask image.

    Returns:
        str: Path to the colored output image.
    """

    output_path = "outputs/colored.png"

    mask = cv2.imread(mask_path, 0)

    num_labels, labels = cv2.connectedComponents(mask)

    h, w = labels.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    np.random.seed(42)

    for i in range(1, num_labels):
        color = np.random.randint(0, 255, 3)
        colored[labels == i] = color

    cv2.imwrite(output_path, colored)

    return output_path