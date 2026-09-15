"""
Shared preprocessing utilities to turn a hand-drawn digit image into an
MNIST-style 28x28 input vector.

MNIST images are: 28x28 grayscale, digit drawn in white (high values) on a
black background (low values), roughly centered using the center of mass
of the ink. We replicate that here so a digit drawn on the GUI canvas
looks statistically similar to what the model was trained on.
"""

import numpy as np
from PIL import Image
from scipy import ndimage


def _center_by_mass(image: np.ndarray) -> np.ndarray:
    """Shift the image so the center of mass of the ink sits at the image center."""
    cy, cx = ndimage.center_of_mass(image)
    if np.isnan(cy) or np.isnan(cx):
        return image  # blank image, nothing to center

    rows, cols = image.shape
    shift_y = int(round(rows / 2.0 - cy))
    shift_x = int(round(cols / 2.0 - cx))
    return ndimage.shift(image, shift=(shift_y, shift_x), cval=0)


def _crop_to_bounding_box(image: np.ndarray, pad_ratio: float = 0.2) -> np.ndarray:
    """Crop the image tightly around the drawn strokes, keeping a small margin."""
    ys, xs = np.where(image > 10)
    if len(ys) == 0:
        return image  # nothing drawn

    top, bottom = ys.min(), ys.max()
    left, right = xs.min(), xs.max()

    height = bottom - top + 1
    width = right - left + 1
    side = max(height, width)
    pad = int(side * pad_ratio)

    cy = (top + bottom) // 2
    cx = (left + right) // 2
    half = side // 2 + pad

    top = max(cy - half, 0)
    bottom = min(cy + half, image.shape[0] - 1)
    left = max(cx - half, 0)
    right = min(cx + half, image.shape[1] - 1)

    return image[top:bottom + 1, left:right + 1]


def canvas_image_to_mnist_vector(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image from the drawing canvas (white ink on black
    background, any size) into a (1, 784) float32 vector scaled to
    [0, 1], matching the format expected by the trained model.
    """
    gray = np.array(pil_image.convert("L"), dtype=np.float32)

    cropped = _crop_to_bounding_box(gray)
    cropped_img = Image.fromarray(cropped.astype(np.uint8))

    # Resize to 20x20 first (like the classic MNIST pipeline), then pad to
    # 28x28 so the digit doesn't touch the border after resizing.
    resized_20 = cropped_img.resize((20, 20), Image.LANCZOS)
    canvas28 = np.zeros((28, 28), dtype=np.float32)
    canvas28[4:24, 4:24] = np.array(resized_20, dtype=np.float32)

    centered = _center_by_mass(canvas28)
    centered = np.clip(centered, 0, 255)

    vector = centered.reshape(1, 784) / 255.0
    return vector.astype(np.float32)
