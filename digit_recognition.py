"""
Handwritten digit recognizer (single-file version).

Draw a digit (0-9) with the mouse on the canvas, then click "Predict" to
have a trained MNIST model guess which digit it is.

Run:
    python digit_recognition.py

Requires `digit_model.joblib` (a scikit-learn MLPClassifier trained on
MNIST) to sit next to this script. If it's missing, this script trains
one automatically (downloads MNIST on first run, then caches it).
"""

import os
import tkinter as tk
from tkinter import messagebox

import joblib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "digit_model.joblib")

CANVAS_SIZE = 280  # on-screen canvas size in pixels (10x the MNIST resolution)
BRUSH_RADIUS = 10  # stroke thickness while drawing


# --------------------------------------------------------------------------
# Preprocessing: turn a hand-drawn image into an MNIST-style 28x28 vector
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Model loading (train automatically if no saved model is found)
# --------------------------------------------------------------------------

def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        print("Loading model...")
        return joblib.load(MODEL_PATH)

    print("No saved model found. Training a new one on MNIST (this may take a minute)...")
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y
    )

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=30,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=42,
        verbose=True,
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------

class DigitRecognizerApp:
    def __init__(self, root: tk.Tk, model):
        self.root = root
        self.model = model
        self.root.title("Handwritten Digit Recognizer")
        self.root.resizable(False, False)

        # Off-screen image kept in sync with the on-screen canvas strokes,
        # so we can run the same pixels through the model later.
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self._build_widgets()
        self._last_x = None
        self._last_y = None

    def _build_widgets(self):
        instructions = tk.Label(
            self.root,
            text="Draw a single digit (0-9), then click Predict",
            font=("Segoe UI", 11),
        )
        instructions.pack(pady=(10, 0))

        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            highlightthickness=1,
            highlightbackground="gray",
            cursor="pencil",
        )
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=(0, 10))

        predict_btn = tk.Button(
            button_frame, text="Predict", width=12, command=self.predict
        )
        predict_btn.grid(row=0, column=0, padx=5)

        clear_btn = tk.Button(
            button_frame, text="Clear", width=12, command=self.clear_canvas
        )
        clear_btn.grid(row=0, column=1, padx=5)

        self.result_label = tk.Label(
            self.root, text="Prediction: -", font=("Segoe UI", 20, "bold")
        )
        self.result_label.pack(pady=(0, 5))

        self.confidence_label = tk.Label(
            self.root, text="", font=("Segoe UI", 10), fg="gray"
        )
        self.confidence_label.pack(pady=(0, 10))

    def _on_mouse_down(self, event):
        self._last_x, self._last_y = event.x, event.y
        self._paint_dot(event.x, event.y)

    def _on_mouse_move(self, event):
        if self._last_x is not None:
            self._paint_line(self._last_x, self._last_y, event.x, event.y)
        self._last_x, self._last_y = event.x, event.y

    def _on_mouse_up(self, event):
        self._last_x, self._last_y = None, None

    def _paint_dot(self, x, y):
        r = BRUSH_RADIUS
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r, fill="white", outline="white"
        )
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def _paint_line(self, x0, y0, x1, y1):
        r = BRUSH_RADIUS
        self.canvas.create_line(
            x0, y0, x1, y1, fill="white", width=r * 2, capstyle=tk.ROUND, smooth=True
        )
        self.draw.line([x0, y0, x1, y1], fill=255, width=r * 2)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Prediction: -")
        self.confidence_label.config(text="")

    def predict(self):
        vector = canvas_image_to_mnist_vector(self.image)

        if not np.any(vector > 0):
            messagebox.showinfo("No drawing", "Please draw a digit first.")
            return

        probabilities = self.model.predict_proba(vector)[0]
        predicted_digit = int(np.argmax(probabilities))
        confidence = probabilities[predicted_digit] * 100

        self.result_label.config(text=f"Prediction: {predicted_digit}")
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")


def main():
    model = load_or_train_model()

    root = tk.Tk()
    DigitRecognizerApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
