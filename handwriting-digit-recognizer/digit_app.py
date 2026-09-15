"""
Handwritten digit recognizer GUI.

Draw a digit (0-9) with the mouse on the canvas, then click "Predict" to
have the trained model guess which digit it is.

Run:
    python digit_app.py

Requires `digit_model.joblib`, created by running `train_model.py` first.
"""

import os
import tkinter as tk
from tkinter import messagebox

import joblib
import numpy as np
from PIL import Image, ImageDraw

from preprocess import canvas_image_to_mnist_vector

MODEL_PATH = "digit_model.joblib"
CANVAS_SIZE = 280  # on-screen canvas size in pixels (10x the MNIST resolution)
BRUSH_RADIUS = 10  # stroke thickness while drawing


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
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"Model file '{MODEL_PATH}' not found. Run 'python train_model.py' first."
        )

    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    root = tk.Tk()
    DigitRecognizerApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
