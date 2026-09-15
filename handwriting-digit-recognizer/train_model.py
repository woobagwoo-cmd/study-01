"""
Train a handwritten digit classifier on the MNIST dataset and save it to disk.

Run once before starting the GUI app:
    python train_model.py

The trained model is saved as `digit_model.joblib` in the same folder.
"""

import time

import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

MODEL_PATH = "digit_model.joblib"


def load_mnist():
    """Download (and locally cache) the MNIST dataset as normalized vectors."""
    print("Loading MNIST dataset (first run downloads ~18 MB, then caches it)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0  # scale pixels to [0, 1]
    y = mnist.target.astype(np.int64)
    return X, y


def train():
    X, y = load_mnist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
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

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f} seconds.")

    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
