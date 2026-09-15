<!-- Created: 2026-09-15 18:44 -->

# CLAUDE.md (desktop_version)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this folder.

## Overview

MNIST digit recognizer (Tkinter GUI). This folder still contains **two copies** of the same app:

- `digit_recognition.py`: single-file version, auto-trains a model on first run if `digit_model.joblib` is missing.
- `handwriting-digit-recognizer/`: modularized version (`train_model.py` + `preprocess.py` + `digit_app.py`), requires running `train_model.py` first — it does **not** auto-train.

This is a standalone project: no shared build system or code with `../web_version`.

When editing digit-recognition logic (preprocessing, model params), check whether the fix needs to be applied in one or both of `digit_recognition.py` and `handwriting-digit-recognizer/preprocess.py` / `train_model.py` — they contain duplicated (copy-pasted) logic, not shared imports.

## Commands

There is no test suite, linter, or build tool configured.

### Single-file version
```bash
python digit_recognition.py
```
or double-click `Run_Digit_Recognition.bat` on Windows. Auto-trains and saves `digit_model.joblib` next to the script if it doesn't exist yet (downloads MNIST via `fetch_openml` on first run).

### Modular version
```bash
cd handwriting-digit-recognizer
pip install -r requirements.txt   # scikit-learn, numpy, Pillow
python train_model.py             # must run first — no auto-train
python digit_app.py
```

## Architecture notes

Both copies share the same pipeline, reimplemented rather than imported:

1. **Canvas capture**: Tkinter canvas paired with an off-screen PIL `Image` (`"L"` mode, black background) kept in sync stroke-by-stroke, so the same pixels drawn on screen can be fed to the model.
2. **Preprocessing** (`canvas_image_to_mnist_vector` in `digit_recognition.py` / `handwriting-digit-recognizer/preprocess.py`): crop tightly to the drawn strokes → resize to 20x20 → pad into a 28x28 canvas → re-center by center-of-mass (via `scipy.ndimage`) → flatten and normalize to a `(1, 784)` float32 vector in `[0, 1]`. This mirrors the classic MNIST normalization pipeline so hand-drawn input matches the training distribution.
3. **Model**: `sklearn.neural_network.MLPClassifier` (hidden layers `(128, 64)`, ReLU, Adam, `max_iter=30` with early stopping), trained on `mnist_784` from `fetch_openml`, persisted with `joblib`.
