# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This directory actually contains **two independent, unrelated projects** that happen to share the same folder. There is no shared build system, package manifest, or code between them — treat them as separate apps when making changes.

1. **Handwriting OCR web app** (root: `app.py`, `templates/index.html`, root `requirements.txt`) — a Flask + EasyOCR app that recognizes Korean/English/digit handwriting from a canvas drawing or uploaded image.
2. **MNIST digit recognizer (Tkinter GUI)** — exists in **two copies**:
   - `digit_recognition.py` at the repo root: single-file version, auto-trains a model on first run if `digit_model.joblib` is missing.
   - `handwriting-digit-recognizer/`: modularized version (`train_model.py` + `preprocess.py` + `digit_app.py`), requires running `train_model.py` first — it does **not** auto-train.

When editing digit-recognition logic (preprocessing, model params), check whether the fix needs to be applied in one, two, or all three of `digit_recognition.py`, `handwriting-digit-recognizer/preprocess.py`, and `handwriting-digit-recognizer/train_model.py` — they contain duplicated (copy-pasted) logic, not shared imports.

## Commands

There is no test suite, linter, or build tool configured in this repo.

### Handwriting OCR web app (root)
```bash
# Activate the existing venv first
venv\Scripts\Activate.ps1        # PowerShell
source venv/Scripts/activate     # git bash

pip install -r requirements.txt  # flask, easyocr
python app.py                    # serves http://127.0.0.1:5000
```
- First request triggers EasyOCR's Korean+English model download (cached under `~/.EasyOCR/model` afterward).
- To use a GPU, change `easyocr.Reader(["ko", "en"], gpu=False)` in `app.py` to `gpu=True` (requires CUDA-enabled torch).

### MNIST digit recognizer — root single-file version
```bash
python digit_recognition.py
```
or double-click `Run_Digit_Recognition.bat` on Windows. Auto-trains and saves `digit_model.joblib` next to the script if it doesn't exist yet (downloads MNIST via `fetch_openml` on first run).

### MNIST digit recognizer — modular version
```bash
cd handwriting-digit-recognizer
pip install -r requirements.txt   # scikit-learn, numpy, Pillow
python train_model.py             # must run first — no auto-train
python digit_app.py
```

## Architecture notes

### Handwriting OCR web app
- `app.py` is a single Flask app with two routes: `/` renders `templates/index.html`, `/recognize` accepts either a multipart file upload (`request.files["image"]`) or a JSON body with a base64 canvas `dataURL` (`payload["image"]`), decodes it to a PIL image, and runs it through a module-level `easyocr.Reader(["ko", "en"])` loaded once at startup (not per-request — model loading is slow).
- `templates/index.html` contains all frontend logic inline (canvas drawing + file upload UI, no separate JS/CSS build step).
- stdout/stderr are forced to UTF-8 at the top of `app.py` specifically to prevent Windows console encoding crashes from EasyOCR's download progress bar (cp949 console).

### MNIST digit recognizer (both copies)
Both copies share the same pipeline, reimplemented rather than imported:
1. **Canvas capture**: Tkinter canvas paired with an off-screen PIL `Image` (`"L"` mode, black background) kept in sync stroke-by-stroke, so the same pixels drawn on screen can be fed to the model.
2. **Preprocessing** (`canvas_image_to_mnist_vector` in `digit_recognition.py` / `handwriting-digit-recognizer/preprocess.py`): crop tightly to the drawn strokes → resize to 20x20 → pad into a 28x28 canvas → re-center by center-of-mass (via `scipy.ndimage`) → flatten and normalize to a `(1, 784)` float32 vector in `[0, 1]`. This mirrors the classic MNIST normalization pipeline so hand-drawn input matches the training distribution.
3. **Model**: `sklearn.neural_network.MLPClassifier` (hidden layers `(128, 64)`, ReLU, Adam, `max_iter=30` with early stopping), trained on `mnist_784` from `fetch_openml`, persisted with `joblib`.
