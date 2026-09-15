<!-- Created: 2026-09-15 18:44 -->

# CLAUDE.md (web_version)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this folder.

## Overview

Handwriting OCR web app — a Flask + EasyOCR app that recognizes Korean/English/digit handwriting from a canvas drawing or uploaded image.

This is a standalone project: no shared build system or code with `../desktop_version`.

## Commands

There is no test suite, linter, or build tool configured.

```bash
# Activate the existing venv first
venv\Scripts\Activate.ps1        # PowerShell
source venv/Scripts/activate     # git bash

pip install -r requirements.txt  # flask, easyocr
python app.py                    # serves http://127.0.0.1:5000
```

- First request triggers EasyOCR's Korean+English model download (cached under `~/.EasyOCR/model` afterward).
- To use a GPU, change `easyocr.Reader(["ko", "en"], gpu=False)` in `app.py` to `gpu=True` (requires CUDA-enabled torch).

## Architecture notes

- `app.py` is a single Flask app with two routes: `/` renders `templates/index.html`, `/recognize` accepts either a multipart file upload (`request.files["image"]`) or a JSON body with a base64 canvas `dataURL` (`payload["image"]`), decodes it to a PIL image, and runs it through a module-level `easyocr.Reader(["ko", "en"])` loaded once at startup (not per-request — model loading is slow).
- `templates/index.html` contains all frontend logic inline (canvas drawing + file upload UI, no separate JS/CSS build step).
- stdout/stderr are forced to UTF-8 at the top of `app.py` specifically to prevent Windows console encoding crashes from EasyOCR's download progress bar (cp949 console).
