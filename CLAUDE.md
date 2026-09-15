# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This repository contains **two independent, unrelated projects**, each in its own top-level folder. There is no shared build system, package manifest, or code between them — treat them as separate apps when making changes. Each folder has its own `CLAUDE.md` with commands and architecture notes specific to that project; read that file when working inside it.

1. **`web_version/`** — Handwriting OCR web app (Flask + EasyOCR) that recognizes Korean/English/digit handwriting from a canvas drawing or uploaded image. See `web_version/CLAUDE.md`.
2. **`desktop_version/`** — MNIST digit recognizer (Tkinter GUI), which itself exists in two copies (a single-file version and a modularized version). See `desktop_version/CLAUDE.md`.

There is no test suite, linter, or build tool configured in this repo.
