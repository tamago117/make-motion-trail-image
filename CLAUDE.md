# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interactive Gradio GUI tool that creates motion-trail composite images using SAM 3 (Segment Anything Model 3). Users click on objects across a sequence of frames to segment them, then generate a single composite showing the object's motion trail over a background frame the user picks.

## Commands

```bash
# Install dependencies
uv sync

# Run the app (Gradio server at http://127.0.0.1:7860)
uv run app.py

# Install pre-commit hooks
make setup-hooks

# Run linting/formatting checks (ruff lint + format, plus general file checks)
make check-hooks
```

## Architecture

The application is split into two modules:

- **`core.py`** — framework-independent logic:
  1. **Image utilities** (`load_images`, `overlay_object_on_background`, `compose_multi_set`) — pure NumPy/OpenCV functions for loading frames and alpha-compositing segmented objects.
  2. **Video output** (`compose_multi_set_progressive`, `pace_steps`, `write_video`) — renders the trail growing over time and encodes it (ffmpeg H.264, falling back to OpenCV).
  3. **Session persistence** (`save_session`, `load_session`, `list_sessions`) — saves / restores the whole workspace under `sessions/<name>/`.
  4. **SAM 3 integration** (`_get_model_and_processor`, `run_predictor_on_frame`) — lazily initializes the SAM 3 model from HuggingFace and runs interactive point-prompt segmentation per frame. The `sam3` package is installed from the Facebook Research GitHub repo.

- **`app.py`** — Gradio GUI and entry point:
  1. **Visualization helpers** (`_draw_points`, `_overlay_mask`) — draw point annotations and mask overlays for the GUI preview.
  2. **Gradio callbacks** — manage per-set state (frames, points, masks) held in `st_sets`. Handle click-to-annotate, undo/clear, frame navigation, composite / video generation, and session save/restore.
  3. **UI builder** (`build_ui`) — constructs the Gradio Blocks layout and wires up callbacks.

Key data flow: frames are stored in both RGB (for display/SAM) and BGR (for OpenCV compositing). Each set in `st_sets` holds its point prompts as `points_map: dict[int, list[(x, y, label)]]` and its masks as `masks: list[np.ndarray | None]`, where `None` means never annotated.

## Linting and Formatting

You DO NOT need to run any formatting or linting commands manually. The pre-commit hooks will automatically check and format code on commit.

## Requirements

- Python >=3.12 (pinned in `.python-version`)
- GPU with >=8GB VRAM recommended; falls back to CPU
- SAM 3 checkpoint auto-downloaded from HuggingFace on first run
