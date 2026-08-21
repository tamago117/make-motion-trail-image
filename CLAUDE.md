# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interactive Gradio GUI tool that creates motion-trail composite images using SAM 3 (Segment Anything Model 3). Users click on objects across a sequence of frames to segment them, then generate a single composite showing the object's motion trail over a median-estimated background.

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
  1. **Image utilities** (`load_images`, `generate_background`, `overlay_object_on_background`) — pure NumPy/OpenCV functions for loading frames, computing median backgrounds, and alpha-compositing segmented objects.
  2. **Session persistence** (`save_session`, `load_session`, `list_sessions`) — writes the whole workspace to `sessions/<name>/` (frames and masks as PNG, everything else in `session.json`) and reads it back into app-shaped state. Frames are stored as pixels because neither input source survives a restart (a dropped image folder has no path, a dropped video lives in an upload temp dir). Re-saving is incremental: `session.json` carries a content hash per image, and only images whose hash changed are re-encoded (~40 ms per 1080p PNG vs ~6 ms to hash), with `_prune` deleting files the session no longer needs. This is what keeps autosave-on-generate cheap.
  3. **SAM 3 integration** (`_get_model_and_processor`, `run_predictor_on_frame`) — lazily initializes the SAM 3 model from HuggingFace and runs interactive point-prompt segmentation per frame. The `sam3` package is installed from the Facebook Research GitHub repo.

- **`app.py`** — Gradio GUI and entry point:
  1. **Visualization helpers** (`_draw_points`, `_overlay_mask`) — draw point annotations and mask overlays for the GUI preview.
  2. **Gradio callbacks** — manage per-frame state (points, masks) via `gr.State` objects keyed by frame index. Handle click-to-annotate, undo/clear, frame navigation, composite generation, and session save/restore.
  3. **UI builder** (`build_ui`) — constructs the Gradio Blocks layout and wires up callbacks.

Output format: the composite's format comes from the output path's extension (`_resolve_output_path` validates it, falling back to PNG — `cv2.imwrite` *raises* on an extension it can't encode). `generate_composite` returns the written file's path, not an array, because `gr.Image` re-encodes arrays to its `format=` (webp by default in Gradio 6), which would ignore the chosen format on download.

Key data flow: frames are stored in both RGB (for display/SAM) and BGR (for OpenCV compositing). Per-frame point prompts are stored in `st_points_map` as `dict[int, list[(x, y, label)]]` — the keys are ints, so anything round-tripping them through JSON must convert back. Masks are stored in `st_masks` as `list[np.ndarray | None]`, where `None` (never annotated) is meaningfully different from an all-zero mask.

## Linting and Formatting

You DO NOT need to run any formatting or linting commands manually. The pre-commit hooks will automatically check and format code on commit.

## Requirements

- Python >=3.12 (pinned in `.python-version`)
- GPU with >=8GB VRAM recommended; falls back to CPU
- SAM 3 checkpoint auto-downloaded from HuggingFace on first run
