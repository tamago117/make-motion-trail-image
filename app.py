#!/usr/bin/env python3
"""
Interactive Gradio GUI for motion-trail image creation using SAM 3.

Workflow
-------
1. Load a directory of frames.
2. For each frame, click to place positive / negative point prompts.
3. SAM 3 segments the object in real time and shows a mask preview.
4. Navigate frames and annotate each independently.
5. Generate the final motion-trail composite.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ---------------------------------------------------------------------------
# Core image utilities (formerly in main.py)
# ---------------------------------------------------------------------------


def load_images(folder: Path) -> Tuple[List[np.ndarray], List[Path]]:
    """
    Load every .png / .jpg / .jpeg in *folder* (non-recursive).

    Returns (frames_bgr, paths) sorted lexicographically.
    """
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    frames = [cv2.imread(str(p)) for p in paths]
    return frames, paths


def generate_background(frames: List[np.ndarray]) -> np.ndarray:
    """Median pixel value across the time dimension -> static background."""
    stack = np.stack(frames, axis=0).astype(np.uint8)
    return np.median(stack, axis=0).astype(np.uint8)


def overlay_object_on_background(
    background: np.ndarray,
    object_layers: List[Tuple[np.ndarray, np.ndarray]],
    alpha: float = 0.5,
) -> np.ndarray:
    """Compose the final image from background + object layers."""
    output = background.copy()
    for idx, (frame, mask) in enumerate(object_layers):
        m = mask.astype(bool)
        if m.sum() == 0:
            continue
        if idx == len(object_layers) - 1:
            output[m] = frame[m]
        else:
            output[m] = (
                (1 - alpha) * output[m].astype(np.float32)
                + alpha * frame[m].astype(np.float32)
            ).astype(np.uint8)
    return output


# ---------------------------------------------------------------------------
# SAM 3 – lazily initialised
# ---------------------------------------------------------------------------
_model = None
_processor: Sam3Processor | None = None


def _get_model_and_processor(device: str = None):
    """Build or return the cached SAM 3 model and processor."""
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Loading SAM 3 model …")
    _model = build_sam3_image_model(
        device=device,
        load_from_HF=True,
        enable_segmentation=True,
        enable_inst_interactivity=True,
    )
    _processor = Sam3Processor(_model, device=device)
    print("[INFO] SAM 3 model loaded")
    return _model, _processor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _draw_points(image: np.ndarray, points: list[tuple[int, int, int]]) -> np.ndarray:
    """Draw coloured circles on *image* for each (x, y, label) tuple."""
    vis = image.copy()
    for x, y, label in points:
        colour = (0, 255, 0) if label == 1 else (255, 0, 0)  # green / red (RGB)
        cv2.circle(vis, (x, y), 6, colour, -1)
        cv2.circle(vis, (x, y), 6, (255, 255, 255), 1)
    return vis


def _overlay_mask(
    image: np.ndarray, mask: np.ndarray, colour=(0, 180, 0), alpha=0.45
) -> np.ndarray:
    """Blend a semi-transparent coloured mask onto *image* (RGB)."""
    vis = image.copy().astype(np.float32)
    overlay = np.full_like(vis, colour, dtype=np.float32)
    m = mask.astype(bool)
    vis[m] = (1 - alpha) * vis[m] + alpha * overlay[m]
    return vis.astype(np.uint8)


def _run_predictor_on_frame(
    rgb: np.ndarray,
    points: list[tuple[int, int, int]],
) -> np.ndarray | None:
    """Run SAM 3 interactive predictor on a single frame.

    Returns (H, W) uint8 mask or None.
    """
    if not points:
        return None
    model, processor = _get_model_and_processor()

    # Encode image through the shared backbone (use PIL to avoid numpy shape bug)
    state = processor.set_image(Image.fromarray(rgb))

    # Run interactive point prediction
    coords = np.array([[x, y] for x, y, _ in points])
    labels = np.array([lab for _, _, lab in points])
    masks, scores, _ = model.predict_inst(
        state,
        point_coords=coords,
        point_labels=labels,
        multimask_output=True,
    )
    # masks: (3, H, W), scores: (3,)
    best = int(np.argmax(scores))
    mask = masks[best].astype(np.uint8)

    # Resize mask to match input image if dimensions differ
    h, w = rgb.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------


def load_dir(input_dir: str):
    """Load images from the given directory and return initial UI state."""
    p = Path(input_dir)
    if not p.is_dir():
        gr.Warning(f"Not a directory: {input_dir}")
        return None, None, [], {}, 0, gr.update(maximum=0), None, None

    frames_bgr, paths = load_images(p)
    if not frames_bgr:
        gr.Warning("No images found in directory")
        return None, None, [], {}, 0, gr.update(maximum=0), None, None

    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    first = frames_rgb[0]

    return (
        first,  # input_image
        first,  # preview_image
        frames_rgb,  # state: frames
        {},  # state: points_map (per-frame)
        0,  # state: current_idx
        gr.update(maximum=max(len(frames_rgb) - 1, 0), value=0),  # frame_slider
        [None] * len(frames_rgb),  # state: masks
        frames_bgr,  # state: frames_bgr
    )


def on_image_click(
    frames: list,
    points_map: dict,
    current_idx: int,
    masks: list,
    evt: gr.SelectData,
    mode: str,
):
    """Handle a click on the image – add point to the current frame only."""
    if not frames:
        return None, None, points_map, masks

    label = 1 if mode == "Positive" else 0
    x, y = evt.index

    pts = points_map.setdefault(current_idx, [])
    pts.append((x, y, label))

    rgb = frames[current_idx]
    mask = _run_predictor_on_frame(rgb, pts)
    masks[current_idx] = mask

    img_with_points = _draw_points(rgb, pts)
    preview = _overlay_mask(rgb, mask) if mask is not None else rgb.copy()
    preview = _draw_points(preview, pts)

    return img_with_points, preview, points_map, masks


def undo_point(frames: list, points_map: dict, current_idx: int, masks: list):
    """Remove the last point for the current frame and re-run prediction."""
    pts = points_map.get(current_idx, [])
    if pts:
        pts.pop()
        if pts:
            rgb = frames[current_idx]
            mask = _run_predictor_on_frame(rgb, pts)
            masks[current_idx] = mask
        else:
            masks[current_idx] = None
    return _current_views(frames, points_map, current_idx, masks)


def clear_points(frames: list, points_map: dict, current_idx: int, masks: list):
    """Clear all points and mask for the current frame."""
    points_map[current_idx] = []
    masks[current_idx] = None
    return _current_views(frames, points_map, current_idx, masks)


def _current_views(frames, points_map, idx, masks):
    """Return (input_image, preview, points_map, masks) for the current frame."""
    if not frames:
        return None, None, points_map, masks
    rgb = frames[idx]
    pts = points_map.get(idx, [])
    img_with_points = _draw_points(rgb, pts)
    mask = masks[idx] if masks else None
    preview = _overlay_mask(rgb, mask) if mask is not None else rgb.copy()
    preview = _draw_points(preview, pts)
    return img_with_points, preview, points_map, masks


def change_frame(
    frames: list,
    points_map: dict,
    masks: list,
    frame_idx: int,
):
    """Switch the displayed frame when the slider moves."""
    idx = int(frame_idx)
    if not frames:
        return None, None, idx
    rgb = frames[idx]
    pts = points_map.get(idx, [])
    img_with_points = _draw_points(rgb, pts)
    mask = masks[idx] if masks and idx < len(masks) else None
    preview = _overlay_mask(rgb, mask) if mask is not None else rgb.copy()
    preview = _draw_points(preview, pts)
    return img_with_points, preview, idx


def generate_composite(
    frames_bgr: list,
    masks: list,
    alpha: float,
    output_path: str,
):
    """Build and save the motion-trail composite."""
    if not frames_bgr or not masks:
        gr.Warning("Load images and generate masks first")
        return None

    object_layers: List[Tuple[np.ndarray, np.ndarray]] = []
    for frame, mask in zip(frames_bgr, masks):
        if mask is not None:
            object_layers.append((frame, mask))

    if not object_layers:
        gr.Warning("No valid masks – cannot generate composite")
        return None

    background = generate_background(frames_bgr)
    composite = overlay_object_on_background(background, object_layers, alpha)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), composite)

    return cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Motion Trail – SAM 3") as demo:
        gr.Markdown("# Motion Trail Image Creator (SAM 3)")

        # ---- state ----
        st_frames = gr.State([])  # list[np.ndarray] RGB
        st_frames_bgr = gr.State([])  # list[np.ndarray] BGR (for compositing)
        st_points_map = gr.State({})  # dict[int, list[(x, y, label)]] per-frame
        st_masks = gr.State([])  # list[np.ndarray | None]
        st_idx = gr.State(0)  # current frame index

        # ---- top row: load ----
        with gr.Row():
            input_dir = gr.Textbox(
                label="Input directory", value="data/samples/", scale=4
            )
            load_btn = gr.Button("Load", scale=1)

        # ---- images ----
        with gr.Row():
            input_image = gr.Image(label="Click to add points", interactive=False)
            preview_image = gr.Image(label="Mask preview", interactive=False)

        # ---- controls ----
        with gr.Row():
            mode_radio = gr.Radio(
                ["Positive", "Negative"],
                value="Positive",
                label="Point mode",
            )
            undo_btn = gr.Button("Undo")
            clear_btn = gr.Button("Clear")

        frame_slider = gr.Slider(
            minimum=0,
            maximum=0,
            step=1,
            value=0,
            label="Frame",
        )

        # ---- composite ----
        with gr.Row():
            alpha_slider = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Alpha")
            out_path = gr.Textbox(
                label="Output path", value="outputs/sample_result.png"
            )
            gen_btn = gr.Button("Generate Motion Trail", variant="primary")
        result_image = gr.Image(label="Result", interactive=False)

        # ---- wiring ----
        load_btn.click(
            load_dir,
            inputs=[input_dir],
            outputs=[
                input_image,
                preview_image,
                st_frames,
                st_points_map,
                st_idx,
                frame_slider,
                st_masks,
                st_frames_bgr,
            ],
        )

        input_image.select(
            on_image_click,
            inputs=[st_frames, st_points_map, st_idx, st_masks, mode_radio],
            outputs=[input_image, preview_image, st_points_map, st_masks],
        )

        undo_btn.click(
            undo_point,
            inputs=[st_frames, st_points_map, st_idx, st_masks],
            outputs=[input_image, preview_image, st_points_map, st_masks],
        )

        clear_btn.click(
            clear_points,
            inputs=[st_frames, st_points_map, st_idx, st_masks],
            outputs=[input_image, preview_image, st_points_map, st_masks],
        )

        frame_slider.change(
            change_frame,
            inputs=[st_frames, st_points_map, st_masks, frame_slider],
            outputs=[input_image, preview_image, st_idx],
        )

        gen_btn.click(
            generate_composite,
            inputs=[st_frames_bgr, st_masks, alpha_slider, out_path],
            outputs=[result_image],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
