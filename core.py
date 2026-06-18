"""
Core image utilities and SAM 3 integration for motion-trail image creation.

This module contains framework-independent logic:
- Image loading and background estimation
- Alpha-compositing of segmented object layers
- SAM 3 model management and per-frame interactive segmentation
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def load_images(folder: Path) -> Tuple[List[np.ndarray], List[Path]]:
    """
    Load every .png / .jpg / .jpeg in *folder* (non-recursive).

    Returns (frames_bgr, paths) sorted lexicographically.
    """
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    frames = [cv2.imread(str(p)) for p in paths]
    if frames:
        target_h, target_w = frames[0].shape[:2]
        frames = [
            cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA)
            if f.shape[:2] != (target_h, target_w)
            else f
            for f in frames
        ]
    return frames, paths


def generate_background(frames: List[np.ndarray]) -> np.ndarray:
    """Median pixel value across the time dimension -> static background."""
    stack = np.stack(frames, axis=0).astype(np.uint8)
    return np.median(stack, axis=0).astype(np.uint8)


def overlay_object_on_background(
    background: np.ndarray,
    object_layers: List[Tuple[np.ndarray, np.ndarray]],
    alpha: float = 0.5,
    last_opaque: bool = True,
    opaque_indices: set | None = None,
) -> np.ndarray:
    """Compose the final image from background + object layers.

    Layers are alpha-blended onto the running output so the motion trail fades,
    except those whose index is in *opaque_indices*, which are painted opaque
    (the object's "position" rendered solid). When *opaque_indices* is None it is
    derived from *last_opaque*: ``{len-1}`` if True (the final frame, as before),
    else empty — so existing single-set behaviour is unchanged.
    """
    if opaque_indices is None:
        opaque_indices = {len(object_layers) - 1} if last_opaque else set()
    output = background.copy()
    for idx, (frame, mask) in enumerate(object_layers):
        m = mask.astype(bool)
        if m.sum() == 0:
            continue
        if idx in opaque_indices:
            output[m] = frame[m]
        else:
            output[m] = (
                (1 - alpha) * output[m].astype(np.float32)
                + alpha * frame[m].astype(np.float32)
            ).astype(np.uint8)
    return output


def tint(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: Tuple[int, int, int],
    strength: float = 0.5,
) -> np.ndarray:
    """Blend the masked object pixels toward *color_bgr*.

    ``strength=0`` keeps the original object colours, ``strength=1`` turns the
    object into a flat colour silhouette. Operates in BGR (compositing space).
    """
    out = frame_bgr.copy()
    m = mask.astype(bool)
    if m.sum() == 0:
        return out
    color = np.array(color_bgr, dtype=np.float32)
    out[m] = (
        (1 - strength) * frame_bgr[m].astype(np.float32) + strength * color
    ).astype(np.uint8)
    return out


def resize_to_canvas(
    frame_bgr: np.ndarray,
    mask: np.ndarray | None,
    size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Resize a frame (and its mask) to the shared canvas *size* = (H, W).

    Frames use INTER_AREA; masks use INTER_NEAREST to stay binary. Lets sets of
    differing dimensions be composited onto one common canvas.
    """
    h, w = size
    if frame_bgr.shape[:2] != (h, w):
        frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    if mask is not None and mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return frame_bgr, mask


def compose_multi_set(
    sets: List[dict],
    background_bgr: np.ndarray,
    alpha: float = 0.5,
    tint_strength: float = 0.5,
    emphasis: str = "last",
) -> np.ndarray:
    """Overlay several colored motion trails onto one chosen background.

    *sets* is a list of dicts, each with ``frames_bgr`` (list), ``masks`` (list
    aligned with frames, entries may be None) and ``color_bgr`` ((B, G, R) or
    None to skip tinting and keep the object's original colours). Each set is
    tinted with its colour and layered onto the running output in order, so
    overlapping sets blend where their masks meet.

    *emphasis* selects which frames of each set are painted opaque (the rest
    fade via *alpha*): ``"none"`` (all blended), ``"last"`` (final frame, the
    default) or ``"first_last"`` (first and final frames).
    """
    size = background_bgr.shape[:2]
    output = background_bgr.copy()
    for s in sets:
        layers: List[Tuple[np.ndarray, np.ndarray]] = []
        for frame, mask in zip(s["frames_bgr"], s["masks"]):
            if mask is None:
                continue
            frame, mask = resize_to_canvas(frame, mask, size)
            color = s.get("color_bgr")
            if color is not None:
                frame = tint(frame, mask, color, tint_strength)
            layers.append((frame, mask))
        if not layers:
            continue
        n = len(layers)
        if emphasis == "none":
            opaque = set()
        elif emphasis == "first_last":
            opaque = {0, n - 1}
        else:  # "last"
            opaque = {n - 1}
        output = overlay_object_on_background(
            output, layers, alpha, opaque_indices=opaque
        )
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


def run_predictor_on_frame(
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
