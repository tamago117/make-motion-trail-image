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


# Video container formats handled by :func:`load_video`.
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _resize_to_first(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Resize every frame to match the first one's (H, W)."""
    if not frames:
        return frames
    h, w = frames[0].shape[:2]
    return [
        cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        if f.shape[:2] != (h, w)
        else f
        for f in frames
    ]


def load_video(
    path: Path,
    start_sec: float = 0.0,
    end_sec: float = 0.0,
    interval_sec: float = 1.0,
) -> List[np.ndarray]:
    """Extract one BGR frame every *interval_sec* seconds from a video.

    Frames are sampled across the ``[start_sec, end_sec]`` interval (in seconds);
    ``end_sec <= 0`` means "until the end". Returns frames resized to the first
    extracted frame's dimensions, or an empty list if the video can't be read.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []

    interval_sec = max(float(interval_sec), 1e-3)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total > 0:
        # Seekable path: jump directly to the chosen frame indices.
        indices = _interval_indices(start_sec, end_sec, interval_sec, fps, total)
        frames = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        cap.release()
        return _resize_to_first(frames)

    # Frame count unknown (some codecs): read sequentially, then sample.
    all_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        all_frames.append(frame)
    cap.release()
    if not all_frames:
        return []
    indices = _interval_indices(start_sec, end_sec, interval_sec, fps, len(all_frames))
    return _resize_to_first([all_frames[i] for i in indices])


def _interval_indices(
    start_sec: float,
    end_sec: float,
    interval_sec: float,
    fps: float,
    total: int,
) -> List[int]:
    """Frame indices at *interval_sec* steps across [start_sec, end_sec]."""
    duration = total / fps
    end_time = end_sec if (end_sec and end_sec > 0) else duration
    end_time = min(end_time, duration)
    start_time = max(min(start_sec, end_time), 0.0)

    indices: List[int] = []
    t = start_time
    while t <= end_time + 1e-9:
        fi = min(max(int(round(t * fps)), 0), total - 1)
        if not indices or fi != indices[-1]:
            indices.append(fi)
        t += interval_sec
    if not indices:
        indices = [min(max(int(round(start_time * fps)), 0), total - 1)]
    return indices


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
