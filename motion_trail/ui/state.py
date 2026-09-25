"""Per-set state and the preview drawing shared by the callbacks."""

from __future__ import annotations

import re

import cv2
import gradio as gr
import numpy as np


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


# Distinct default colours assigned to new sets (RGB).
PALETTE_RGB = [
    (255, 64, 64),  # red
    (64, 128, 255),  # blue
    (64, 200, 96),  # green
    (255, 176, 32),  # orange
    (192, 64, 255),  # purple
    (0, 200, 200),  # cyan
    (255, 96, 160),  # pink
    (160, 160, 64),  # olive
]


def _next_color(n: int) -> tuple[int, int, int]:
    """Pick a distinct palette colour for the n-th set (cycles if needed)."""
    return PALETTE_RGB[n % len(PALETTE_RGB)]


def _new_set(color: tuple[int, int, int]) -> dict:
    """Create an empty set record."""
    return {
        "dir": "",  # source folder
        "frames": [],  # RGB frames (display + SAM)
        "frames_bgr": [],  # BGR frames (compositing)
        "points_map": {},  # dict[int, list[(x, y, label)]]
        "masks": [],  # list[np.ndarray | None]
        "color": color,  # (R, G, B)
        "extract": None,  # video extraction params, None for an image folder
    }


def _set_choices(sets: list) -> list[str]:
    """Radio labels for the current sets."""
    return [f"Set {i + 1}" for i in range(len(sets))]


def _label_to_index(label, sets: list) -> int:
    """Map a selector label back to its set index."""
    choices = _set_choices(sets)
    if label in choices:
        return choices.index(label)
    m = re.match(r"Set (\d+)", str(label or ""))
    if m:
        return min(max(int(m.group(1)) - 1, 0), max(len(sets) - 1, 0))
    return 0


def _rgb_to_hex(color: tuple[int, int, int]) -> str:
    """(R, G, B) -> '#rrggbb' for the colour picker."""
    r, g, b = color
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _picker_hex(color, idx: int) -> str:
    """Colour-picker hex for a set (placeholder palette colour if 'no colour')."""
    return _rgb_to_hex(color if color is not None else _next_color(idx))


def _parse_color(value) -> tuple[int, int, int] | None:
    """Parse a picker value ('#rrggbb' or 'rgb(...)') to (R, G, B)."""
    if not value:
        return None
    value = str(value).strip()
    if value.startswith("#"):
        h = value.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) >= 6:
            return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
        return None
    if value.startswith("rgb"):
        nums = re.findall(r"[\d.]+", value)
        if len(nums) >= 3:
            return tuple(int(round(float(n))) for n in nums[:3])
    return None


def _extract_updates(s: dict):
    """Widget updates showing how the active set's frames were extracted.

    Start / End / Interval are shared by every set but describe one extraction,
    so switching sets repoints them at that set's own values. A set with none
    recorded — an image folder, or a session saved before they were kept —
    leaves the widgets alone rather than inventing numbers.
    """
    ex = s.get("extract")
    if not ex:
        return gr.update(), gr.update(), gr.update()
    return (
        gr.update(value=ex["start_sec"]),
        gr.update(value=ex["end_sec"]),
        gr.update(value=ex["interval_sec"]),
    )


def _current_views(frames, points_map, idx, masks):
    """Return (input_image, preview, points_map, masks) for the frame *idx*."""
    if not frames:
        return None, None, points_map, masks
    rgb = frames[idx]
    pts = points_map.get(idx, [])
    img_with_points = _draw_points(rgb, pts)
    mask = masks[idx] if masks and idx < len(masks) else None
    preview = _overlay_mask(rgb, mask) if mask is not None else rgb.copy()
    preview = _draw_points(preview, pts)
    return img_with_points, preview, points_map, masks
