#!/usr/bin/env python3
"""
Interactive Gradio GUI for motion-trail image creation using SAM 3.

Workflow
-------
1. Load a directory of frames into a "set".
2. For each frame, click to place positive / negative point prompts.
3. SAM 3 segments the object in real time and shows a mask preview.
4. Navigate frames and annotate each independently.
5. Add more sets (+), each annotated separately and given its own colour.
6. Choose one frame as the background, then generate a composite that
   overlays every set's motion trail in its own colour.
"""

from __future__ import annotations

import hashlib
import platform
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from core import (
    VIDEO_EXTS,
    compose_multi_set,
    load_images,
    load_video,
    run_predictor_on_frame,
)

# ---------------------------------------------------------------------------
# Visualisation helpers
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


# ---------------------------------------------------------------------------
# Multi-set helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Directory browser
# ---------------------------------------------------------------------------


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def _parse_time(value) -> float:
    """Parse a time spec to seconds.

    Accepts plain seconds (``"12.5"``), ``"m:s"`` (``"1:23.5"`` -> 83.5) and
    ``"h:m:s"`` (``"1:02:03"`` -> 3723), all with optional decimals. Blank or
    unparseable input returns ``0.0``.
    """
    if value is None:
        return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    try:
        parts = [float(p) for p in s.split(":")]
    except ValueError:
        gr.Warning(f"Invalid time: {value}")
        return 0.0
    total = 0.0
    for p in parts:  # left-to-right: each field is 60× the next (h, m, s)
        total = total * 60 + p
    return max(total, 0.0)


def _native_picker(pick_file: bool):
    """Open a native OS picker for a directory (or a file) and return the path."""
    system = platform.system()
    if system == "Linux" and _is_wsl():
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$f = New-Object System.Windows.Forms.OpenFileDialog;"
            "$f.ShowDialog() | Out-Null;$f.FileName"
            if pick_file
            else "Add-Type -AssemblyName System.Windows.Forms;"
            "$f = New-Object System.Windows.Forms.FolderBrowserDialog;"
            "$f.ShowDialog() | Out-Null;$f.SelectedPath"
        )
        result = subprocess.run(
            ["powershell.exe", "-Command", ps],
            capture_output=True,
            text=True,
        )
        path = result.stdout.strip()
        if path:
            wsl = subprocess.run(
                ["wslpath", "-u", path],
                capture_output=True,
                text=True,
            )
            path = wsl.stdout.strip()
    elif system == "Darwin":
        script = "POSIX path of (choose file)" if pick_file else (
            "POSIX path of (choose folder)"
        )
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
        )
        path = result.stdout.strip()
    else:
        cmd = ["zenity", "--file-selection", "--title=Select"]
        if not pick_file:
            cmd.append("--directory")
        result = subprocess.run(cmd, capture_output=True, text=True)
        path = result.stdout.strip()
    return path if path else gr.update()


def browse_dir():
    """Open a native OS directory picker and return the selected path."""
    return _native_picker(pick_file=False)


# Codecs browsers can play directly, so no transcode is needed.
_BROWSER_CODECS = {"h264", "avc1", "vp8", "vp9", "av1"}


def _playable_video(path: str):
    """Return a browser-playable version of *path* for the gr.Video widget.

    Browser-compatible files are served as-is. Others (e.g. mpeg4) are
    transcoded to H.264 mp4 in the system temp dir, so Gradio doesn't try to
    convert them in place (which fails when the source dir isn't writable).
    Returns the path string, or ``gr.update()`` if it can't be prepared.
    """
    src = Path(path)
    if not src.is_file():
        return gr.update()

    codec = ""
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1", str(src),
            ],
            capture_output=True,
            text=True,
        )
        codec = probe.stdout.strip()
    except OSError:
        pass

    if codec in _BROWSER_CODECS and src.suffix.lower() in {".mp4", ".webm"}:
        return str(src)

    out_dir = Path(tempfile.gettempdir()) / "motion_trail_preview"
    out_dir.mkdir(exist_ok=True)
    # Key the cache on the absolute path + mtime + size so different videos that
    # happen to share a filename (e.g. several "overhead.mp4") never collide.
    stat = src.stat()
    key = hashlib.md5(
        f"{src.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode()
    ).hexdigest()
    out = out_dir / f"{key}.mp4"
    if out.is_file():
        return str(out)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-movflags", "+faststart", str(out),
            ],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        gr.Warning("Could not prepare the video for in-browser playback")
        return gr.update()
    return str(out)


def browse_file():
    """Open a file picker; return the path for both the textbox and the player."""
    path = _native_picker(pick_file=True)
    if isinstance(path, str):
        # textbox gets the real path; the player gets a browser-playable copy
        return path, _playable_video(path)
    return gr.update(), gr.update()  # cancelled: leave both unchanged


# ---------------------------------------------------------------------------
# Set management callbacks
# ---------------------------------------------------------------------------


def add_set(sets: list):
    """Append a new empty set, select it, and clear the workspace."""
    color = _next_color(len(sets))
    sets = sets + [_new_set(color)]
    active = len(sets) - 1
    return (
        sets,  # st_sets
        active,  # st_active
        0,  # st_idx
        gr.update(choices=_set_choices(sets), value=f"Set {active + 1}"),
        None,  # input_image
        None,  # preview_image
        gr.update(maximum=0, value=0),  # frame_slider
        _rgb_to_hex(color),  # color_picker
        False,  # no_color_checkbox
        gr.update(value=""),  # input_dir
    )


def remove_set(sets: list, active: int):
    """Remove the active set; keep at least one set."""
    sets = list(sets)
    if 0 <= active < len(sets):
        sets.pop(active)
    if not sets:
        sets = [_new_set(_next_color(0))]
        active = 0
    else:
        active = min(active, len(sets) - 1)

    s = sets[active]
    if s["frames"]:
        img, preview, _, _ = _current_views(s["frames"], s["points_map"], 0, s["masks"])
        slider = gr.update(maximum=max(len(s["frames"]) - 1, 0), value=0)
    else:
        img, preview = None, None
        slider = gr.update(maximum=0, value=0)

    return (
        sets,
        active,
        0,
        gr.update(choices=_set_choices(sets), value=f"Set {active + 1}"),
        img,
        preview,
        slider,
        _picker_hex(s["color"], active),
        s["color"] is None,
        gr.update(value=s["dir"]),
    )


def select_set(sets: list, label):
    """Switch the active set and repaint the workspace from its state."""
    active = _label_to_index(label, sets)
    s = sets[active]
    if s["frames"]:
        img, preview, _, _ = _current_views(s["frames"], s["points_map"], 0, s["masks"])
        slider = gr.update(maximum=max(len(s["frames"]) - 1, 0), value=0)
    else:
        img, preview = None, None
        slider = gr.update(maximum=0, value=0)
    return (
        active,  # st_active
        0,  # st_idx
        img,  # input_image
        preview,  # preview_image
        slider,  # frame_slider
        _picker_hex(s["color"], active),  # color_picker
        s["color"] is None,  # no_color_checkbox
        gr.update(value=s["dir"]),  # input_dir
    )


def set_color(sets: list, active: int, value):
    """Store a user-picked colour on the active set (clears 'no colour')."""
    rgb = _parse_color(value)
    if rgb is not None and 0 <= active < len(sets):
        sets[active]["color"] = rgb
    return sets, False  # picking a colour implies the set is coloured


def toggle_no_color(sets: list, active: int, no_color: bool, picker_value):
    """Toggle tinting for the active set; 'no colour' keeps original pixels."""
    if 0 <= active < len(sets):
        if no_color:
            sets[active]["color"] = None
        else:
            sets[active]["color"] = _parse_color(picker_value) or _next_color(active)
    return sets


def set_background(sets: list, active: int, idx: int):
    """Capture the active set's current frame as the composite background."""
    if not (0 <= active < len(sets)):
        return None, None
    s = sets[active]
    if not s["frames_bgr"]:
        gr.Warning("Load images for this set first")
        return None, None
    idx = int(idx)
    if idx >= len(s["frames_bgr"]):
        idx = 0
    bg_bgr = s["frames_bgr"][idx].copy()
    bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
    return bg_bgr, bg_rgb


# ---------------------------------------------------------------------------
# Annotation callbacks (operate on the active set)
# ---------------------------------------------------------------------------


def load_source(
    input_dir: str,
    sets: list,
    active: int,
    start_sec: str,
    end_sec: str,
    interval_sec: float,
):
    """Load frames from *input_dir* (a folder of images or a video) into the set."""
    p = Path(input_dir)
    video_out = None  # path to feed the playback widget (None clears it)
    if p.is_dir():
        frames_bgr, _ = load_images(p)
        if not frames_bgr:
            gr.Warning("No images found in directory")
            return None, None, gr.update(), 0, sets, gr.update()
    elif p.is_file() and p.suffix.lower() in VIDEO_EXTS:
        frames_bgr = load_video(
            p,
            _parse_time(start_sec),
            _parse_time(end_sec),
            float(interval_sec or 1.0),
        )
        if not frames_bgr:
            gr.Warning("Could not read frames from video")
            return None, None, gr.update(), 0, sets, gr.update()
        video_out = _playable_video(str(p))
    else:
        gr.Warning(f"Not a directory or supported video file: {input_dir}")
        return None, None, gr.update(), 0, sets, gr.update()

    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]

    if not sets:
        sets = [_new_set(_next_color(0))]
        active = 0
    s = sets[active]
    s["dir"] = input_dir
    s["frames"] = frames_rgb
    s["frames_bgr"] = frames_bgr
    s["points_map"] = {}
    s["masks"] = [None] * len(frames_rgb)

    first = frames_rgb[0]
    return (
        first,  # input_image
        first,  # preview_image
        gr.update(maximum=max(len(frames_rgb) - 1, 0), value=0),  # frame_slider
        0,  # st_idx
        sets,  # st_sets
        video_out,  # video_player (path for a movie, None for an image folder)
    )


def on_image_click(
    sets: list,
    active: int,
    current_idx: int,
    evt: gr.SelectData,
    mode: str,
):
    """Add a point to the active set's current frame and re-run SAM 3."""
    if not (0 <= active < len(sets)):
        return None, None, sets
    s = sets[active]
    if not s["frames"]:
        return None, None, sets

    label = 1 if mode == "Positive" else 0
    x, y = evt.index

    pts = s["points_map"].setdefault(current_idx, [])
    pts.append((x, y, label))

    rgb = s["frames"][current_idx]
    mask = run_predictor_on_frame(rgb, pts)
    s["masks"][current_idx] = mask

    img_with_points = _draw_points(rgb, pts)
    preview = _overlay_mask(rgb, mask) if mask is not None else rgb.copy()
    preview = _draw_points(preview, pts)

    return img_with_points, preview, sets


def undo_point(sets: list, active: int, current_idx: int):
    """Remove the last point for the active set's current frame."""
    if not (0 <= active < len(sets)):
        return None, None, sets
    s = sets[active]
    pts = s["points_map"].get(current_idx, [])
    if pts:
        pts.pop()
        if pts:
            rgb = s["frames"][current_idx]
            s["masks"][current_idx] = run_predictor_on_frame(rgb, pts)
        else:
            s["masks"][current_idx] = None
    img, preview, _, _ = _current_views(
        s["frames"], s["points_map"], current_idx, s["masks"]
    )
    return img, preview, sets


def clear_points(sets: list, active: int, current_idx: int):
    """Clear all points and the mask for the active set's current frame."""
    if not (0 <= active < len(sets)):
        return None, None, sets
    s = sets[active]
    s["points_map"][current_idx] = []
    if current_idx < len(s["masks"]):
        s["masks"][current_idx] = None
    img, preview, _, _ = _current_views(
        s["frames"], s["points_map"], current_idx, s["masks"]
    )
    return img, preview, sets


def change_frame(sets: list, active: int, frame_idx: int):
    """Switch the displayed frame when the slider moves."""
    idx = int(frame_idx)
    if not (0 <= active < len(sets)) or not sets[active]["frames"]:
        return None, None, idx
    s = sets[active]
    img, preview, _, _ = _current_views(s["frames"], s["points_map"], idx, s["masks"])
    return img, preview, idx


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


EMPHASIS_MODES = {
    "None": "none",
    "Last frame": "last",
    "First & last frames": "first_last",
}


def generate_composite(
    sets: list,
    background,
    alpha: float,
    tint_strength: float,
    emphasis_label: str,
    output_path: str,
):
    """Overlay every annotated set's trail onto the chosen background."""
    usable = [
        s for s in sets if s["frames_bgr"] and any(m is not None for m in s["masks"])
    ]
    if not usable:
        gr.Warning("No sets with masks – annotate at least one set first")
        return None

    if background is None:
        background = usable[0]["frames_bgr"][0]

    payload = [
        {
            "frames_bgr": s["frames_bgr"],
            "masks": s["masks"],
            # RGB -> BGR, or None to keep the object's original colours
            "color_bgr": None if s["color"] is None else tuple(s["color"][::-1]),
        }
        for s in usable
    ]
    composite = compose_multi_set(
        payload,
        background,
        alpha=alpha,
        tint_strength=tint_strength,
        emphasis=EMPHASIS_MODES.get(emphasis_label, "last"),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), composite)

    return cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    css = (
        "#browse-or{flex:0 0 auto !important;min-width:0 !important;"
        "display:flex;align-items:center;justify-content:center;}"
        "#browse-or p{margin:0;}"
    )
    with gr.Blocks(title="Motion Trail – SAM 3", css=css) as demo:
        gr.Markdown("# Motion Trail Image Creator (SAM 3)")
        gr.Markdown(
            "Load a folder per set, annotate each set, give it a colour, "
            "pick a background frame, then overlay every trail."
        )

        init_color = _next_color(0)

        # ---- state ----
        st_sets = gr.State([_new_set(init_color)])  # list[set dict]
        st_active = gr.State(0)  # active set index
        st_idx = gr.State(0)  # current frame within active set
        st_bg = gr.State(None)  # chosen background frame (BGR)

        # ---- set management ----
        with gr.Row():
            set_selector = gr.Radio(
                choices=["Set 1"], value="Set 1", label="Active set", scale=4
            )
            add_btn = gr.Button("+ Add Set", scale=1)
            remove_btn = gr.Button("Remove Set", scale=1)

        # ---- load ----
        with gr.Row():
            input_dir = gr.Textbox(
                label="Input directory or video file (active set)",
                value="data/samples/",
                scale=4,
            )
            browse_btn = gr.Button("Image Directory", scale=1)
            gr.Markdown("or", elem_id="browse-or")
            browse_file_btn = gr.Button("Movie File", scale=1)
            load_btn = gr.Button("Load", scale=1)

        with gr.Row():
            start_sec = gr.Textbox(
                label="Start (video)",
                value="0",
                placeholder="sec or mm:ss.s, e.g. 1:23.5",
            )
            end_sec = gr.Textbox(
                label="End (0 = until end)",
                value="0",
                placeholder="sec or mm:ss.s, e.g. 2:05",
            )
            interval_sec = gr.Number(
                label="Interval (sec, video)",
                value=1.0,
                minimum=0.01,
            )

        # Raw playback to review the movie and pick the start / end interval.
        video_player = gr.Video(label="Movie preview", interactive=False)

        with gr.Row():
            color_picker = gr.ColorPicker(
                label="Set colour", value=_rgb_to_hex(init_color)
            )
            no_color_checkbox = gr.Checkbox(
                label="No colour (keep original)", value=False
            )

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

        # ---- background ----
        with gr.Row():
            bg_btn = gr.Button("Use current frame as background")
            bg_preview = gr.Image(label="Background", interactive=False)

        # ---- composite ----
        with gr.Row():
            alpha_slider = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Alpha")
            tint_slider = gr.Slider(
                0.0, 1.0, value=0.5, step=0.05, label="Tint strength"
            )
            emphasis_radio = gr.Radio(
                ["None", "Last frame", "First & last frames"],
                value="Last frame",
                label="Emphasize (opaque) frames",
            )
            out_path = gr.Textbox(
                label="Output path", value="outputs/sample_result.png"
            )
            gen_btn = gr.Button("Generate Motion Trail", variant="primary")
        result_image = gr.Image(label="Result", interactive=False)

        # ---- wiring ----
        # User-only events (.input / .release) so programmatic updates from
        # add/remove/select/load do not re-trigger the same handlers.
        set_selector.input(
            select_set,
            inputs=[st_sets, set_selector],
            outputs=[
                st_active,
                st_idx,
                input_image,
                preview_image,
                frame_slider,
                color_picker,
                no_color_checkbox,
                input_dir,
            ],
        )

        add_btn.click(
            add_set,
            inputs=[st_sets],
            outputs=[
                st_sets,
                st_active,
                st_idx,
                set_selector,
                input_image,
                preview_image,
                frame_slider,
                color_picker,
                no_color_checkbox,
                input_dir,
            ],
        )

        remove_btn.click(
            remove_set,
            inputs=[st_sets, st_active],
            outputs=[
                st_sets,
                st_active,
                st_idx,
                set_selector,
                input_image,
                preview_image,
                frame_slider,
                color_picker,
                no_color_checkbox,
                input_dir,
            ],
        )

        color_picker.input(
            set_color,
            inputs=[st_sets, st_active, color_picker],
            outputs=[st_sets, no_color_checkbox],
        )

        no_color_checkbox.input(
            toggle_no_color,
            inputs=[st_sets, st_active, no_color_checkbox, color_picker],
            outputs=[st_sets],
        )

        browse_btn.click(browse_dir, inputs=[], outputs=[input_dir])
        browse_file_btn.click(
            browse_file, inputs=[], outputs=[input_dir, video_player]
        )

        load_btn.click(
            load_source,
            inputs=[input_dir, st_sets, st_active, start_sec, end_sec, interval_sec],
            outputs=[
                input_image,
                preview_image,
                frame_slider,
                st_idx,
                st_sets,
                video_player,
            ],
        )

        input_image.select(
            on_image_click,
            inputs=[st_sets, st_active, st_idx, mode_radio],
            outputs=[input_image, preview_image, st_sets],
        )

        undo_btn.click(
            undo_point,
            inputs=[st_sets, st_active, st_idx],
            outputs=[input_image, preview_image, st_sets],
        )

        clear_btn.click(
            clear_points,
            inputs=[st_sets, st_active, st_idx],
            outputs=[input_image, preview_image, st_sets],
        )

        frame_slider.release(
            change_frame,
            inputs=[st_sets, st_active, frame_slider],
            outputs=[input_image, preview_image, st_idx],
        )

        bg_btn.click(
            set_background,
            inputs=[st_sets, st_active, st_idx],
            outputs=[st_bg, bg_preview],
        )

        gen_btn.click(
            generate_composite,
            inputs=[
                st_sets,
                st_bg,
                alpha_slider,
                tint_slider,
                emphasis_radio,
                out_path,
            ],
            outputs=[result_image],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    # Allow the movie-preview widget to serve videos the user browses to from
    # anywhere on the machine (this is a local, single-user tool on 127.0.0.1).
    demo.launch(allowed_paths=["/"])
