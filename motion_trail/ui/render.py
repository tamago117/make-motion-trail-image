"""Callbacks for generating the composite / video and saving / restoring sessions."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import gradio as gr

from motion_trail.compose import (
    compose_multi_set,
    compose_multi_set_progressive,
    pace_steps,
)
from motion_trail.session import (
    default_session_name,
    list_sessions,
    load_session,
    save_session,
)
from motion_trail.ui.edit import _playable_video
from motion_trail.ui.state import (
    _current_views,
    _new_set,
    _next_color,
    _picker_hex,
    _set_choices,
)
from motion_trail.video import VIDEO_OUT_EXTS, write_video


EMPHASIS_MODES = {
    "None": "none",
    "Last frame": "last",
    "First & last frames": "first_last",
}

# Extensions OpenCV can encode; the output path's suffix picks the format.
OUTPUT_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
# Of those, the ones a browser can display inline in the Result panel.
BROWSER_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _resolve_output_path(output_path: str, exts: set, fallback: str) -> Path:
    """Path to write to, in one of the *exts* formats the encoder can produce.

    A blank or unsupported suffix falls back to *fallback*'s format (with a
    warning naming the file actually written) rather than failing after the
    user has waited for the whole render — ``cv2.imwrite`` raises on an
    extension it can't encode, and ffmpeg refuses an unknown container.
    """
    out = Path(str(output_path or "").strip() or fallback)
    if not out.name:
        out = Path(fallback)
    ext = out.suffix.lower()
    if ext not in exts:
        out = out.with_suffix(Path(fallback).suffix)
        gr.Warning(
            f"Unsupported output format '{ext or '(none)'}' – saved as {out.name}"
        )
    return out


def _trail_payload(sets: list, background, default_interval: float = 1.0):
    """core-shaped sets + the background to draw them on, or (None, None).

    Sets without a single mask are dropped, and if the user never picked a
    background frame the first annotated set's first frame stands in. Note the
    fallback follows the set order, so rearranging the sets changes it.

    Each set carries the interval its frames were extracted at, which is what
    puts the video on a real timeline; *default_interval* covers a set with
    none recorded (an image folder, or a session saved before they were kept).
    """
    usable = [
        s for s in sets if s["frames_bgr"] and any(m is not None for m in s["masks"])
    ]
    if not usable:
        gr.Warning("No sets with masks – annotate at least one set first")
        return None, None

    if background is None:
        background = usable[0]["frames_bgr"][0]

    payload = [
        {
            "frames_bgr": s["frames_bgr"],
            "masks": s["masks"],
            # RGB -> BGR, or None to keep the object's original colours
            "color_bgr": None if s["color"] is None else tuple(s["color"][::-1]),
            "interval_sec": (s.get("extract") or {}).get(
                "interval_sec", default_interval
            ),
        }
        for s in usable
    ]
    return payload, background


def generate_composite(
    sets: list,
    background,
    alpha: float,
    tint_strength: float,
    emphasis_label: str,
    output_path: str,
):
    """Overlay every annotated set's trail onto the chosen background."""
    payload, background = _trail_payload(sets, background)
    if payload is None:
        return None

    composite = compose_multi_set(
        payload,
        background,
        alpha=alpha,
        tint_strength=tint_strength,
        emphasis=EMPHASIS_MODES.get(emphasis_label, "last"),
    )

    out = _resolve_output_path(
        output_path, OUTPUT_EXTS, DEFAULT_SETTINGS["output_path"]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        written = cv2.imwrite(str(out), composite)
    except cv2.error as exc:
        gr.Warning(f"Could not write {out}: {exc}")
        return cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
    if not written:
        gr.Warning(f"Could not write {out}")
        return cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)

    gr.Info(f"Saved {out}")
    # Hand the Result panel the file itself, so it downloads in exactly the
    # format that was written; a numpy array would be re-encoded (as webp by
    # default). Formats a browser can't display fall back to the pixels — the
    # file on disk is still in the requested format either way.
    if out.suffix.lower() in BROWSER_EXTS:
        return str(out.resolve())  # absolute: Gradio serves it regardless of cwd
    return cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)


def generate_video(
    sets: list,
    background,
    alpha: float,
    tint_strength: float,
    emphasis_label: str,
    video_output_path: str,
    fps: float,
    interval_sec: float,
):
    """Render the trail as a video that grows on the source's timeline.

    Step *t* holds every set's trail up to frame *t*, so the object walks
    across the background leaving its fading trail behind; the video's final
    frame is exactly the still composite the Generate button produces from the
    same settings.

    **Each set starts at 0 s on its own first annotated frame**, so trails
    picked out at different points of different videos all begin together.
    From there a step is held for the interval that set's frames were sampled
    at, so the trail grows at the speed the object actually moved and an
    unannotated stretch shows up as a pause rather than being skipped. *fps*
    only picks how smoothly that timeline is encoded.

    *interval_sec* covers sets with no interval of their own — an image folder,
    or a session saved before the parameters were kept per set — where it is
    simply seconds per frame.
    """
    payload, background = _trail_payload(sets, background, interval_sec)
    if payload is None:
        return None

    out = _resolve_output_path(
        video_output_path, VIDEO_OUT_EXTS, DEFAULT_SETTINGS["video_output_path"]
    )
    steps = compose_multi_set_progressive(
        payload,
        background,
        alpha=alpha,
        tint_strength=tint_strength,
        emphasis=EMPHASIS_MODES.get(emphasis_label, "last"),
    )
    try:
        codec = write_video(pace_steps(steps, fps), out, fps)
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        gr.Warning(f"Could not write {out}: {exc}")
        return None

    if codec != "h264":
        gr.Warning(
            f"ffmpeg not found – wrote {out.name} as mpeg4, which desktop "
            "players handle but the preview below cannot show"
        )
    gr.Info(f"Saved {out}")
    return str(out.resolve())  # absolute: Gradio serves it regardless of cwd


# Single source of truth for the settings widgets' initial values; also the
# fallback when a restored session predates one of them.
DEFAULT_SETTINGS = {
    "start_sec": "0",
    "end_sec": "0",
    "interval_sec": 1.0,
    "alpha": 0.7,
    "tint_strength": 0.5,
    "emphasis": "Last frame",
    "output_path": "outputs/sample_result.png",
    "video_output_path": "outputs/sample_result.mp4",
    "video_fps": 30.0,
}

# Number of outputs restore_session_cb feeds back into the UI.
_RESTORE_OUTPUTS = 23


def _no_restore():
    """Leave every restore output untouched (used on error paths)."""
    return tuple(gr.update() for _ in range(_RESTORE_OUTPUTS))


def save_session_cb(
    sets: list,
    active: int,
    idx: int,
    background,
    video_path,
    name: str,
    start_sec: str,
    end_sec: str,
    interval_sec: float,
    alpha: float,
    tint_strength: float,
    emphasis_label: str,
    output_path: str,
    video_output_path: str,
    video_fps: float,
):
    """Write the current workspace (frames, masks, points, settings) to disk."""
    if not any(s["frames_bgr"] for s in sets):
        gr.Warning("Nothing to save – load frames into a set first")
        return gr.update(), gr.update()

    name = str(name or "").strip() or default_session_name()
    settings = {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "interval_sec": interval_sec,
        "alpha": alpha,
        "tint_strength": tint_strength,
        "emphasis": emphasis_label,
        "output_path": output_path,
        "video_output_path": video_output_path,
        "video_fps": video_fps,
    }
    try:
        path = save_session(
            name,
            sets,
            active=active,
            idx=idx,
            background_bgr=background,
            video_path=video_path,
            settings=settings,
        )
    except (OSError, ValueError) as exc:
        gr.Warning(f"Could not save the session: {exc}")
        return gr.update(), gr.update()

    gr.Info(f"Saved session to {path}")
    return gr.update(choices=list_sessions(), value=path.name), path.name


def _autosave(autosave: bool, save_args: tuple):
    """Snapshot the session behind a just-rendered output, if autosave is on.

    The session keeps the name shown in the box, so repeated renders update one
    session rather than piling up; a blank box gets a timestamped name that is
    written back, and later renders then update that one.
    """
    if not autosave:
        return gr.update(), gr.update()
    return save_session_cb(*save_args)


# Both render buttons take the same inputs in the same order, so they can share
# one inputs= list and hand save_session_cb the same arguments.
def generate_and_autosave(
    sets: list,
    background,
    alpha: float,
    tint_strength: float,
    emphasis_label: str,
    output_path: str,
    video_output_path: str,
    video_fps: float,
    active: int,
    idx: int,
    video_path,
    name: str,
    start_sec: str,
    end_sec: str,
    interval_sec: float,
    autosave: bool,
):
    """Generate the still composite, then snapshot the session behind it."""
    result = generate_composite(
        sets, background, alpha, tint_strength, emphasis_label, output_path
    )
    if result is None:
        return result, gr.update(), gr.update()
    selector, saved_name = _autosave(
        autosave,
        (
            sets,
            active,
            idx,
            background,
            video_path,
            name,
            start_sec,
            end_sec,
            interval_sec,
            alpha,
            tint_strength,
            emphasis_label,
            output_path,
            video_output_path,
            video_fps,
        ),
    )
    return result, selector, saved_name


def generate_video_and_autosave(
    sets: list,
    background,
    alpha: float,
    tint_strength: float,
    emphasis_label: str,
    output_path: str,
    video_output_path: str,
    video_fps: float,
    active: int,
    idx: int,
    video_path,
    name: str,
    start_sec: str,
    end_sec: str,
    interval_sec: float,
    autosave: bool,
):
    """Render the growing-trail video, then snapshot the session behind it."""
    result = generate_video(
        sets,
        background,
        alpha,
        tint_strength,
        emphasis_label,
        video_output_path,
        video_fps,
        interval_sec,
    )
    if result is None:
        return result, gr.update(), gr.update()
    selector, saved_name = _autosave(
        autosave,
        (
            sets,
            active,
            idx,
            background,
            video_path,
            name,
            start_sec,
            end_sec,
            interval_sec,
            alpha,
            tint_strength,
            emphasis_label,
            output_path,
            video_output_path,
            video_fps,
        ),
    )
    return result, selector, saved_name


def restore_session_cb(name):
    """Repaint the whole workspace from a saved session."""
    if not name:
        gr.Warning("Select a saved session first")
        return _no_restore()
    try:
        data = load_session(name)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        gr.Warning(f"Could not restore session '{name}': {exc}")
        return _no_restore()

    sets = data["sets"] or [_new_set(_next_color(0))]
    active = min(max(data["active"], 0), len(sets) - 1)
    s = sets[active]

    if s["frames"]:
        # Keep st_idx and the slider in sync: change_frame only fires on
        # .release, so a mismatch would annotate a frame that isn't shown.
        idx = min(max(data["idx"], 0), len(s["frames"]) - 1)
        img, preview, _, _ = _current_views(
            s["frames"], s["points_map"], idx, s["masks"]
        )
        slider = gr.update(maximum=max(len(s["frames"]) - 1, 0), value=idx)
    else:
        idx = 0
        img, preview = None, None
        slider = gr.update(maximum=0, value=0)

    bg = data["background_bgr"]
    bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB) if bg is not None else None

    # The dropped video lived in an upload temp dir, so it is usually gone by
    # now; the extracted frames are restored either way.
    video_path = data["video_path"]
    if video_path and Path(video_path).is_file():
        player = _playable_video(video_path)
    else:
        video_path, player = None, None

    cfg = {**DEFAULT_SETTINGS, **data["settings"]}
    # Sets saved before the parameters were kept per set fall back to the
    # session-wide values, which are the ones the widgets held at save time.
    ex = s.get("extract") or {}
    emphasis = cfg["emphasis"]
    if emphasis not in EMPHASIS_MODES:
        emphasis = DEFAULT_SETTINGS["emphasis"]

    gr.Info(f"Restored session '{name}'")
    return (
        sets,  # st_sets
        active,  # st_active
        idx,  # st_idx
        bg,  # st_bg
        video_path,  # st_video
        gr.update(choices=_set_choices(sets), value=f"Set {active + 1}"),
        img,  # input_image
        preview,  # preview_image
        slider,  # frame_slider
        _picker_hex(s["color"], active),  # color_picker
        s["color"] is None,  # no_color_checkbox
        bg_rgb,  # bg_preview
        player,  # video_player
        ex.get("start_sec", cfg["start_sec"]),
        ex.get("end_sec", cfg["end_sec"]),
        ex.get("interval_sec", cfg["interval_sec"]),
        cfg["alpha"],
        cfg["tint_strength"],
        emphasis,
        cfg["output_path"],
        cfg["video_output_path"],
        cfg["video_fps"],
        name,  # session_name
    )
