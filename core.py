"""
Core image utilities and SAM 3 integration for motion-trail image creation.

This module contains framework-independent logic:
- Image loading and background estimation
- Alpha-compositing of segmented object layers, as a still or a growing video
- Session persistence (save / restore work in progress)
- SAM 3 model management and per-frame interactive segmentation
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime
from itertools import chain
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


def _set_layers(
    s: dict,
    size: Tuple[int, int],
    tint_strength: float,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """``(frame_index, frame_bgr, mask)`` for every annotated frame of a set.

    The frame index is kept because it is *not* the layer's position: frames
    without a mask are skipped, so a sparsely annotated set has gaps. Emphasis
    is decided by layer position, the growing video by frame index.
    """
    layers = []
    for i, (frame, mask) in enumerate(zip(s["frames_bgr"], s["masks"])):
        if mask is None:
            continue
        frame, mask = resize_to_canvas(frame, mask, size)
        color = s.get("color_bgr")
        if color is not None:
            frame = tint(frame, mask, color, tint_strength)
        layers.append((i, frame, mask))
    return layers


def _opaque_indices(n: int, emphasis: str) -> set:
    """Layer positions painted opaque for *n* layers under *emphasis*."""
    if emphasis == "none":
        return set()
    if emphasis == "first_last":
        return {0, n - 1}
    return {n - 1}  # "last"


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
        layers = [(f, m) for _, f, m in _set_layers(s, size, tint_strength)]
        if not layers:
            continue
        output = overlay_object_on_background(
            output, layers, alpha, opaque_indices=_opaque_indices(len(layers), emphasis)
        )
    return output


def compose_multi_set_progressive(
    sets: List[dict],
    background_bgr: np.ndarray,
    alpha: float = 0.5,
    tint_strength: float = 0.5,
    emphasis: str = "last",
):
    """Yield ``(time_sec, composite)`` as the trails grow, each from its own 0 s.

    **Every set's clock starts at its first annotated frame** — the first one
    SAM 3 segmented — so trails picked out at different points of different
    videos all begin together and can be compared side by side. A layer at
    frame *i* of a set first annotated at *i0* therefore lands at
    ``(i - i0) * interval_sec``, using that set's own ``interval_sec`` (1.0 if
    it has none), so sets sampled at different rates each advance at their true
    speed.

    One step is produced per distinct time any set has a layer at, in ascending
    order, and a set whose annotations run out stops growing rather than
    disappearing — that is what lets sets of different lengths share one
    timeline. The time comes out with the image because :func:`pace_steps`
    needs it: the steps are not evenly spaced when only some frames were
    annotated.

    *emphasis* is applied to the layers visible so far, so with the default
    ``"last"`` the newest frame is the opaque one and the trail behind it
    fades. The final step is therefore pixel-identical to what
    :func:`compose_multi_set` returns for the same arguments.

    Layers are tinted once up front and reused across steps, which costs one
    extra frame-sized array per annotated frame of a coloured set.
    """
    size = background_bgr.shape[:2]
    per_set = []
    for s in sets:
        layers = _set_layers(s, size, tint_strength)
        if not layers:
            per_set.append([])
            continue
        first = layers[0][0]
        step = max(float(s.get("interval_sec") or 1.0), 0.0)
        # rounded so two sets landing on the same moment share one step rather
        # than producing a pair of steps a float wobble apart
        per_set.append([(round((i - first) * step, 6), f, m) for i, f, m in layers])

    for t in sorted({at for layers in per_set for at, _, _ in layers}):
        output = background_bgr.copy()
        for layers in per_set:
            visible = [(f, m) for at, f, m in layers if at <= t]
            if not visible:
                continue
            output = overlay_object_on_background(
                output,
                visible,
                alpha,
                opaque_indices=_opaque_indices(len(visible), emphasis),
            )
        yield t, output


def pace_steps(steps, fps: float):
    """Repeat each composite so the video runs on the timeline *steps* carries.

    *steps* is the ``(time_sec, image)`` stream from
    :func:`compose_multi_set_progressive`. A step stays on screen until the
    next one's moment arrives, so a gap in the annotations reads as a pause and
    the trail grows at the speed the object actually moved, whatever *fps* the
    file is encoded at. The last step is held for as long as the one before it
    lasted, so the finished trail does not flash past.

    Output positions are derived from absolute times rather than accumulated
    per gap, so rounding cannot drift over a long clip. Every step gets at
    least one frame, so no annotation is dropped even at a tiny interval.
    """
    rate = max(float(fps), 0.1)
    first = None
    previous = None
    previous_at = None
    last_gap = None
    emitted = 0
    for at, image in steps:
        if first is None:
            first = at
        else:
            last_gap = at - previous_at
            target = max(int(round((at - first) * rate)), emitted + 1)
            for _ in range(target - emitted):
                yield previous
            emitted = target
        previous, previous_at = image, at
    if previous is not None:
        tail = last_gap if last_gap else 1.0 / rate
        for _ in range(max(int(round(tail * rate)), 1)):
            yield previous


# ---------------------------------------------------------------------------
# Video output
# ---------------------------------------------------------------------------

# Containers an H.264 stream can go into; the output path's suffix picks one.
VIDEO_OUT_EXTS = {".mp4", ".mov", ".mkv", ".avi"}


def write_video(frames_bgr, path: Path, fps: float = 10.0) -> str:
    """Encode an iterable of BGR frames to *path*; return the codec written.

    ffmpeg (H.264) is used when it is on PATH, so the result plays in a browser
    as well as everywhere else; without it OpenCV's mpeg4 writer takes over,
    which most desktop players handle but no browser does. The encoder is
    chosen before any frame is read, because *frames_bgr* may be a generator
    that cannot be replayed — which also keeps memory at one frame at a time.
    """
    frames = iter(frames_bgr)
    first = next(frames, None)
    if first is None:
        raise ValueError("no frames to write")
    frames = chain([first], frames)
    h, w = first.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(float(fps), 0.1)

    if shutil.which("ffmpeg"):
        _write_video_ffmpeg(frames, path, fps, w, h)
        return "h264"
    _write_video_opencv(frames, path, fps, w, h)
    return "mpeg4"


def _write_video_ffmpeg(frames, path: Path, fps: float, w: int, h: int) -> None:
    """Pipe raw BGR frames into ffmpeg and encode them as H.264."""
    # fmt: off
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}",
        "-i", "-",
        # yuv420p needs even dimensions, which an odd-sized frame would break
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
    ]
    # fmt: on
    if path.suffix.lower() in {".mp4", ".mov"}:
        cmd += ["-movflags", "+faststart"]  # only the mov muxer knows this one
    proc = subprocess.Popen(
        cmd + [str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass  # ffmpeg died early; its stderr below says why
    err = proc.stderr.read().decode("utf-8", "replace").strip()
    proc.stderr.close()
    if proc.wait() != 0:
        raise RuntimeError(err or "ffmpeg could not encode the video")


def _write_video_opencv(frames, path: Path, fps: float, w: int, h: int) -> None:
    """Fallback encoder for machines without ffmpeg (mpeg4, not browser-safe).

    Frames are scaled to even dimensions, matching what the ffmpeg branch's
    scale filter does: this writer accepts an odd size but then silently drops
    the last row and column, losing an edge of every frame.
    """
    w, h = max(w - w % 2, 2), max(h - h % 2, 2)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open {path} for writing")
    try:
        for frame in frames:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------
#
# A session is a directory holding everything needed to resume annotating:
#
#     sessions/<name>/
#         session.json          metadata (points, colours, widget settings)
#         background.png        chosen background frame, if any
#         set00/frame_0000.png  the set's frames, losslessly (BGR)
#         set00/mask_0000.png   masks, only for frames that have one
#
# Frames are stored as pixels rather than as a reference to their source,
# because neither input path survives: a dropped image folder has no path at
# all and a dropped video lives in an upload temp dir that is gone on restart.
# PNG (not JPEG) keeps a restored session's composite identical to the original.

SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"
SESSION_VERSION = 1


def _sanitize_name(name) -> str:
    """Reduce user input to a safe single path component (may be empty)."""
    return re.sub(r"[^\w\-. ]", "_", str(name or "").strip()).strip(". ")


def _session_dir(name) -> Path:
    """Resolve ``sessions/<name>``, refusing anything outside SESSIONS_DIR."""
    safe = _sanitize_name(name)
    root = SESSIONS_DIR.resolve()
    path = (root / safe).resolve() if safe else root
    if not safe or path.parent != root:
        raise ValueError(f"Invalid session name: {name!r}")
    return path


def list_sessions() -> List[str]:
    """Names of saved sessions, most recently written first."""
    if not SESSIONS_DIR.is_dir():
        return []
    dirs = [p for p in SESSIONS_DIR.iterdir() if (p / "session.json").is_file()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in dirs]


def default_session_name() -> str:
    """Timestamped fallback name for a session saved without one."""
    return datetime.now().strftime("session_%Y%m%d_%H%M%S")


def _array_key(arr: np.ndarray) -> str:
    """Content hash of an image, used to skip re-encoding unchanged files."""
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _at(seq, i):
    """``seq[i]`` or None – reads a hash list that may be short or missing."""
    return seq[i] if seq and i < len(seq) else None


def _read_manifest(path: Path) -> dict:
    """The session's previous ``session.json``, or {} if absent / unreadable."""
    try:
        meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


# Names a session owns, and may therefore delete when they go stale. Anything
# else in the directory is left alone — a user is free to point the composite's
# output path at their session folder, and autosave must not eat the result.
_OWNED_TOP = re.compile(r"set\d+$")
_OWNED_SET = re.compile(r"(frame|mask)_\d+\.png$")


def _prune(directory: Path, keep: set, owned: re.Pattern) -> None:
    """Delete the session's own *owned* entries in *directory* unless kept."""
    for entry in directory.iterdir():
        if entry.name in keep or not owned.match(entry.name):
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def save_session(
    name: str,
    sets: List[dict],
    active: int = 0,
    idx: int = 0,
    background_bgr: np.ndarray | None = None,
    video_path: str | None = None,
    settings: dict | None = None,
) -> Path:
    """Write the whole workspace to ``sessions/<name>/`` and return that dir.

    *sets* are the app's set records (``frames_bgr``, ``masks``, ``points_map``,
    ``color``, ``dir``); only ``frames_bgr`` is written, since the RGB copy is
    derived on load. A blank *name* becomes a timestamp; an existing session of
    the same name is updated in place.

    Re-saving is incremental: images whose content hash matches what the
    session already holds are left untouched, and files no longer needed are
    pruned. Re-encoding a 1080p PNG costs ~40 ms against ~6 ms to hash it, so
    autosaving after every composite stays cheap when the frames haven't
    changed and only the settings have.
    """
    out = _session_dir(_sanitize_name(name) or default_session_name())
    prev_meta = _read_manifest(out) if out.is_dir() else {}
    previous = prev_meta.get("sets")
    previous = previous if isinstance(previous, list) else []
    out.mkdir(parents=True, exist_ok=True)

    meta_sets = []
    for i, s in enumerate(sets):
        sub = out / f"set{i:02d}"
        sub.mkdir(exist_ok=True)
        old = previous[i] if i < len(previous) else {}
        old_frames = old.get("frame_hashes")
        old_masks = old.get("mask_hashes")

        frames = list(s.get("frames_bgr") or [])
        masks = list(s.get("masks") or [])
        frame_hashes: List[str] = []
        mask_hashes: List[str | None] = []
        keep = set()
        for j, frame in enumerate(frames):
            key = _array_key(frame)
            frame_hashes.append(key)
            path = sub / f"frame_{j:04d}.png"
            keep.add(path.name)
            if key != _at(old_frames, j) or not path.is_file():
                cv2.imwrite(str(path), frame)

            mask = masks[j] if j < len(masks) else None
            if mask is None:
                mask_hashes.append(None)  # absence of a file means "no mask"
                continue
            binary = (mask.astype(np.uint8) > 0).astype(np.uint8)
            key = _array_key(binary)
            mask_hashes.append(key)
            path = sub / f"mask_{j:04d}.png"
            keep.add(path.name)
            if key != _at(old_masks, j) or not path.is_file():
                cv2.imwrite(str(path), binary * 255)  # 0/1 -> 0/255, viewable
        _prune(sub, keep, _OWNED_SET)

        # What this set was extracted with, so a restore can say where its
        # frames came from; None for a folder of images, which has no such
        # parameters. Kept per set because the widgets are shared and only ever
        # hold the last extraction's values.
        extract = s.get("extract") or None
        if extract:
            extract = {
                "start_sec": str(extract.get("start_sec", "")),
                "end_sec": str(extract.get("end_sec", "")),
                "interval_sec": float(extract.get("interval_sec", 1.0)),
            }

        color = s.get("color")
        meta_sets.append(
            {
                "dir": s.get("dir", ""),
                "extract": extract,
                "color": [int(c) for c in color] if color is not None else None,
                "n_frames": len(frames),
                # JSON keys are strings; load_session converts them back to int
                "points_map": {
                    str(k): [[int(x), int(y), int(lab)] for x, y, lab in v]
                    for k, v in (s.get("points_map") or {}).items()
                },
                "frame_hashes": frame_hashes,
                "mask_hashes": mask_hashes,
            }
        )

    bg_path = out / "background.png"
    bg_hash = None
    if background_bgr is not None:
        bg_hash = _array_key(background_bgr)
        if bg_hash != prev_meta.get("background_hash") or not bg_path.is_file():
            cv2.imwrite(str(bg_path), background_bgr)
    elif bg_path.is_file():
        bg_path.unlink()

    meta = {
        "version": SESSION_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "active": int(active),
        "idx": int(idx),
        "video_path": str(video_path) if video_path else None,
        "has_background": background_bgr is not None,
        "background_hash": bg_hash,
        "settings": dict(settings or {}),
        "sets": meta_sets,
    }
    # Drop set dirs left behind by a previous save that had more sets.
    _prune(out, {f"set{i:02d}" for i in range(len(sets))}, _OWNED_TOP)
    (out / "session.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def load_session(name: str) -> dict:
    """Read a session back into app-shaped state.

    Returns ``{"sets", "active", "idx", "background_bgr", "video_path",
    "settings"}``, where each set matches the app's in-memory record.
    """
    path = _session_dir(name)
    meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
    version = int(meta.get("version", 0))
    if version > SESSION_VERSION:
        raise ValueError(
            f"session format v{version} is newer than this app (v{SESSION_VERSION})"
        )

    sets: List[dict] = []
    for i, m in enumerate(meta.get("sets", [])):
        sub = path / f"set{i:02d}"
        frames_bgr: List[np.ndarray] = []
        masks: List[np.ndarray | None] = []
        for j in range(int(m.get("n_frames", 0))):
            frame = cv2.imread(str(sub / f"frame_{j:04d}.png"))
            if frame is None:
                break  # truncate rather than skip, to keep frame indices aligned
            frames_bgr.append(frame)
            mask_path = sub / f"mask_{j:04d}.png"
            mask = (
                cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_path.is_file()
                else None
            )
            # None (never annotated) must stay distinct from an all-zero mask
            masks.append(None if mask is None else (mask > 127).astype(np.uint8))
        color = m.get("color")
        sets.append(
            {
                "dir": m.get("dir", ""),
                "frames": [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr],
                "frames_bgr": frames_bgr,
                # JSON stringified the int frame indices – convert them back
                "points_map": {
                    int(k): [tuple(p) for p in v]
                    for k, v in (m.get("points_map") or {}).items()
                },
                "masks": masks,
                "color": tuple(color) if color is not None else None,
                "extract": m.get("extract") or None,
            }
        )

    background = None
    if meta.get("has_background"):
        background = cv2.imread(str(path / "background.png"))

    return {
        "sets": sets,
        "active": int(meta.get("active", 0)),
        "idx": int(meta.get("idx", 0)),
        "background_bgr": background,
        "video_path": meta.get("video_path") or None,
        "settings": meta.get("settings") or {},
    }


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
