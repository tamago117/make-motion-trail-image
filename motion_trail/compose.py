"""Alpha-composite segmented objects into a motion trail, as a still or a growing video."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


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
