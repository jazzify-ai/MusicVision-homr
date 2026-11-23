"""Image-level clef/key carry-forward for lead-sheet lines that print neither.

Lead sheets often print the clef and key signature on the first line only. HOMR predicts every
note's accidental from the staff image, so on a later line a plain E notehead under a carried
three-flat key comes out as E natural: forcing the carried key *token* (decoding change B) fixes
the written key but not the notes ('Round Midnight bars 5-18, 2026-11-22).

This module copies the clef + key-signature strip of the last staff that printed them and puts it
in front of a staff that prints neither, so the model reads the line the way it was trained: with
the key signature in view. It works on the transformer's input canvas (grayscale, content
left-aligned, fixed size) and returns a mapping that moves the decoded symbols' coordinates back
onto the original canvas, so chord alignment and MusicXML positions are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

INK = 128  # gray value below which a pixel counts as ink
# A row belongs to a staff line when ink covers more than this share of the content width.
LINE_COVERAGE = 0.5
# Five line centres count as one staff when their spacings vary by less than this ratio.
SPACING_TOLERANCE = 0.25


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    import cv2

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _ink_columns(gray: np.ndarray) -> np.ndarray:
    return np.where((gray < INK).any(axis=0))[0]


def staff_lines(image: np.ndarray) -> np.ndarray | None:
    """Y centres of the five staff lines (top to bottom), or None if no clean staff is found."""
    gray = _gray(image)
    columns = _ink_columns(gray)
    if len(columns) == 0:
        return None
    ink = gray[:, columns[0] : columns[-1] + 1] < INK
    # Thin or slightly tilted lines spread over 2-3 rows ('Round Midnight staff 5): measure the
    # coverage of a 3-row band so no single row has to carry the whole line.
    band = ink.copy()
    band[1:] |= ink[:-1]
    band[:-1] |= ink[1:]
    rows = np.where(band.mean(axis=1) > LINE_COVERAGE)[0]
    if len(rows) < 5:
        return None
    groups = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
    centres = [float(group.mean()) for group in groups]
    best: tuple[float, np.ndarray] | None = None
    for start in range(len(centres) - 4):
        window = np.array(centres[start : start + 5])
        spacing = np.diff(window)
        if spacing.min() <= 0:
            continue
        spread = float(spacing.std() / spacing.mean())
        if best is None or spread < best[0]:
            best = (spread, window)
    if best is None or best[0] > SPACING_TOLERANCE:
        return None
    return best[1]


@dataclass(frozen=True)
class ClefKeyStrip:
    image: np.ndarray  # grayscale crop from the line's left edge to the end of the key signature
    top: float  # y of the top staff line within ``image``
    unit: float  # staff line spacing within ``image``


def extract_strip(canvas: np.ndarray, x_end: float) -> ClefKeyStrip | None:
    """Cut the clef + key signature (everything left of ``x_end``) from a staff canvas.

    Returns None when no staff is found or the strip would be implausibly narrow or wide (a wrong
    ``x_end`` must not paste half a bar of music onto other lines).
    """
    gray = _gray(canvas)
    lines = staff_lines(gray)
    columns = _ink_columns(gray)
    if lines is None or len(columns) == 0:
        return None
    unit = float(np.diff(lines).mean())
    x0, x1 = int(columns[0]), int(columns[-1]) + 1
    x_end = int(round(x_end))
    if x_end - x0 < 2 * unit or x_end - x0 > 0.35 * (x1 - x0):
        return None
    return ClefKeyStrip(gray[:, x0:x_end].copy(), float(lines[0]), unit)


def paste_strip(
    canvas: np.ndarray, strip: ClefKeyStrip
) -> tuple[np.ndarray, Callable[[float, float], tuple[float, float]]] | None:
    """Put ``strip`` in front of the staff on ``canvas``.

    The strip is scaled to the target's staff line spacing and shifted so the staff lines meet.
    If the result is wider than the canvas, everything is scaled down to fit. Returns the new
    canvas and a function mapping a point on it back to the original canvas (points inside the
    pasted strip map to the original line's left edge), or None if the target has no clean staff.
    """
    import cv2

    gray = _gray(canvas)
    height, width = gray.shape
    lines = staff_lines(gray)
    columns = _ink_columns(gray)
    if lines is None or len(columns) == 0:
        return None
    unit = float(np.diff(lines).mean())
    x0, x1 = int(columns[0]), int(columns[-1]) + 1
    scale = unit / strip.unit
    strip_height, strip_width = strip.image.shape
    scaled = cv2.resize(
        strip.image,
        (max(1, round(strip_width * scale)), max(1, round(strip_height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    paste_width = scaled.shape[1]
    combined = np.full((height, paste_width + (x1 - x0)), 255, dtype=np.uint8)
    # Align the strip's top staff line with the target's.
    dy = int(round(float(lines[0]) - strip.top * scale))
    top, bottom = max(0, dy), min(height, dy + scaled.shape[0])
    if bottom > top:
        combined[top:bottom, :paste_width] = scaled[top - dy : bottom - dy]
    combined[:, paste_width:] = gray[:, x0:x1]
    fit = min(1.0, width / combined.shape[1])
    out = np.full((height, width), 255, dtype=np.uint8)
    if fit < 1.0:
        fitted = cv2.resize(
            combined,
            (width, max(1, round(height * fit))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        fitted = combined
    y_offset = (height - fitted.shape[0]) // 2
    out[y_offset : y_offset + fitted.shape[0], : fitted.shape[1]] = fitted

    def to_original(x: float, y: float) -> tuple[float, float]:
        return max(float(x0), x / fit - paste_width + x0), (y - y_offset) / fit

    return out, to_original
