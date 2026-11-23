import numpy as np

from homr.jazzify.clef_key_strip import extract_strip, paste_strip, staff_lines

H, W = 256, 1280


def staff_canvas(line_top: int = 100, unit: int = 12, width: int = 900, marks=()) -> np.ndarray:
    canvas = np.full((H, W), 255, dtype=np.uint8)
    for i in range(5):
        y = line_top + i * unit
        canvas[y : y + 2, :width] = 0
    for x, y in marks:  # small filled squares standing in for glyphs
        canvas[y : y + 8, x : x + 8] = 0
    return canvas


def test_staff_lines_finds_five_evenly_spaced_lines():
    lines = staff_lines(staff_canvas(line_top=90, unit=14))
    assert lines is not None
    assert np.allclose(np.diff(lines), 14, atol=0.5)
    assert abs(lines[0] - 90.5) < 1


def test_staff_lines_finds_thin_tilted_lines():
    canvas = np.full((H, W), 255, dtype=np.uint8)
    for i in range(5):
        for x in range(1000):  # one-pixel lines drifting down by two rows across the width
            canvas[100 + i * 13 + (x * 3) // 1000, x] = 0
    lines = staff_lines(canvas)
    assert lines is not None
    assert np.allclose(np.diff(lines), 13, atol=1)


def test_staff_lines_none_without_staff():
    assert staff_lines(np.full((H, W), 255, dtype=np.uint8)) is None


def test_extract_rejects_implausible_widths():
    canvas = staff_canvas()
    assert extract_strip(canvas, 10) is None  # narrower than two staff spaces
    assert extract_strip(canvas, 600) is None  # most of the line: a wrong cut
    strip = extract_strip(canvas, 120)
    assert strip is not None and strip.image.shape[1] == 120


def test_paste_aligns_lines_and_maps_points_back():
    source = staff_canvas(line_top=100, unit=12, marks=[(20, 110), (60, 104)])
    strip = extract_strip(source, 100)
    target = staff_canvas(line_top=80, unit=16, width=1000, marks=[(400, 90)])
    pasted, to_original = paste_strip(target, strip)
    assert pasted.shape == (H, W)
    lines = staff_lines(pasted)
    assert lines is not None  # the joined line is still one clean staff
    # The target's own mark moved right by the pasted width (scaled) and maps back to x = 400.
    ys, xs = np.where(pasted[:, :] < 128)
    x, y = to_original(400 + 100 * 16 / 12, 90)
    assert abs(x - 400) < 1.5 and abs(y - 90) < 1.5
    # Points inside the pasted strip map to the original left edge.
    assert to_original(5, 90)[0] == 0


def test_paste_scales_down_when_too_wide():
    source = staff_canvas(width=1280)
    strip = extract_strip(source, 300)
    target = staff_canvas(width=1280)
    pasted, to_original = paste_strip(target, strip)
    assert pasted.shape == (H, W)
    # Right edge of the pasted canvas maps to the right edge of the original content.
    x, _ = to_original(W - 1, H / 2)
    assert abs(x - (W - 1)) < 3
