"""Retain uncertain rhythm and reject a decorative title staff (Jazzify).

Ported from MusicVision-ml-private ``musicvision_ml/homr_rhythm_refinement.py``. The only change
is the title filter hook, which works on this fork's ``DetectionArtifacts`` instead of upstream's
tuple return value.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import replace as replace_fields
from fractions import Fraction
from pathlib import Path


def timing_diagnostics(measures: list[list[dict]]) -> dict:
    """Describe mismatches and review proposals; never alter an event's rhythm."""
    totals = [sum((Fraction(e["duration"]) for e in m), Fraction()) for m in measures]
    ordinary = Counter(
        total
        for m, total in zip(measures, totals, strict=True)
        if m and total > 0 and all(e["ratio"] == "1:1" for e in m)
    )
    ranked = ordinary.most_common()
    reference = None
    if ranked and ranked[0][1] >= 3 and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
        reference = ranked[0][0]
    flags = []
    for index, (events, total) in enumerate(zip(measures, totals, strict=True)):
        if not events:
            continue
        reasons = []
        if reference is not None and total != reference:
            reasons.append("measure_duration_mismatch")
        if any(e.get("unpitched_note") for e in events):
            reasons.append("note_token_without_pitch")
        if not reasons:
            continue
        flag = {"token_measure_index": index + 1, "duration": str(total), "reasons": reasons}
        # A proposed rest value is evidence for review, never an automatic repair.
        rests = [i for i, e in enumerate(events) if e["kind"] == "rest"]
        if reference is not None and len(rests) == 1:
            rest_index = rests[0]
            needed = reference - total + Fraction(events[rest_index]["duration"])
            if needed > 0:
                flag["review_only_rest_proposal"] = {
                    "event_index": rest_index,
                    "quarter_duration": str(needed),
                    "applied": False,
                }
        flags.append(flag)
    return {
        "reference_quarter_duration": str(reference) if reference is not None else None,
        "reference_support": ranked[0][1] if ranked else 0,
        "flags": flags,
    }


def reject_leading_decoration(features: list[dict]) -> bool:
    """Require positive text evidence and no notation anchors; never drop all staffs."""
    if len(features) < 2:
        return False
    first, following = features[:2]
    return (
        first["staff_count"] == 1
        and not first["grandstaff"]
        and first["notes"] == 0
        and first["rests"] == 0
        and first["clef_key_anchors"] == 0
        and first["barlines"] == 0
        and first["large_text_components"] >= 3
        and following["notes"] > 0
        and following["clef_key_anchors"] > 0
    )


def _near_staff(box, staff) -> bool:
    x, y = box.center
    padding = 2 * float(staff.average_unit_size)
    return (
        staff.min_x - padding <= x <= staff.max_x + padding
        and staff.min_y - padding <= y <= staff.max_y + padding
    )


def _leading_clef_key_anchor(box, staff) -> bool:
    return _near_staff(box, staff) and box.center[0] <= staff.min_x + 8 * float(
        staff.average_unit_size
    )


def _large_text_components(image, staff) -> int:
    import cv2
    import numpy as np

    unit = float(staff.average_unit_size)
    y0, y1 = max(0, int(staff.min_y - 4 * unit)), min(image.shape[0], int(staff.max_y + 4 * unit))
    x0, x1 = max(0, int(staff.min_x)), min(image.shape[1], int(staff.max_x))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    ink = (gray < 128).astype(np.uint8)
    # Remove long horizontal staff lines before measuring connected text-like ink.
    lines = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, np.ones((1, max(20, crop.shape[1] // 4)), np.uint8)
    )
    remaining = cv2.subtract(ink, lines)
    # Removing a line also cuts through letters. Bridge only that narrow vertical gap.
    remaining = cv2.morphologyEx(
        remaining, cv2.MORPH_CLOSE, np.ones((max(3, int(unit / 4)), 1), np.uint8)
    )
    _, _, stats, _ = cv2.connectedComponentsWithStats(remaining.astype(np.uint8), connectivity=8)
    return sum(int(w >= unit and h >= 2.5 * unit) for _, _, w, h, _ in stats[1:])


def _staff_features(rows, image, clefs, barlines) -> list[dict]:
    result = []
    for index, row in enumerate(rows):
        result.append(
            {
                "system_index": index,
                "staff_count": len(row.staffs),
                "grandstaff": any(s.is_grandstaff for s in row.staffs),
                "notes": sum(len(s.get_notes()) for s in row.staffs),
                "rests": sum(type(v).__name__ == "Rest" for s in row.staffs for v in s.symbols),
                "all_clef_key_anchors": sum(
                    any(_near_staff(box, s) for s in row.staffs) for box in clefs
                ),
                "clef_key_anchors": sum(
                    any(_leading_clef_key_anchor(box, s) for s in row.staffs) for box in clefs
                ),
                "barlines": sum(any(_near_staff(box, s) for s in row.staffs) for box in barlines),
                "large_text_components": sum(_large_text_components(image, s) for s in row.staffs),
                "bounds": [
                    [float(s.min_x), float(s.min_y), float(s.max_x), float(s.max_y)]
                    for s in row.staffs
                ],
            }
        )
    return result


def _events_from_chords(chords) -> list[dict]:
    events = []
    for chord in chords:
        symbols = [s for s in chord if s.rhythm.startswith(("note", "rest"))]
        if not symbols:
            continue
        # Do not imply monophonic timing validity for ambiguous chords/voices.
        if len(symbols) != 1 or symbols[0].position not in ("upper", "."):
            return []
        symbol = symbols[0]
        duration = symbol.get_duration()
        if duration.fraction <= 0:
            continue
        events.append(
            {
                "duration": str(4 * duration.fraction),
                "ratio": f"{duration.actual_notes}:{duration.normal_notes}",
                "kind": (
                    "rest" if symbol.rhythm.startswith("rest") or symbol.pitch == "_" else "note"
                ),
                "unpitched_note": symbol.rhythm.startswith("note") and symbol.pitch == "_",
            }
        )
    return events


@contextmanager
def upstream_experiment(hm, *, preserve_tuplets: bool, filter_header: bool):
    """Apply process-local hooks only; the pinned upstream source stays untouched."""
    if not preserve_tuplets and not filter_header:
        yield {}
        return
    import hashlib

    from homr import staff_parsing
    from homr.transformer import vocabulary

    evidence = {
        "preserve_tuplets": preserve_tuplets,
        "filter_header": filter_header,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rhythm": [],
        "staff_filter": {},
    }
    originals = []

    def replace(owner, name, function):
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, function)

    if preserve_tuplets:

        def preserve(chords):
            measures = vocabulary._group_into_measures(chords)
            evidence["rhythm"].append(
                timing_diagnostics([_events_from_chords(m) for m in measures])
            )
            return chords

        replace(vocabulary, "_fix_over_eager_tuplets", preserve)
        # Isolate preservation from another heuristic which invents unmarked tuplets.
        replace(staff_parsing, "restore_unmarked_triplets", lambda symbols: symbols)
    if filter_header:
        anchors: dict[str, list] = {"clefs": [], "barlines": []}
        predict_symbols = hm.predict_symbols
        detect_barlines = hm.detect_bar_lines
        detect_staffs = hm.detect_staffs_in_image

        def capture_symbols(*args, **kwargs):
            symbols = predict_symbols(*args, **kwargs)
            anchors["clefs"] = symbols.clefs_keys
            return symbols

        def capture_barlines(*args, **kwargs):
            lines = detect_barlines(*args, **kwargs)
            anchors["barlines"] = lines
            return lines

        def detect(*args, **kwargs):
            detection = detect_staffs(*args, **kwargs)
            rows = detection.multi_staffs
            features = _staff_features(
                rows, detection.processed_image, anchors["clefs"], anchors["barlines"]
            )
            reject = reject_leading_decoration(features)
            evidence["staff_filter"] = {
                "systems_before": len(rows),
                "systems_after": len(rows) - int(reject),
                "rejected_system_indices": [0] if reject else [],
                "features": features,
            }
            if reject:
                return replace_fields(detection, multi_staffs=rows[1:])
            return detection

        replace(hm, "predict_symbols", capture_symbols)
        replace(hm, "detect_bar_lines", capture_barlines)
        replace(hm, "detect_staffs_in_image", detect)
    try:
        yield evidence
    finally:
        for owner, name, function in reversed(originals):
            setattr(owner, name, function)
