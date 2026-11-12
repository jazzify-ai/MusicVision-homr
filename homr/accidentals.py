"""Add displayed accidentals to MusicXML whose pitches carry only sounding ``<alter>`` values.

The transformer predicts the sounding alteration of every note (the "lift"), and the exporter
writes it as ``<pitch><alter>``. Renderers such as Verovio only draw an accidental when the note
also has an ``<accidental>`` element, so without this pass the sheet shows no sharps, flats or
naturals even though the pitches are correct.

Standard engraving rules are applied per part, staff and measure:

- A note needs an accidental when its alteration differs from what the reader would assume:
  the key signature, or an earlier accidental on the same step and octave in the same measure
  and staff.
- A note that continues a tie needs no new accidental; it inherits the tied note's alteration.
- Existing ``<accidental>`` elements are kept and update the measure state.

The module only uses the standard library so it can also post-process MusicXML written by other
HOMR checkouts (``python -m homr.accidentals score.musicxml``).
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Order in which flats/sharps are added by a key signature.
_SHARP_ORDER = "FCGDAEB"
_FLAT_ORDER = "BEADGCF"

ALTER_TO_ACCIDENTAL = {
    -2: "flat-flat",
    -1: "flat",
    0: "natural",
    1: "sharp",
    2: "double-sharp",
}

# Children that must follow <accidental> inside <note> (MusicXML 4.0 sequence).
_AFTER_ACCIDENTAL = {
    "time-modification",
    "stem",
    "notehead",
    "notehead-text",
    "staff",
    "beam",
    "notations",
    "lyric",
    "play",
    "listen",
}

ACCIDENTAL_TO_ALTER = {name: alter for alter, name in ALTER_TO_ACCIDENTAL.items()}
ACCIDENTAL_TO_ALTER["sharp-sharp"] = 2
ACCIDENTAL_TO_ALTER["double-flat"] = -2


def key_alterations(fifths: int) -> dict[str, int]:
    """Map each step to the alteration implied by a key signature with ``fifths``."""
    if fifths >= 0:
        return dict.fromkeys(_SHARP_ORDER[: min(fifths, 7)], 1)
    return dict.fromkeys(_FLAT_ORDER[: min(-fifths, 7)], -1)


def _note_alter(note: ET.Element) -> int | None:
    pitch = note.find("pitch")
    if pitch is None:
        return None
    text = pitch.findtext("alter")
    if text is None or text.strip() == "":
        return 0
    value = float(text)
    if not value.is_integer():
        return None  # Microtones are left to the writer.
    return int(value)


def _staff(note: ET.Element) -> str:
    return note.findtext("staff") or "1"


def _is_tie_continuation(note: ET.Element) -> bool:
    return any(tie.get("type") == "stop" for tie in note.findall("tie"))


def _insert_accidental(note: ET.Element, value: str) -> None:
    accidental = ET.Element("accidental")
    accidental.text = value
    children = list(note)
    for index, child in enumerate(children):
        if child.tag in _AFTER_ACCIDENTAL:
            note.insert(index, accidental)
            return
    note.append(accidental)


def _update_keys(attributes: ET.Element, keys: dict[str, dict[str, int]], staves: set[str]) -> None:
    for key in attributes.findall("key"):
        fifths_text = key.findtext("fifths")
        if fifths_text is None:
            continue
        alterations = key_alterations(int(fifths_text))
        number = key.get("number")
        targets = {number} if number else (staves | {"*"})
        for staff in targets:
            keys[staff] = alterations


def add_display_accidentals(root: ET.Element) -> int:
    """Insert ``<accidental>`` elements in place. Returns the number of accidentals added."""
    added = 0
    for part in root.iter("part"):
        keys: dict[str, dict[str, int]] = {"*": {}}
        staves: set[str] = {"1"}
        # Alteration of the last note per (staff, step, octave), used for ties across barlines.
        last_alter: dict[tuple[str, str, str], int] = {}
        for measure in part.findall("measure"):
            in_measure: dict[tuple[str, str, str], int] = {}
            for element in measure:
                if element.tag == "attributes":
                    _update_keys(element, keys, staves)
                    continue
                if element.tag != "note" or element.find("pitch") is None:
                    continue
                alter = _note_alter(element)
                if alter is None:
                    continue
                staff = _staff(element)
                staves.add(staff)
                step = element.findtext("pitch/step") or ""
                octave = element.findtext("pitch/octave") or ""
                slot = (staff, step, octave)
                key = keys.get(staff, keys["*"])
                expected = in_measure.get(slot, key.get(step, 0))

                existing = element.findtext("accidental")
                if existing is not None:
                    in_measure[slot] = ACCIDENTAL_TO_ALTER.get(existing.strip(), alter)
                elif (
                    _is_tie_continuation(element)
                    and slot not in in_measure
                    and last_alter.get(slot) == alter
                ):
                    # The tied note already showed its accidental. It carries over only for this
                    # continuation; a later untied note in the measure still needs its own.
                    pass
                elif alter != expected and alter in ALTER_TO_ACCIDENTAL:
                    _insert_accidental(element, ALTER_TO_ACCIDENTAL[alter])
                    in_measure[slot] = alter
                    added += 1
                last_alter[slot] = alter
    return added


def add_display_accidentals_to_file(path: str | Path) -> int:
    """Rewrite a MusicXML file with displayed accidentals. Returns the number added."""
    tree = ET.parse(path)  # noqa: S314 - MusicXML written by this process.
    added = add_display_accidentals(tree.getroot())
    ET.indent(tree, space="  ")
    tree.write(path, encoding="UTF-8", xml_declaration=True)
    return added


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        count = add_display_accidentals_to_file(argument)
        sys.stdout.write(f"{argument}: {count} accidentals added\n")
