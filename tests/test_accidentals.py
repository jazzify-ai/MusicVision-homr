"""Displayed accidentals derived from sounding <alter> values."""

import xml.etree.ElementTree as ET
from pathlib import Path

from homr.accidentals import (
    add_display_accidentals,
    add_display_accidentals_to_file,
    key_alterations,
)
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.transformer.vocabulary import EncodedSymbol


def pitch_note(
    step: str,
    octave: int,
    alter: int | None = None,
    *,
    staff: int = 1,
    chord: bool = False,
    tie: str | None = None,
) -> str:
    alter_xml = f"<alter>{alter}</alter>" if alter is not None else ""
    chord_xml = "<chord />" if chord else ""
    tie_xml = f'<tie type="{tie}" />' if tie else ""
    return (
        f"<note>{chord_xml}<pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch>"
        f"<duration>1</duration>{tie_xml}<voice>1</voice><type>quarter</type>"
        f"<staff>{staff}</staff><notations /></note>"
    )


def score(*measures: str, fifths: int = 0) -> ET.Element:
    body = "".join(
        f'<measure number="{i + 1}">'
        + (f"<attributes><key><fifths>{fifths}</fifths></key></attributes>" if i == 0 else "")
        + m
        + "</measure>"
        for i, m in enumerate(measures)
    )
    xml = f'<score-partwise><part id="P1">{body}</part></score-partwise>'
    return ET.fromstring(xml)  # noqa: S314 - XML built by this test.


def shown(root: ET.Element) -> list[str | None]:
    return [n.findtext("accidental") for n in root.iter("note") if n.find("pitch") is not None]


def test_key_alterations_follow_circle_of_fifths() -> None:
    assert key_alterations(-1) == {"B": -1}
    assert key_alterations(-3) == {"B": -1, "E": -1, "A": -1}
    assert key_alterations(2) == {"F": 1, "C": 1}
    assert key_alterations(0) == {}


def test_notes_matching_the_key_signature_show_nothing() -> None:
    root = score(pitch_note("B", 4, -1) + pitch_note("C", 5), fifths=-1)
    assert add_display_accidentals(root) == 0
    assert shown(root) == [None, None]


def test_chromatic_notes_get_accidentals_and_naturals_cancel_the_key() -> None:
    # F major: E-flat, F-sharp, and B-natural are all outside the key.
    root = score(pitch_note("E", 5, -1) + pitch_note("F", 5, 1) + pitch_note("B", 4), fifths=-1)
    add_display_accidentals(root)
    assert shown(root) == ["flat", "sharp", "natural"]


def test_accidental_is_remembered_until_the_barline() -> None:
    root = score(
        pitch_note("E", 6, -1) + pitch_note("E", 6, -1) + pitch_note("E", 5),
        pitch_note("E", 6, -1),
    )
    add_display_accidentals(root)
    # Same octave repeats silently; a different octave is a different slot; new bar resets.
    assert shown(root) == ["flat", None, None, "flat"]


def test_returning_to_the_key_note_in_the_same_measure_needs_a_natural() -> None:
    root = score(pitch_note("F", 5, 1) + pitch_note("F", 5))
    add_display_accidentals(root)
    assert shown(root) == ["sharp", "natural"]


def test_tied_note_across_barline_does_not_repeat_its_accidental() -> None:
    root = score(
        pitch_note("B", 5, 0, tie="start"),
        pitch_note("B", 5, 0, tie="stop") + pitch_note("B", 5, 0),
        fifths=-1,
    )
    add_display_accidentals(root)
    # Airegin m15-16: natural on the first B; the tied continuation inherits it. A later
    # untied B in the new measure needs the natural again.
    assert shown(root) == ["natural", None, "natural"]


def test_chord_tones_and_second_staff_are_tracked_separately() -> None:
    root = score(
        pitch_note("C", 4, 1)
        + pitch_note("E", 4, chord=True)
        + pitch_note("C", 4, 1, staff=2)
        + pitch_note("C", 4, 1)
    )
    add_display_accidentals(root)
    assert shown(root) == ["sharp", None, "sharp", None]


def test_existing_accidentals_are_kept() -> None:
    root = score(pitch_note("B", 4, -1), fifths=-1)
    note = next(root.iter("note"))
    ET.SubElement(note, "accidental").text = "flat"  # courtesy accidental from the writer
    assert add_display_accidentals(root) == 0
    assert shown(root) == ["flat"]


def test_accidental_is_placed_before_time_modification_and_staff() -> None:
    root = score(pitch_note("F", 5, 1))
    note = next(root.iter("note"))
    note.insert(5, ET.Element("time-modification"))
    add_display_accidentals(root)
    tags = [child.tag for child in note]
    assert tags.index("type") < tags.index("accidental") < tags.index("time-modification")
    assert tags.index("accidental") < tags.index("staff")


def test_generated_score_file_gains_accidentals(tmp_path: Path) -> None:
    symbols = [
        EncodedSymbol("clef_G2", position="upper"),
        EncodedSymbol("keySignature_-1"),
        EncodedSymbol("timeSignature/4"),
        EncodedSymbol("note_4", "E5", "b", "_", "_", "upper"),
        EncodedSymbol("note_4", "F5", "#", "_", "_", "upper"),
        EncodedSymbol("note_4", "B4", "_", "_", "_", "upper"),
        EncodedSymbol("note_4", "B4", "b", "_", "_", "upper"),
        EncodedSymbol("barline"),
    ]
    path = tmp_path / "score.musicxml"
    ET.ElementTree(generate_xml(XmlGeneratorArguments(), [symbols], "Accidentals")).write(
        path, encoding="unicode", xml_declaration=True
    )
    assert add_display_accidentals_to_file(path) == 4
    root = ET.parse(path).getroot()  # noqa: S314 - file written by this test.
    assert shown(root) == ["flat", "sharp", "natural", "flat"]
    alters = [n.findtext("pitch/alter") for n in root.iter("note")]
    assert alters == ["-1", "1", None, "-1"]  # sounding pitches are unchanged
