"""Regression cases for notation lost after semantic recognition."""

import xml.etree.ElementTree as ET
from fractions import Fraction

import pytest

import homr.main as hm
from homr.jazzify.refinement import upstream_experiment
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.transformer.vocabulary import EncodedSymbol, remove_duplicated_symbols


def note(
    rhythm: str = "note_4",
    pitch: str = "C4",
    articulation: str = "_",
    position: str = "upper",
    slur: str = "_",
) -> EncodedSymbol:
    return EncodedSymbol(rhythm, pitch, "_", articulation, slur, position)


def score(symbols: list[EncodedSymbol]) -> ET.Element:
    prefix = [EncodedSymbol("clef_G2", position="upper"), EncodedSymbol("timeSignature/4")]
    xml = generate_xml(XmlGeneratorArguments(), [prefix + symbols], "Rhythm regression")
    return xml


def marks(element: ET.Element, path: str) -> list[str | None]:
    return [child.get("type") for child in element.findall(path)]


def test_tie_has_both_playback_and_notation_endpoints() -> None:
    # Model 509 has no tie tokens: a tie is predicted as a slur between two equal pitches.
    xml = score([note("note_2", slur="slurStart"), note("note_2", slur="slurStop")])
    notes = xml.findall(".//note")
    assert marks(notes[0], "tie") == ["start"]
    assert marks(notes[0], "notations/tied") == ["start"]
    assert marks(notes[1], "tie") == ["stop"]
    assert marks(notes[1], "notations/tied") == ["stop"]


def test_tie_continuation_stops_before_starting_next_tie() -> None:
    xml = score(
        [
            note(slur="slurStart"),
            note(slur="slurStart_slurStop"),
            EncodedSymbol("barline"),
            note(slur="slurStop"),
        ]
    )
    middle = xml.findall(".//note")[1]
    # Both endpoints on the middle note; MusicXML does not prescribe their order.
    assert sorted(marks(middle, "tie")) == ["start", "stop"]
    assert sorted(marks(middle, "notations/tied")) == ["start", "stop"]


def test_slurs_remain_slurs_and_preserve_other_articulations() -> None:
    xml = score(
        [note(articulation="slurStart_staccato"), note(pitch="D4", articulation="slurStop")]
    )
    assert marks(xml, ".//slur") == ["start", "stop"]
    assert xml.find(".//staccato") is not None
    assert xml.find(".//tie") is None
    assert xml.find(".//tied") is None


def test_chord_tie_stays_on_the_predicted_pitch() -> None:
    xml = score(
        [
            note("note_2", "C4", slur="slurStart"),
            EncodedSymbol("chord"),
            note("note_2", "E4"),
            note("note_2", "C4", slur="slurStop"),
            EncodedSymbol("chord"),
            note("note_2", "E4"),
        ]
    )
    tied_notes = [n for n in xml.findall(".//note") if n.find("tie") is not None]
    assert [n.findtext("pitch/step") for n in tied_notes] == ["C", "C"]
    assert [marks(n, "tie") for n in tied_notes] == [["start"], ["stop"]]


@pytest.mark.parametrize(
    "ending", [[], [note()], [EncodedSymbol("barline")], [EncodedSymbol("newline")]]
)
def test_triplet_closes_on_its_last_note(ending: list[EncodedSymbol]) -> None:
    xml = score([note("note_12", p) for p in ("C4", "D4", "E4")] + ending)
    notes = xml.findall(".//note")
    assert [marks(n, "notations/tuplet") for n in notes[:3]] == [["start"], [], ["stop"]]
    assert all(not marks(n, "notations/tuplet") for n in notes[3:])
    assert [n.findtext("time-modification/actual-notes") for n in notes[:3]] == ["3"] * 3
    division = int(xml.findtext(".//divisions", "0"))
    assert sum(Fraction(int(n.findtext("duration", "0")), division) for n in notes[:3]) == 1


def test_two_adjacent_triplets_have_separate_boundaries() -> None:
    xml = score([note("note_12", p) for p in ("C4", "D4", "E4", "F4", "G4", "A4")])
    assert [marks(n, "notations/tuplet") for n in xml.findall(".//note")] == [
        ["start"],
        [],
        ["stop"],
        ["start"],
        [],
        ["stop"],
    ]


@pytest.mark.xfail(strict=True, reason="upstream exporter closes dotted-triplet groups early")
def test_dotted_triplets_keep_a_three_note_group() -> None:
    xml = score([note("note_12.", p) for p in ("C4", "D4", "E4")])
    assert [marks(n, "notations/tuplet") for n in xml.findall(".//note")] == [
        ["start"],
        [],
        ["stop"],
    ]


def test_changing_tuplet_ratio_closes_the_previous_group() -> None:
    xml = score([note("note_12")] * 3 + [note("note_20")] * 5)
    notes = xml.findall(".//note")
    assert [marks(n, "notations/tuplet") for n in notes] == [
        ["start"],
        [],
        ["stop"],
        ["start"],
        [],
        [],
        [],
        ["stop"],
    ]
    assert notes[3].findtext("time-modification/actual-notes") == "5"
    assert notes[3].findtext("time-modification/normal-notes") == "4"


def test_lower_staff_ordinary_notes_do_not_inherit_upper_tuplet_brackets() -> None:
    symbols = []
    for pitch in ("C4", "D4", "E4"):
        symbols.extend(
            [
                note("note_12", pitch),
                EncodedSymbol("chord"),
                note("note_4", "C3", position="lower"),
            ]
        )
    xml = score(symbols)
    assert marks(xml, ".//tuplet") == ["start", "stop"]
    for n in xml.findall(".//note"):
        if n.findtext("staff") == "2":
            assert n.find("notations/tuplet") is None


def test_tuplet_rest_and_mixed_note_values_keep_exact_duration() -> None:
    xml = score([note("note_6"), note("rest_12", "_"), note("note_2.")])
    notes = xml.findall(".//note")
    assert [marks(n, "notations/tuplet") for n in notes] == [["start"], ["stop"], []]
    assert notes[1].find("rest") is not None
    division = int(xml.findtext(".//divisions", "0"))
    assert sum(Fraction(int(n.findtext("duration", "0")), division) for n in notes) == 4


def test_single_tuplet_event_can_start_and_stop_on_the_same_note() -> None:
    xml = score([note("note_12")])
    assert marks(xml, ".//tuplet") == ["start", "stop"]


@pytest.mark.xfail(strict=True, reason="upstream exporter brackets chord tones separately")
def test_chord_tones_do_not_duplicate_tuplet_brackets() -> None:
    symbols = []
    for _ in range(3):
        symbols.extend([note("note_12", "C4"), EncodedSymbol("chord"), note("note_12", "E4")])
    xml = score(symbols)
    assert len(xml.findall(".//time-modification")) == 6
    assert marks(xml, ".//tuplet") == ["start", "stop"]


def test_ordinary_notes_are_not_rewritten_as_tuplets() -> None:
    xml = score([note("note_8"), note("note_8"), note("note_4.")])
    assert xml.find(".//time-modification") is None
    assert xml.find(".//tuplet") is None


def test_jazzify_preserves_a_short_triplet_pickup() -> None:
    pickup = [note("note_12", p) for p in ("C4", "D4", "E4")] + [EncodedSymbol("barline")]
    symbols = pickup + [note("note_1"), EncodedSymbol("barline")] * 2
    # Upstream's heuristic unscales a short triplet pickup; Jazzify switches it off.
    assert [s.rhythm for s in remove_duplicated_symbols(symbols)[:3]] != ["note_12"] * 3
    with upstream_experiment(hm, preserve_tuplets=True, filter_header=False):
        cleaned = remove_duplicated_symbols(symbols)
    assert [s.rhythm for s in cleaned[:3]] == ["note_12"] * 3


def test_cleanup_still_repairs_tuplets_when_unscaling_exactly_fills_measure() -> None:
    symbols = [note("note_6")] * 4 + [EncodedSymbol("barline")]
    symbols += [note("note_1"), EncodedSymbol("barline")] * 2
    cleaned = remove_duplicated_symbols(symbols)
    assert [s.rhythm for s in cleaned[:4]] == ["note_4"] * 4
