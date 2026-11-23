"""Jazzify accuracy changes for HOMR on lead sheets, switched on for every ``process_image``.

What is enabled (letters as in MusicVision-ml-private ``docs/homr_decoding_experiments.md``):

- Tuplet preservation: upstream's two tuplet heuristics (``_fix_over_eager_tuplets``,
  ``restore_unmarked_triplets``) are disabled; the bar-length beam search below handles short
  bars instead.
- Title filter: a leading staff-like title decoration with no notes, clef or barlines is dropped.
- B: the clef/key read on the last staff that printed them is forced on unmarked staffs
  (lead sheets print them on the first line only); misread printed clefs are re-decoded.
- C: consistent heads (notes always get a pitch, rests none) and a width-8 beam search with a
  soft penalty for bars that do not add up to the page's reference bar length.
- E: orphan slur stop -> tie, low-confidence ornaments dropped, repeated/odd time signatures
  normalised, and beams written into the MusicXML.
- Clef/key strip: on an unmarked line under a carried key with flats or sharps, the printed
  clef + key-signature image of the last marked line is pasted in front of the line, so notes are
  read with the key in view ('Round Midnight 14 -> 18/18 with run 12, 2026-11-22). Polish scores
  and sheet-music photos: no page changes by more than 2 points (2026-11-23).
- Displayed accidentals (``homr.accidentals``).

Measured before porting (2026-11-12): Airegin 37/38 exact bars (from 29 on run 331),
'Round Midnight 15/18 (from 3), Omnibook validation 20% exact bars and EER 41.7% (from 10% /
68.2%). Set ``HOMR_JAZZIFY_DECODING=0`` to fall back to plain upstream decoding.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

from homr import accidentals
from homr.jazzify.decoding import DecodeOptions, add_beams, decoding_experiment
from homr.jazzify.refinement import upstream_experiment

PRODUCTION_OPTIONS = DecodeOptions(
    carry_clef_key=True,
    carry_across_pages=True,
    consistent_heads=True,
    beam_width=8,
    bar_penalty=3.0,
    meter_change_penalty=1.0,
    ornament_min_prob=0.75,
    cleanup=True,
    paste_clef_key=True,
)


def enabled() -> bool:
    return os.getenv("HOMR_JAZZIFY_DECODING", "1").strip().lower() not in {"0", "false", "no"}


@contextmanager
def jazzify_processing(
    main_module: ModuleType, carry: dict | None = None, options: DecodeOptions | None = None
) -> Iterator[dict]:
    """Install the Jazzify hooks into ``homr.main`` for one page; yields diagnostic evidence.

    ``carry`` is shared between the pages of one document (clef/key context).
    The hooks replace module attributes for the duration of the call, so pages must be
    processed one at a time per process (the service handles one request at a time).
    """
    if not enabled():
        yield {}
        return
    options = options or PRODUCTION_OPTIONS
    with (
        upstream_experiment(main_module, preserve_tuplets=True, filter_header=True) as refinement,
        decoding_experiment(main_module, options, carry) as decoding,
    ):
        yield {"refinement": refinement, "decoding": decoding}


def postprocess_musicxml(path: str | Path) -> dict:
    """Displayed accidentals and, with Jazzify decoding on, beams. Rewrites the file in place."""
    tree = ET.parse(path)  # noqa: S314 - MusicXML written by this process
    root = tree.getroot()
    result = {"accidentals_added": accidentals.add_display_accidentals(root)}
    if enabled() and PRODUCTION_OPTIONS.cleanup:
        result["beamed_notes"] = add_beams(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="UTF-8", xml_declaration=True)
    return result
