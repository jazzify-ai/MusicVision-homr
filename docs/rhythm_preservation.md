> **Superseded (2026-11-12).** This fork now runs upstream HOMR at commit `dbd17e23` with model
> run 509, and uses upstream's MusicXML exporter, which already writes ties (same-pitch slurs
> become `<tie>` + `<tied>`) and tuplet brackets. Short-bar handling moved to the Jazzify
> bar-length beam search (`homr/jazzify/`). The regression tests in
> `tests/test_rhythm_preservation.py` were kept and adapted; two bracket edge cases that
> upstream's exporter handles differently are marked `xfail`. The original text follows for
> history.

# Rhythm preservation before model training

The November 2026 investigation separated decoder misses from symbols lost after
decoding. Run 331 missed all four printed triplet groups in the Airegin (Ab)
benchmark page and two of three groups in 'Round Midnight. This change does not
replace or train that model.

## Corrected conversion behavior

- Keep each note's predicted slur/tie articulation instead of stripping it at
  chord conversion. A tie exports both sounding `<tie>` and visible `<tied>`
  elements. Continuations stop the preceding tie before starting the next.
- Keep tie markers on their predicted chord tone. Slurs remain slurs; this code
  does not infer that a same-pitch slur was intended to be a tie.
- Write tuplet stops on the final tuplet note/rest, including at a barline or the
  end of a score. Do not attach a stop to the following ordinary note.
- Split adjacent simple tuplets using the durations already present in the
  token stream. Include rests and dotted durations. Keep marks separate between
  upper/lower staffs and avoid duplicating brackets on chord tones.
- Only remove tuplets from a short measure when unscaling them exactly matches
  the typical measure duration. A short duration by itself is insufficient.

The cleanup remains a heuristic: the old vocabulary does not encode a complete
time signature or explicit tuplet groups. The bracket inference covers simple,
unnested same-ratio runs and uses the smallest written value in a run. It does
not recover the original engraving of ambiguous mixed-duration groups or nested
tuplets. Multiple overlapping voices on one staff remain a broader limitation
of this exporter. The mixed-staff eighth/tuplet case can also expose an existing
division-resolution issue and needs separate work; it is not solved here.

These changes preserve decoder-provided durations and pitches during export.
They do not restore missing predictions, validate every predicted tie, or claim
a recognition accuracy improvement.

## Verification

`tests/test_rhythm_preservation.py` uses complete MusicXML output to verify
sounding/visible ties, chord-tone ownership, slur preservation, tuplet endpoints,
ordinary-note controls, rational durations, dotted/mixed values, and short pickup
preservation. The original defect tests failed before the fixes.

Run the existing suite and checks from this checkout:

```powershell
& .\.venv\Scripts\python.exe -X utf8 -m pytest tests -q -o log_cli=false -o addopts=""
& .\.venv\Scripts\python.exe -m black --check homr/music_xml_generator.py homr/transformer/vocabulary.py tests/test_rhythm_preservation.py tests/test_music_xml_generator.py
& .\.venv\Scripts\python.exe -m isort --check-only homr/music_xml_generator.py homr/transformer/vocabulary.py tests/test_rhythm_preservation.py tests/test_music_xml_generator.py
& .\.venv\Scripts\python.exe -m ruff check homr/music_xml_generator.py homr/transformer/vocabulary.py tests/test_rhythm_preservation.py tests/test_music_xml_generator.py
```

The working Modal deployment is unchanged. Local source changes only reach it
after a separate deployment. Rollback source tag:
`rollback/pre-homr-rhythm-setup-20261110`.

For an isolated model comparison, use the ML repository's
`docs/homr_rhythm_comparison.md`. The candidate keeps its own checkout, virtual
environment, vocabulary, and weights outside the production source tree.
