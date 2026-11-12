# Modifications

Base project: https://github.com/liebharc/homr

This repository contains the modified HOMR subtree split from the original
MusicVision working tree on 2026-07-10.

High-level changes in this split:

- Geometry artifact export support for `geometry.json`.
- Processed-image artifact export support for `homr_processed.png`.
- Geometry-only execution mode for staff/barline detection without MusicXML generation.
- FastAPI service wrapper in `homr_service/` exposing HOMR over an HTTP/file-artifact API.
- Container build that runs the HOMR service as a standalone AGPL component.
- Upstream base: HOMR commit `dbd17e23` (2026-11-09) with transformer model run 509, merged
  on 2026-11-12 from the earlier split (upstream `c257453`, model run 331). Upstream's
  MusicXML exporter (ties, tuplet brackets) replaces this fork's earlier exporter changes;
  `docs/rhythm_preservation.md` is kept as history.
- Lead-sheet decoding (`homr/jazzify/`), on for every `process_image` call and switchable off
  with `HOMR_JAZZIFY_DECODING=0`: tuplet preservation (two upstream tuplet heuristics off),
  removal of a decorative title staff, clef/key carried to staffs that do not print them,
  consistent note/rest heads, a width-8 beam search with a soft penalty for bars that do not
  add up, and cleanup (orphan slur -> tie, low-confidence ornaments, meter normalisation,
  beams). Developed and measured in MusicVision-ml-private (`docs/homr_decoding_experiments.md`):
  Airegin 29 -> 37 of 38 exact bars, 'Round Midnight 3 -> 15 of 18, Omnibook validation
  EER 68% -> 42%. About 2x the transformer time per page.
- Displayed accidentals: `homr/accidentals.py` adds `<accidental>` elements after MusicXML is
  written. The model predicts sounding pitches (`<alter>`), which renderers such as Verovio do
  not draw on their own. Sharps, flats and naturals are written where the pitch differs from
  the key signature or an earlier accidental in the same measure, staff and octave; tied
  continuations do not repeat them. `<alter>` values are unchanged.

Run locally:

```powershell
python -m pip install poetry
poetry install
poetry run uvicorn homr_service.main:app --host 127.0.0.1 --port 8010
```

Build the container:

```powershell
docker build -t jazzify-musicvision-homr .
```
