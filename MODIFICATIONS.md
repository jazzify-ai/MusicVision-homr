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
