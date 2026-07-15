import io
import json
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

import homr_service.main as service


def test_health_and_source(monkeypatch) -> None:
    monkeypatch.setenv("HOMR_PRELOAD_MODELS", "false")
    monkeypatch.setenv("HOMR_SOURCE_URL", "https://example.test/homr")

    with TestClient(service.app) as client:
        health = client.get("/health")
        source = client.get("/source")

    assert health.status_code == 200
    assert health.json()["license"] == "AGPL-3.0"
    assert source.json() == {
        "source_url": "https://example.test/homr",
        "license": "AGPL-3.0",
    }


def test_old_shared_path_endpoints_are_not_exposed(monkeypatch) -> None:
    monkeypatch.setenv("HOMR_PRELOAD_MODELS", "false")

    with TestClient(service.app) as client:
        full_response = client.post("/v1/omr", json={})
        geometry_response = client.post("/v1/geometry", json={})

    assert full_response.status_code == 404
    assert geometry_response.status_code == 404


def test_full_upload_success_with_mocked_homr(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOMR_PRELOAD_MODELS", "false")
    monkeypatch.setattr(service, "_run_homr", _fake_run_homr)

    with TestClient(service.app) as client:
        response = client.post(
            "/v1/omr/upload",
            data={"job_id": "job-1", "mode": "full"},
            files={"file": ("preprocessed.png", b"fake-image", "image/png")},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    entries = _zip_entries(response.content)
    assert entries["score.musicxml"] == b"<score-partwise/>"
    assert entries["geometry.json"] == b"{}"
    assert entries["homr_processed.png"] == b"fake"
    manifest = json.loads(entries["manifest.json"].decode("utf-8"))
    assert manifest["mode"] == "full"
    assert manifest["artifacts"] == {
        "musicxml_path": "score.musicxml",
        "geometry_json_path": "geometry.json",
        "processed_image_path": "homr_processed.png",
    }
    assert manifest["source_url"].endswith("Jazzify-MusicVision-homr-agpl")


def test_geometry_upload_success_with_mocked_homr(monkeypatch) -> None:
    monkeypatch.setenv("HOMR_PRELOAD_MODELS", "false")
    monkeypatch.setattr(service, "_run_homr", _fake_run_homr)

    with TestClient(service.app) as client:
        response = client.post(
            "/v1/geometry/upload",
            data={"job_id": "job-1", "mode": "geometry"},
            files={"file": ("preprocessed.png", b"fake-image", "image/png")},
        )

    assert response.status_code == 200
    entries = _zip_entries(response.content)
    assert "score.musicxml" not in entries
    assert entries["geometry.json"] == b"{}"
    assert entries["homr_processed.png"] == b"fake"
    manifest = json.loads(entries["manifest.json"].decode("utf-8"))
    assert manifest["mode"] == "geometry"
    assert manifest["artifacts"]["musicxml_path"] is None


def test_homr_failure_returns_structured_error(monkeypatch) -> None:
    monkeypatch.setenv("HOMR_PRELOAD_MODELS", "false")

    def fail_homr(*, input_path: Path, output_dir: Path, geometry_only: bool) -> None:
        raise RuntimeError("model failure")

    monkeypatch.setattr(service, "_run_homr", fail_homr)

    with TestClient(service.app) as client:
        response = client.post(
            "/v1/omr/upload",
            data={"job_id": "job-1", "mode": "full"},
            files={"file": ("preprocessed.png", b"fake-image", "image/png")},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "error": "HOMR processing failed",
        "message": "model failure",
    }


def _fake_run_homr(*, input_path: Path, output_dir: Path, geometry_only: bool) -> None:
    assert input_path.name == "input.png"
    assert input_path.read_bytes() == b"fake-image"
    (output_dir / "geometry.json").write_text("{}", encoding="utf-8")
    (output_dir / "homr_processed.png").write_bytes(b"fake")
    if not geometry_only:
        (output_dir / "score.musicxml").write_text("<score-partwise/>", encoding="utf-8")


def _zip_entries(content: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}
