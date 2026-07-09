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


def test_rejects_path_traversal(tmp_path: Path, monkeypatch) -> None:
    shared_root = tmp_path / "shared" / "jobs"
    shared_root.mkdir(parents=True)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"fake")
    monkeypatch.setenv("HOMR_PRELOAD_MODELS", "false")
    monkeypatch.setenv("HOMR_SHARED_JOBS_ROOT", str(shared_root))

    with TestClient(service.app) as client:
        response = client.post(
            "/v1/omr",
            json={
                "job_id": "job-1",
                "input_image_path": str(outside),
                "output_dir": str(shared_root / "job-1" / "output"),
                "mode": "full",
            },
        )

    assert response.status_code == 400
    assert "path must resolve under" in response.json()["detail"]["error"]


def test_full_mode_success_with_mocked_homr(tmp_path: Path, monkeypatch) -> None:
    shared_root, input_path, output_dir = _job_paths(tmp_path)
    monkeypatch.setenv("HOMR_PRELOAD_MODELS", "false")
    monkeypatch.setenv("HOMR_SHARED_JOBS_ROOT", str(shared_root))
    monkeypatch.setattr(service, "_run_homr", _fake_run_homr)

    with TestClient(service.app) as client:
        response = client.post(
            "/v1/omr",
            json={
                "job_id": "job-1",
                "input_image_path": str(input_path),
                "output_dir": str(output_dir),
                "mode": "full",
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["mode"] == "full"
    assert payload["artifacts"]["musicxml_path"] == str((output_dir / "score.musicxml").resolve())
    assert payload["artifacts"]["geometry_json_path"] == str((output_dir / "geometry.json").resolve())
    assert payload["source_url"].endswith("Jazzify-homr-agpl")


def test_geometry_mode_success_with_mocked_homr(tmp_path: Path, monkeypatch) -> None:
    shared_root, input_path, output_dir = _job_paths(tmp_path)
    monkeypatch.setenv("HOMR_PRELOAD_MODELS", "false")
    monkeypatch.setenv("HOMR_SHARED_JOBS_ROOT", str(shared_root))
    monkeypatch.setattr(service, "_run_homr", _fake_run_homr)

    with TestClient(service.app) as client:
        response = client.post(
            "/v1/geometry",
            json={
                "job_id": "job-1",
                "input_image_path": str(input_path),
                "output_dir": str(output_dir),
                "mode": "geometry",
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["mode"] == "geometry"
    assert payload["artifacts"]["musicxml_path"] is None
    assert not (output_dir / "score.musicxml").exists()


def test_homr_failure_returns_structured_error(tmp_path: Path, monkeypatch) -> None:
    shared_root, input_path, output_dir = _job_paths(tmp_path)
    monkeypatch.setenv("HOMR_PRELOAD_MODELS", "false")
    monkeypatch.setenv("HOMR_SHARED_JOBS_ROOT", str(shared_root))

    def fail_homr(*, input_path: Path, output_dir: Path, geometry_only: bool) -> None:
        raise RuntimeError("model failure")

    monkeypatch.setattr(service, "_run_homr", fail_homr)

    with TestClient(service.app) as client:
        response = client.post(
            "/v1/omr",
            json={
                "job_id": "job-1",
                "input_image_path": str(input_path),
                "output_dir": str(output_dir),
                "mode": "full",
            },
        )

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "error": "HOMR processing failed",
        "message": "model failure",
    }


def _job_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    shared_root = tmp_path / "shared" / "jobs"
    input_dir = shared_root / "job-1" / "input"
    output_dir = shared_root / "job-1" / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    input_path = input_dir / "preprocessed.png"
    input_path.write_bytes(b"fake-image")
    return shared_root, input_path, output_dir


def _fake_run_homr(*, input_path: Path, output_dir: Path, geometry_only: bool) -> None:
    (output_dir / "geometry.json").write_text("{}", encoding="utf-8")
    (output_dir / "homr_processed.png").write_bytes(b"fake")
    if not geometry_only:
        (output_dir / "score.musicxml").write_text("<score-partwise/>", encoding="utf-8")
