from __future__ import annotations

import os
import json
import shutil
import tempfile
import time
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from homr.main import (
    GpuSupport,
    InvalidProgramArgumentException,
    ProcessingConfig,
    download_weights,
    process_image,
)
from homr.music_xml_generator import XmlGeneratorArguments
from homr.title_detection import download_ocr_weights

DEFAULT_SOURCE_URL = "https://github.com/<owner>/Jazzify-homr-agpl"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    del app
    preload_models()
    yield


app = FastAPI(title="Jazzify HOMR AGPL Service", version="0.1.0", lifespan=lifespan)


class ArtifactPaths(BaseModel):
    musicxml_path: str | None
    geometry_json_path: str
    processed_image_path: str


class ArtifactManifest(BaseModel):
    job_id: str
    mode: Literal["full", "geometry"]
    artifacts: dict[str, str | None]
    timings_ms: dict[str, int]
    source_url: str


def preload_models() -> None:
    if os.getenv("HOMR_PRELOAD_MODELS", "true").strip().lower() not in {"1", "true", "yes"}:
        return
    download_weights(_use_gpu_inference())
    download_ocr_weights()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "homr",
        "license": "AGPL-3.0",
        "source_url": _source_url(),
    }


@app.get("/source")
def source() -> dict[str, str]:
    return {
        "source_url": _source_url(),
        "license": "AGPL-3.0",
    }


@app.post("/v1/omr/upload")
def run_full_omr_upload(
    job_id: str = Form(..., min_length=1, max_length=128),
    mode: Literal["full"] = Form("full"),
    file: UploadFile = File(...),
) -> Response:
    del mode
    return _process_uploaded_homr_request(job_id=job_id, file=file, geometry_only=False)


@app.post("/v1/geometry/upload")
def run_geometry_upload(
    job_id: str = Form(..., min_length=1, max_length=128),
    mode: Literal["geometry"] = Form("geometry"),
    file: UploadFile = File(...),
) -> Response:
    del mode
    return _process_uploaded_homr_request(job_id=job_id, file=file, geometry_only=True)


def _process_uploaded_homr_request(
    *,
    job_id: str,
    file: UploadFile,
    geometry_only: bool,
) -> Response:
    suffix = Path(file.filename or "preprocessed.png").suffix.lower() or ".png"
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix=f"homr-{job_id}-") as temp_dir:
            work_dir = Path(temp_dir)
            input_path = work_dir / f"input{suffix}"
            output_dir = work_dir / "output"
            output_dir.mkdir()
            _write_upload_to_path(file, input_path)
            _run_homr(input_path=input_path, output_dir=output_dir, geometry_only=geometry_only)

            artifacts = _artifact_paths(output_dir, geometry_only=geometry_only)
            _validate_artifacts(artifacts, geometry_only=geometry_only)
            total_ms = int((time.perf_counter() - started) * 1000)
            manifest = ArtifactManifest(
                job_id=job_id,
                mode="geometry" if geometry_only else "full",
                artifacts={
                    "musicxml_path": "score.musicxml" if not geometry_only else None,
                    "geometry_json_path": "geometry.json",
                    "processed_image_path": "homr_processed.png",
                },
                timings_ms={"total": total_ms},
                source_url=_source_url(),
            )
            zip_bytes = _build_artifact_zip(
                output_dir=output_dir,
                manifest=manifest,
                geometry_only=geometry_only,
            )
    except InvalidProgramArgumentException as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "HOMR processing failed", "message": str(exc)},
        ) from exc

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{job_id}-homr-artifacts.zip"',
            "X-HOMR-Source-URL": _source_url(),
        },
    )


def _run_homr(*, input_path: Path, output_dir: Path, geometry_only: bool) -> None:
    geometry_json_path = output_dir / "geometry.json"
    processed_image_path = output_dir / "homr_processed.png"
    config = ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        read_staff_positions=False,
        selected_staff=-1,
        use_gpu_inference=_use_gpu_inference(),
    )
    xml_generator_args = XmlGeneratorArguments(
        output_large_page=False,
        output_metronome=None,
        output_tempo=None,
    )
    process_image(
        str(input_path),
        config,
        xml_generator_args,
        geometry_json_path=str(geometry_json_path),
        processed_image_path=str(processed_image_path),
        geometry_only=geometry_only,
    )
    if not geometry_only:
        generated_musicxml_path = input_path.with_suffix(".musicxml")
        musicxml_path = output_dir / "score.musicxml"
        if generated_musicxml_path.exists():
            shutil.move(str(generated_musicxml_path), musicxml_path)


def _write_upload_to_path(file: UploadFile, input_path: Path) -> None:
    with input_path.open("wb") as output_file:
        shutil.copyfileobj(file.file, output_file)


def _build_artifact_zip(
    *,
    output_dir: Path,
    manifest: ArtifactManifest,
    geometry_only: bool,
) -> bytes:
    zip_path = output_dir / "homr_artifacts.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", json.dumps(_model_to_dict(manifest), indent=2))
        if not geometry_only:
            archive.write(output_dir / "score.musicxml", "score.musicxml")
        archive.write(output_dir / "geometry.json", "geometry.json")
        archive.write(output_dir / "homr_processed.png", "homr_processed.png")
    return zip_path.read_bytes()


def _model_to_dict(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _artifact_paths(output_dir: Path, *, geometry_only: bool) -> ArtifactPaths:
    return ArtifactPaths(
        musicxml_path=None if geometry_only else str((output_dir / "score.musicxml").resolve()),
        geometry_json_path=str((output_dir / "geometry.json").resolve()),
        processed_image_path=str((output_dir / "homr_processed.png").resolve()),
    )


def _validate_artifacts(artifacts: ArtifactPaths, *, geometry_only: bool) -> None:
    required_paths = [
        Path(artifacts.geometry_json_path),
        Path(artifacts.processed_image_path),
    ]
    if not geometry_only and artifacts.musicxml_path is not None:
        required_paths.append(Path(artifacts.musicxml_path))
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise HTTPException(
            status_code=500,
            detail={"error": "HOMR did not produce expected artifacts", "missing": missing},
        )


def _source_url() -> str:
    return os.getenv("HOMR_SOURCE_URL", DEFAULT_SOURCE_URL)


def _use_gpu_inference() -> bool:
    raw_gpu_mode = os.getenv("HOMR_GPU", GpuSupport.AUTO.value).strip().lower()
    gpu_mode = GpuSupport(raw_gpu_mode) if raw_gpu_mode in {item.value for item in GpuSupport} else GpuSupport.AUTO
    has_gpu_support = "CUDAExecutionProvider" in ort.get_available_providers()
    return (gpu_mode == GpuSupport.AUTO and has_gpu_support) or gpu_mode == GpuSupport.FORCE
