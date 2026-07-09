from __future__ import annotations

import os
import shutil
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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


class OmrRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=128)
    input_image_path: str
    output_dir: str
    mode: Literal["full", "geometry"] = "full"


class ArtifactPaths(BaseModel):
    musicxml_path: str | None
    geometry_json_path: str
    processed_image_path: str


class OmrResponse(BaseModel):
    job_id: str
    mode: Literal["full", "geometry"]
    artifacts: ArtifactPaths
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


@app.post("/v1/omr", response_model=OmrResponse)
def run_full_omr(request: OmrRequest) -> OmrResponse:
    return _process_homr_request(request, geometry_only=False)


@app.post("/v1/geometry", response_model=OmrResponse)
def run_geometry(request: OmrRequest) -> OmrResponse:
    return _process_homr_request(request, geometry_only=True)


def _process_homr_request(request: OmrRequest, *, geometry_only: bool) -> OmrResponse:
    if geometry_only and request.mode != "geometry":
        raise HTTPException(status_code=400, detail={"error": "mode must be geometry"})
    if not geometry_only and request.mode != "full":
        raise HTTPException(status_code=400, detail={"error": "mode must be full"})

    input_path = _resolve_under_shared_jobs(request.input_image_path)
    output_dir = _resolve_under_shared_jobs(request.output_dir)
    if not input_path.is_file():
        raise HTTPException(status_code=400, detail={"error": f"input image not found: {input_path}"})

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        _run_homr(input_path=input_path, output_dir=output_dir, geometry_only=geometry_only)
    except InvalidProgramArgumentException as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "HOMR processing failed", "message": str(exc)},
        ) from exc

    artifacts = _artifact_paths(output_dir, geometry_only=geometry_only)
    _validate_artifacts(artifacts, geometry_only=geometry_only)
    total_ms = int((time.perf_counter() - started) * 1000)
    return OmrResponse(
        job_id=request.job_id,
        mode="geometry" if geometry_only else "full",
        artifacts=artifacts,
        timings_ms={"total": total_ms},
        source_url=_source_url(),
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


def _resolve_under_shared_jobs(raw_path: str) -> Path:
    shared_root = _shared_jobs_root()
    path = Path(raw_path).expanduser().resolve()
    try:
        path.relative_to(shared_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": f"path must resolve under {shared_root}: {raw_path}"},
        ) from exc
    return path


def _shared_jobs_root() -> Path:
    return Path(os.getenv("HOMR_SHARED_JOBS_ROOT", "/shared/jobs")).expanduser().resolve()


def _source_url() -> str:
    return os.getenv("HOMR_SOURCE_URL", DEFAULT_SOURCE_URL)


def _use_gpu_inference() -> bool:
    raw_gpu_mode = os.getenv("HOMR_GPU", GpuSupport.AUTO.value).strip().lower()
    gpu_mode = GpuSupport(raw_gpu_mode) if raw_gpu_mode in {item.value for item in GpuSupport} else GpuSupport.AUTO
    has_gpu_support = "CUDAExecutionProvider" in ort.get_available_providers()
    return (gpu_mode == GpuSupport.AUTO and has_gpu_support) or gpu_mode == GpuSupport.FORCE
