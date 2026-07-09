from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from homr.bounding_boxes import RotatedBoundingBox
from homr.model import MultiStaff, Staff
from homr.type_definitions import NDArray


@dataclass(frozen=True)
class BoxGeometry:
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class BarlineGeometry(BoxGeometry):
    center: tuple[float, float]
    angle: float


@dataclass(frozen=True)
class StaffGeometry(BoxGeometry):
    index: int
    average_unit_size: float


@dataclass(frozen=True)
class SystemGeometry(BoxGeometry):
    index: int
    staffs: list[StaffGeometry]


@dataclass(frozen=True)
class GeometryArtifact:
    coordinate_space: str
    image: dict[str, int]
    systems: list[SystemGeometry]
    barlines: list[BarlineGeometry]

    def to_dict(self) -> dict:
        return asdict(self)


def build_geometry_artifact(
    *,
    processed_image: NDArray,
    multi_staffs: list[MultiStaff],
    bar_line_boxes: list[RotatedBoundingBox],
) -> GeometryArtifact:
    height, width = processed_image.shape[:2]
    staff_index = 1
    systems: list[SystemGeometry] = []

    for system_index, multi_staff in enumerate(
        sorted(multi_staffs, key=_system_sort_key),
        start=1,
    ):
        staffs: list[StaffGeometry] = []
        for staff in multi_staff.staffs:
            staffs.append(_staff_geometry(staff, staff_index))
            staff_index += 1

        systems.append(
            SystemGeometry(
                index=system_index,
                bbox=_bbox_union([staff.bbox for staff in staffs]),
                staffs=staffs,
            )
        )

    barlines = [
        BarlineGeometry(
            bbox=_bbox_from_rotated_box(box),
            center=(float(box.center[0]), float(box.center[1])),
            angle=float(box.angle),
        )
        for box in bar_line_boxes
    ]

    return GeometryArtifact(
        coordinate_space="homr_processed_image",
        image={"width": int(width), "height": int(height)},
        systems=systems,
        barlines=barlines,
    )


def write_geometry_json(*, artifact: GeometryArtifact, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.to_dict(), indent=2), encoding="utf-8")


def write_processed_image(*, processed_image: NDArray, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), processed_image):
        raise RuntimeError(f"Failed to write processed HOMR image to {path}")


def _system_sort_key(multi_staff: MultiStaff) -> float:
    return min(staff.min_y for staff in multi_staff.staffs)


def _staff_geometry(staff: Staff, index: int) -> StaffGeometry:
    return StaffGeometry(
        index=index,
        bbox=(
            float(staff.min_x),
            float(staff.min_y),
            float(staff.max_x),
            float(staff.max_y),
        ),
        average_unit_size=float(staff.average_unit_size),
    )


def _bbox_from_rotated_box(box: RotatedBoundingBox) -> tuple[float, float, float, float]:
    points = np.asarray(box.polygon)
    xs = points[:, 0]
    ys = points[:, 1]
    return (
        float(np.min(xs)),
        float(np.min(ys)),
        float(np.max(xs)),
        float(np.max(ys)),
    )


def _bbox_union(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)

    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )
