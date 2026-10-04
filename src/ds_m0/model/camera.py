"""Canonical camera model. All geometry lives in `ds_m0.geometry`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Normalised internal convention: x_cam = R @ x_world + t; X right, Y down, Z forward.
COORDINATE_CONVENTION = "opencv_x_cam=R*x_world+t"
PROJECTION_PINHOLE = "pinhole"


@dataclass
class Camera:
    """Pinhole camera. `rotation`/`translation` are world->camera (float64)."""

    projection_type: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    skew: float
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    coordinate_convention: str = COORDINATE_CONVENTION

    @staticmethod
    def from_K_R_C(
        K: np.ndarray, R: np.ndarray, C: np.ndarray, width: int, height: int
    ) -> "Camera":
        """Build from the upstream (K, R, C) triple: x_cam = R @ (x_world - C)."""
        K = np.asarray(K, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        C = np.asarray(C, dtype=np.float64)
        return Camera(
            projection_type=PROJECTION_PINHOLE,
            width=int(width),
            height=int(height),
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            skew=float(K[0, 1]),
            rotation=R.copy(),
            translation=-R @ C,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "projection_type": self.projection_type,
            "width": int(self.width),
            "height": int(self.height),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "skew": float(self.skew),
            "rotation": np.asarray(self.rotation, dtype=np.float64).tolist(),
            "translation": np.asarray(self.translation, dtype=np.float64).tolist(),
            "coordinate_convention": self.coordinate_convention,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Camera":
        try:
            return Camera(
                projection_type=str(data["projection_type"]),
                width=int(data["width"]),
                height=int(data["height"]),
                fx=float(data["fx"]),
                fy=float(data["fy"]),
                cx=float(data["cx"]),
                cy=float(data["cy"]),
                skew=float(data.get("skew", 0.0)),
                rotation=np.array(data["rotation"], dtype=np.float64).reshape(3, 3),
                translation=np.array(data["translation"], dtype=np.float64).reshape(3),
                coordinate_convention=str(
                    data.get("coordinate_convention", COORDINATE_CONVENTION)
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            from ..errors import GeometryError

            raise GeometryError(f"malformed camera definition: {exc}") from exc

    def equals(self, other: "Camera") -> bool:
        """Exact equality (used for the round-trip structural check)."""
        return self.to_dict() == other.to_dict()
