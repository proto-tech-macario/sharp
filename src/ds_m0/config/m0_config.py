"""M0 configuration objects. Every algorithmic choice is a named, recorded value."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError

ROUNDING_MODES = ("half_up", "half_even")
DEPTH_CODECS = ("float32_zlib", "uint16_quant_zlib")
SUBSAMPLING = ("4:4:4", "4:2:2", "4:2:0")


@dataclass
class InputConfig:
    dataset_path: str | None = None


@dataclass
class BuilderConfig:
    base_view_id: int = 4
    relative_depth_threshold: float = 0.02  # experimental starting point, not a standard
    absolute_depth_threshold: float = 0.0
    depth_epsilon: float = 1e-5
    rounding_mode: str = "half_up"

    def validate(self) -> None:
        if not isinstance(self.base_view_id, int) or self.base_view_id < 0:
            raise ConfigurationError(f"base_view_id must be a non-negative int, got {self.base_view_id!r}")
        if not (self.relative_depth_threshold >= 0 and self.absolute_depth_threshold >= 0):
            raise ConfigurationError("depth thresholds must be >= 0")
        if self.relative_depth_threshold == 0 and self.absolute_depth_threshold == 0:
            raise ConfigurationError("at least one depth threshold must be positive")
        if not self.depth_epsilon >= 0:
            raise ConfigurationError("depth_epsilon must be >= 0")
        if self.rounding_mode not in ROUNDING_MODES:
            raise ConfigurationError(f"rounding_mode must be one of {ROUNDING_MODES}")


@dataclass
class DSImageWriteConfig:
    jpeg_quality: int = 90
    jpeg_subsampling: str = "4:2:0"
    depth_codec: str = "float32_zlib"
    sparse_layer1_codec: str = "bitmask_raster_v1"
    zlib_level: int = 9

    def validate(self) -> None:
        if not 1 <= self.jpeg_quality <= 100:
            raise ConfigurationError("jpeg_quality must be in 1..100")
        if self.jpeg_subsampling not in SUBSAMPLING:
            raise ConfigurationError(f"jpeg_subsampling must be one of {SUBSAMPLING}")
        if self.depth_codec not in DEPTH_CODECS:
            raise ConfigurationError(f"depth_codec must be one of {DEPTH_CODECS}")
        if self.sparse_layer1_codec != "bitmask_raster_v1":
            raise ConfigurationError("only sparse_layer1_codec 'bitmask_raster_v1' exists in M0")
        if not 0 <= self.zlib_level <= 9:
            raise ConfigurationError("zlib_level must be in 0..9")


@dataclass
class DSImageReadConfig:
    verify_checksums: bool = True


@dataclass
class PresenterConfig:
    depth_epsilon: float = 1e-5
    rounding_mode: str = "half_up"
    output_width: int | None = None  # None: use the target camera's width
    output_height: int | None = None
    enable_coverage_output: bool = True

    def validate(self) -> None:
        if not self.depth_epsilon >= 0:
            raise ConfigurationError("depth_epsilon must be >= 0")
        if self.rounding_mode not in ROUNDING_MODES:
            raise ConfigurationError(f"rounding_mode must be one of {ROUNDING_MODES}")
        for name in ("output_width", "output_height"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise ConfigurationError(f"{name} must be positive")


@dataclass
class EvaluationConfig:
    output_directory: str = "ds_m0_output"
    # Target-camera offsets in base-camera axes (x right, y down, z forward).
    target_offsets: list[list[float]] = field(
        default_factory=lambda: [
            [0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [-0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, -0.05, 0.0],
        ]
    )
    offsets_scaled_by_median_depth: bool = True
    rgb_mean_abs_tolerance: float = 12.0  # sanity bound for JPEG error (tiny hard-edged images are the worst case); not a DS standard
    depth_extra_tolerance: float = 1e-6
    camera_math_epsilon: float = 1e-9
    miv_file: str | None = None


@dataclass
class M0Config:
    input: InputConfig = field(default_factory=InputConfig)
    builder: BuilderConfig = field(default_factory=BuilderConfig)
    writer: DSImageWriteConfig = field(default_factory=DSImageWriteConfig)
    reader: DSImageReadConfig = field(default_factory=DSImageReadConfig)
    presenter: PresenterConfig = field(default_factory=PresenterConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        self.builder.validate()
        self.writer.validate()
        self.presenter.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "M0Config":
        cfg = M0Config()
        for f in fields(M0Config):
            section = data.get(f.name)
            if section is None:
                continue
            target = getattr(cfg, f.name)
            valid = {x.name for x in fields(target)}
            unknown = set(section) - valid
            if unknown:
                raise ConfigurationError(f"unknown {f.name} config keys: {sorted(unknown)}")
            for k, v in section.items():
                setattr(target, k, v)
        unknown_top = set(data) - {f.name for f in fields(M0Config)}
        if unknown_top:
            raise ConfigurationError(f"unknown config sections: {sorted(unknown_top)}")
        cfg.validate()
        return cfg

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @staticmethod
    def load(path: str | Path) -> "M0Config":
        try:
            return M0Config.from_dict(json.loads(Path(path).read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"cannot read config {path}: {exc}") from exc
