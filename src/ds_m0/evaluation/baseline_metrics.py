from __future__ import annotations

from pathlib import Path

from ..errors import InputError


def measure_miv_baseline(ds_image_path: str | Path, miv_path: str | Path) -> dict:
    """Storage-only comparison. Says nothing about rendering quality (spec: M1)."""
    miv = Path(miv_path)
    if not miv.is_file():
        raise InputError(f"MIV file not found: {miv}")
    ds_size, miv_size = Path(ds_image_path).stat().st_size, miv.stat().st_size
    return {
        "miv_total_size": miv_size, "ds_total_size": ds_size,
        "ds_to_miv_ratio": ds_size / miv_size if miv_size else None,
        "note": "storage experiment only; no quality claim (see spec Part IV section 24)",
    }
