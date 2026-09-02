"""Command-line entry points: `sharp-spatialize` and `sharp-spatialize-validate`."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .validation import ValidationError, validate_file


@click.command()
@click.option(
    "-i", "--input", "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True, help="Input JPEG/PNG image.",
)
@click.option(
    "-o", "--output", "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True, help="Output spatial_photo.h5 path.",
)
@click.option(
    "--angle", "angle_deg", type=float, default=10.0, show_default=True,
    help="Horizontal/vertical camera angle in degrees. Any positive float works; "
    "5, 10, and 15 are the values this tool is validated against.",
)
@click.option(
    "--width", "output_width", type=int, default=None,
    help="Output width in pixels (defaults to the source image width).",
)
@click.option(
    "--height", "output_height", type=int, default=None,
    help="Output height in pixels (defaults to the source image height).",
)
@click.option(
    "-c", "--checkpoint", "checkpoint_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
    help="SHARP checkpoint path (downloads the default model if omitted).",
)
@click.option(
    "--device", default="default",
    help="Device for SHARP inference: cpu, mps, cuda, or default (auto-detect). "
    "Rendering always requires CUDA regardless of this setting.",
)
def generate_cli(
    input_path: Path,
    output_path: Path,
    angle_deg: float,
    output_width: int | None,
    output_height: int | None,
    checkpoint_path: Path | None,
    device: str,
) -> None:
    """Convert one RGB image into a 9-view spatial_photo.h5 file."""
    from .api import generate_spatial_photo

    try:
        result = generate_spatial_photo(
            input_path, angle_deg=angle_deg, output_width=output_width,
            output_height=output_height, checkpoint_path=checkpoint_path, device=device,
        )
    except ValidationError as exc:
        click.echo(f"ERROR: generated scene failed validation: {exc}", err=True)
        sys.exit(1)

    result.save(output_path)
    click.echo(
        f"Wrote {output_path} "
        f"({result.rgb.shape[0]} views, {result.rgb.shape[2]}x{result.rgb.shape[1]})."
    )


@click.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate_cli(input_path: Path) -> None:
    """Validate an existing spatial_photo.h5 file. Does not require SHARP or CUDA."""
    report = validate_file(input_path)
    if report.ok:
        click.echo(f"PASS: {input_path} is a valid spatial photo.")
        return
    click.echo(f"FAIL: {input_path} failed validation:", err=True)
    for failure in report.failures:
        click.echo(f"  - {failure}", err=True)
    sys.exit(1)
