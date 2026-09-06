"""A local web UI for previewing spatial photos.

`python -m sharp_spatialize.webui` (or the `sharp-spatialize-webui` script)
serves a page that takes an uploaded image, runs the same pipeline as
`sharp-spatialize`, and previews the resulting 9 views with real parallax.

Standard library + numpy/Pillow only -- no web framework, nothing to install
beyond what SHARP already needs.
"""

from .jobs import Job, JobParams, JobStore

__all__ = ["Job", "JobParams", "JobStore", "serve"]


def __getattr__(name: str):
    """Lazily expose `serve` so importing the package does not import torch."""
    if name == "serve":
        from .server import serve
        return serve
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
