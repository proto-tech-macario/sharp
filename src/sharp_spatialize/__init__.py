"""sharp_spatialize: converts one RGB image into a 9-view spatial photo."""

from .hdf5_io import SpatialPhotoResult
from .validation import ValidationError

__all__ = ["generate_spatial_photo", "load_spatial_photo", "SpatialPhotoResult", "ValidationError"]


def __getattr__(name: str):
    """Lazy imports for functions that require SHARP/CUDA.

    generate_spatial_photo and load_spatial_photo are imported only when
    accessed, so code that only needs validate_cli, SpatialPhotoResult, or
    ValidationError never pulls in the inference/SHARP/CUDA stack.
    """
    if name == "generate_spatial_photo":
        from .api import generate_spatial_photo
        return generate_spatial_photo
    if name == "load_spatial_photo":
        from .api import load_spatial_photo
        return load_spatial_photo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
