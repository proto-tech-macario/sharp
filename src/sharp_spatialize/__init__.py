"""sharp_spatialize: converts one RGB image into a 9-view spatial photo."""

from .hdf5_io import SpatialPhotoResult
from .validation import ValidationError

__all__ = ["generate_spatial_photo", "load_spatial_photo", "SpatialPhotoResult", "ValidationError"]


def __getattr__(name: str):
    """Lazy imports for functions that may require SHARP/CUDA.

    generate_spatial_photo is imported from .api only when accessed, since it
    genuinely needs SHARP/CUDA for inference and rendering. load_spatial_photo
    is imported from .hdf5_io directly (not .api) so that merely accessing it
    never pulls in the inference/SHARP/CUDA stack -- reading a spatial photo
    back never needs SHARP or a GPU.
    """
    if name == "generate_spatial_photo":
        from .api import generate_spatial_photo
        return generate_spatial_photo
    if name == "load_spatial_photo":
        from .hdf5_io import load as load_spatial_photo
        return load_spatial_photo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
