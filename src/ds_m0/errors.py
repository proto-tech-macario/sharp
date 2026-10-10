"""Error model (spec Part III section 54). Each class maps to one CLI exit code."""

from __future__ import annotations


class DSError(Exception):
    """Base class of every error raised deliberately by ds_m0."""

    exit_code = 1


class ConfigurationError(DSError):
    """Invalid user configuration."""

    exit_code = 2


class InputError(DSError):
    """Invalid source dataset."""

    exit_code = 3


class GeometryError(DSError):
    """Invalid camera or unusable geometry."""

    exit_code = 4


class FormatError(DSError):
    """Malformed DS-Image."""

    exit_code = 5


class DSRuntimeError(DSError):
    """Unexpected processing failure."""

    exit_code = 6


class ValidationError(DSError):
    """The software ran but the result violates a declared invariant."""

    exit_code = 7
