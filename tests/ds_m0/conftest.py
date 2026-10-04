"""Shared fixtures. Nothing here needs torch, SHARP, a GPU or network access."""

from __future__ import annotations

import pytest
from ds_m0.config.m0_config import BuilderConfig
from ds_m0.creator import build_ds_asset
from ds_m0.tools.synthetic import make_synthetic_dataset


@pytest.fixture(scope="session")
def dataset():
    return make_synthetic_dataset(48, 48)


@pytest.fixture(scope="session")
def built_asset(dataset):
    return build_ds_asset(dataset, BuilderConfig())
