#!/usr/bin/env python3
"""
tests/test_preprocessing.py — Tests for BPM and normality definitions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from logicqa.preprocessing.bpm import apply_bpm
from logicqa.config import BPMConfig
from logicqa.data.normality_definitions import (
    get_normality_definition,
    list_classes,
    BPM_CLASSES,
    LANGSAM_CLASSES,
)


# --------------------------------------------------------------------------- #
# BPM tests
# --------------------------------------------------------------------------- #

def _make_image_with_border(size=64, border_color=200, center_color=50):
    """Create a synthetic image with a uniform border and a dark center."""
    arr = np.full((size, size, 3), border_color, dtype=np.uint8)
    margin = size // 4
    arr[margin: size - margin, margin: size - margin] = center_color
    return Image.fromarray(arr)


def test_bpm_returns_pil_image():
    img = _make_image_with_border()
    cfg = BPMConfig(enabled=True, patch_size=8, threshold=0.05)
    result = apply_bpm(img, cfg)
    assert isinstance(result, Image.Image)
    assert result.size == img.size


def test_bpm_whitens_border_patches():
    """Border patches should be white after BPM, center should be dark."""
    img = _make_image_with_border(size=64, border_color=200, center_color=50)
    cfg = BPMConfig(enabled=True, patch_size=8, threshold=0.25)
    result = apply_bpm(img, cfg)
    arr = np.array(result)

    # Top-left corner (border) should be whitened
    top_left_mean = arr[0:8, 0:8].mean()
    assert top_left_mean > 200, f"Border patch not whitened: mean={top_left_mean}"

    # Center should remain dark
    center_mean = arr[20:44, 20:44].mean()
    assert center_mean < 100, f"Center was incorrectly whitened: mean={center_mean}"


def test_bpm_disabled_returns_original():
    from logicqa.preprocessing.bpm import apply_bpm_from_config
    img = _make_image_with_border()
    cfg = BPMConfig(enabled=False)
    result = apply_bpm_from_config(img, cfg)
    assert np.array_equal(np.array(result), np.array(img))


# --------------------------------------------------------------------------- #
# Normality definitions tests
# --------------------------------------------------------------------------- #

def test_all_classes_have_definition():
    for cls in list_classes():
        defn = get_normality_definition(cls)
        assert isinstance(defn, str) and len(defn) > 10, f"Missing definition for {cls}"


def test_unknown_class_raises():
    with pytest.raises(KeyError):
        get_normality_definition("unknown_product")


def test_bpm_classes_are_valid():
    all_cls = set(list_classes())
    assert BPM_CLASSES.issubset(all_cls), f"BPM_CLASSES has unknown classes: {BPM_CLASSES - all_cls}"


def test_langsam_classes_are_valid():
    all_cls = set(list_classes())
    assert LANGSAM_CLASSES.issubset(all_cls), \
        f"LANGSAM_CLASSES has unknown classes: {LANGSAM_CLASSES - all_cls}"
