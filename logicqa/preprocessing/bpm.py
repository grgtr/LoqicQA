"""Back Patch Masking (BPM) preprocessing.

BPM isolates the foreground object from the background by masking out
background patches — producing a target-centric image. This helps the VLM
focus on the relevant object and reduces hallucinations caused by large
irrelevant background areas.

Reference: Lee et al. (2023). Applied to: Screw Bag, Splicing Connectors.

Algorithm:
    1. Divide the image into non-overlapping patches of size `patch_size` x `patch_size`.
    2. Compute the average colour of each patch.
    3. Estimate the background colour as the most common patch colour (mode of
       the colour histogram, or median of the border patches).
    4. Mark patches whose average colour is within `threshold` (L1 distance
       normalised to [0, 1]) of the background colour as background.
    5. Replace background patches with white pixels (or a neutral fill).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from logicqa.config import BPMConfig


def apply_bpm(
    image: Image.Image,
    cfg: BPMConfig,
) -> Image.Image:
    """
    Apply Back Patch Masking to isolate the foreground object.

    Args:
        image: Input PIL image (RGB).
        cfg:   BPMConfig with patch_size and threshold.

    Returns:
        Masked PIL image with background patches replaced by white.
    """
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0  # H x W x 3
    h, w = arr.shape[:2]
    ps = cfg.patch_size

    # ------------------------------------------------------------------ #
    # Step 1: Estimate background colour from border patches
    # ------------------------------------------------------------------ #
    border_colours = []
    for r in range(0, h, ps):
        for c in range(0, w, ps):
            is_border = (r == 0 or r + ps >= h or c == 0 or c + ps >= w)
            if is_border:
                patch = arr[r: r + ps, c: c + ps]
                if patch.size > 0:
                    border_colours.append(patch.mean(axis=(0, 1)))

    if len(border_colours) == 0:
        return img  # Cannot determine background; return original

    bg_colour = np.median(border_colours, axis=0)  # (3,)

    # ------------------------------------------------------------------ #
    # Step 2: Mask patches close to background colour
    # ------------------------------------------------------------------ #
    result = arr.copy()
    for r in range(0, h, ps):
        for c in range(0, w, ps):
            patch = arr[r: r + ps, c: c + ps]
            if patch.size == 0:
                continue
            patch_mean = patch.mean(axis=(0, 1))
            dist = np.abs(patch_mean - bg_colour).mean()  # L1 in [0, 1]
            if dist < cfg.threshold:
                result[r: r + ps, c: c + ps] = 1.0  # white fill

    return Image.fromarray((result * 255).clip(0, 255).astype(np.uint8))


def apply_bpm_from_config(image: Image.Image, cfg: BPMConfig) -> Image.Image:
    """Wrapper: only applies BPM if cfg.enabled is True."""
    if not cfg.enabled:
        return image
    return apply_bpm(image, cfg)
