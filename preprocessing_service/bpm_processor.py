"""Self-contained Back Patch Masking (BPM) — no logicqa dependency."""
import numpy as np
from PIL import Image


def process_image_bpm(image: Image.Image, patch_size: int = 16, threshold: float = 0.05) -> Image.Image:
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]

    border_colours = []
    for r in range(0, h, patch_size):
        for c in range(0, w, patch_size):
            if r == 0 or r + patch_size >= h or c == 0 or c + patch_size >= w:
                patch = arr[r:r + patch_size, c:c + patch_size]
                if patch.size > 0:
                    border_colours.append(patch.mean(axis=(0, 1)))

    if not border_colours:
        return img

    bg = np.median(border_colours, axis=0)
    result = arr.copy()
    for r in range(0, h, patch_size):
        for c in range(0, w, patch_size):
            patch = arr[r:r + patch_size, c:c + patch_size]
            if patch.size > 0 and np.abs(patch.mean(axis=(0, 1)) - bg).mean() < threshold:
                result[r:r + patch_size, c:c + patch_size] = 1.0

    return Image.fromarray((result * 255).clip(0, 255).astype(np.uint8))
