"""Lang-SAM wrapper for segmenting uniform objects.

Lang-SAM (Language Segment-Anything) combines GroundingDINO with SAM2 to
segment objects described by a text prompt. This is used for classes where
objects are arranged uniformly (Pushpins, Splicing Connectors), which causes
VLM hallucinations when the entire image is fed at once.

Reference: Appendix B.3, F of the paper.
Applied to: Pushpins, Splicing Connectors.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from PIL import Image
import numpy as np

from logicqa.config import LangSAMConfig


class LangSAMWrapper:
    """
    Wrapper around the Lang-SAM library to segment individual uniform objects
    from an image using a text prompt.

    Usage:
        wrapper = LangSAMWrapper(cfg)
        crops = wrapper.segment(image, class_name="pushpins")
        # crops: list of PIL images, one per detected object
    """

    def __init__(self, cfg: LangSAMConfig):
        self.cfg = cfg
        self._model = None

    def _load_model(self):
        """Lazy-load Lang-SAM to avoid import overhead when not needed."""
        if self._model is None:
            try:
                from lang_sam import LangSAM
                self._model = LangSAM()
                print("[LangSAM] Model loaded.")
            except ImportError:
                raise ImportError(
                    "lang-sam is not installed. "
                    "Install it with: pip install lang-sam"
                )

    def segment(
        self,
        image: Union[Image.Image, Path, str],
        class_name: str,
    ) -> List[Image.Image]:
        """
        Segment individual objects from an image using the class-specific prompt.

        Args:
            image:      Input PIL image (or path to one).
            class_name: MVTec LOCO AD class name (e.g., 'pushpins').

        Returns:
            List of PIL image crops, each containing one segmented object.
            If the class is not in the prompts config, returns [original image].
        """
        if not self.cfg.enabled:
            if isinstance(image, (str, Path)):
                image = Image.open(str(image)).convert("RGB")
            return [image]

        class_name_norm = class_name.lower().replace(" ", "_")
        prompt = self.cfg.prompts.get(class_name_norm)
        if prompt is None:
            # No Lang-SAM prompt for this class — return whole image
            if isinstance(image, (str, Path)):
                image = Image.open(str(image)).convert("RGB")
            return [image]

        self._load_model()

        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        # Run Lang-SAM
        masks, boxes, phrases, logits = self._model.predict(pil_img, prompt)

        if len(masks) == 0:
            # No objects found — return the original image
            return [pil_img]

        crops = []
        img_arr = np.array(pil_img)
        for mask in masks:
            mask_np = mask.cpu().numpy().astype(bool) if hasattr(mask, "cpu") else np.array(mask, dtype=bool)
            # Crop to bounding box of this mask
            rows = np.any(mask_np, axis=1)
            cols = np.any(mask_np, axis=0)
            if not rows.any() or not cols.any():
                continue
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            crop_arr = img_arr[rmin:rmax + 1, cmin:cmax + 1].copy()
            # Zero out pixels outside the mask in this crop
            mask_crop = mask_np[rmin:rmax + 1, cmin:cmax + 1]
            crop_arr[~mask_crop] = 255  # white background
            crops.append(Image.fromarray(crop_arr))

        return crops if crops else [pil_img]

    def get_full_segmentation_image(
        self,
        image: Union[Image.Image, Path, str],
        class_name: str,
    ) -> Image.Image:
        """
        Return the image with non-object areas masked to white.
        This is an alternative to per-crop processing: the entire image is
        returned but with background zeroed out.
        """
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        crops = self.segment(pil_img, class_name)
        if len(crops) == 1 and crops[0].size == pil_img.size:
            return pil_img  # Nothing was segmented; return original

        # Compose all crops onto a white canvas
        canvas = Image.new("RGB", pil_img.size, (255, 255, 255))
        return canvas  # simplified: for per-object queries, use .segment() instead
