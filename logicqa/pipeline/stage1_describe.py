"""Stage 1: Describe normal images using VLM.

For each of the N few-shot normal images, call the VLM with a description
prompt and the image, and collect the textual descriptions.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Union

from PIL import Image

from logicqa.vlm.base import VLMBase
from logicqa.prompts import DESCRIBE_PROMPT


def describe_normal_images(
    vlm: VLMBase,
    normal_images: List[Union[Path, Image.Image]],
    normality_definition: str,
    class_name: str = "object",
) -> List[str]:
    """
    Stage 1: Generate textual descriptions of normal images.

    Args:
        vlm:                  VLM backend instance.
        normal_images:        List of N normal image paths or PIL Images.
        normality_definition: Normality definition string from Appendix C.2.

    Returns:
        List of N description strings, one per image.
    """
    prompt = DESCRIBE_PROMPT.format(
        class_name=class_name,
        normality_definition=normality_definition,
    )

    descriptions = []
    for i, img in enumerate(normal_images):
        print(f"  [Stage 1] Describing normal image of class {class_name} {i + 1}/{len(normal_images)} ...")
        if isinstance(img, (str, Path)):
            img = Image.open(str(img)).convert("RGB")
        response = vlm.query(prompt=prompt, image=img)
        descriptions.append(response.text.strip())
    return descriptions
