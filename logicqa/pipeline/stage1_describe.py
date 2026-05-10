"""Stage 1: Describe normal images using VLM.

For each of the N few-shot normal images, call the VLM with a description
prompt and the image, and collect the textual descriptions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

from PIL import Image

from logicqa.vlm.base import VLMBase
from logicqa.prompts import DESCRIBE_PROMPT
from logicqa.logging import PipelineLogger


def _flag_hallucinations(description: str, impossible_objects: List[str]) -> bool:
    """
    Improvement 6: Lightweight keyword check for impossible objects in a description.

    Returns True if the description likely contains hallucinated content
    (mentions objects that cannot appear in this class).
    """
    desc_lower = description.lower()
    for obj in impossible_objects:
        if obj.lower() in desc_lower:
            return True
    return False


def describe_normal_images(
    vlm: VLMBase,
    normal_images: List[Union[Path, Image.Image]],
    normality_definition: str,
    class_name: str = "object",
    image_paths: Optional[List[str]] = None,
    logger: Optional[PipelineLogger] = None,
    llm_judge=None,
) -> List[str]:
    """
    Stage 1: Generate textual descriptions of normal images.

    Args:
        vlm:                  VLM backend instance.
        normal_images:        List of N normal image paths or PIL Images.
        normality_definition: Normality definition string from Appendix C.2.
        llm_judge:            Optional LLMJudge instance for hallucination detection
                              (Improvement 6). If provided, descriptions containing
                              impossible objects are flagged with a warning tag so
                              Stage 2 sanitization can suppress them.

    Returns:
        List of N description strings, one per image.
    """
    from logicqa.data.normality_definitions import IMPOSSIBLE_OBJECTS

    prompt = DESCRIBE_PROMPT.format(
        class_name=class_name,
        normality_definition=normality_definition,
    )
    impossible = IMPOSSIBLE_OBJECTS.get(class_name.lower().replace(" ", "_"), [])

    descriptions = []
    for i, img in enumerate(normal_images):
        print(f"  [Stage 1] Describing normal image of class {class_name} {i + 1}/{len(normal_images)} ...")
        if isinstance(img, (str, Path)):
            img = Image.open(str(img)).convert("RGB")
        response = vlm.query(prompt=prompt, image=img)
        text = response.text.strip()

        # Improvement 6: hallucination detection
        if llm_judge is not None:
            try:
                report = llm_judge.detect_hallucinations(
                    description=text,
                    normality_definition=normality_definition,
                    class_name=class_name,
                    impossible_objects=impossible,
                )
                if report.is_hallucinated:
                    severity = report.severity()
                    print(
                        f"  [Hallucination] description {i+1} severity={severity}: "
                        f"impossible={report.impossible_objects}, "
                        f"counts={report.count_violations}, "
                        f"spatial={report.spatial_violations}, "
                        f"absent={report.absent_required}"
                    )
                    text = text + f"\n[UNCERTAIN: hallucination detected (severity={severity})]"
            except Exception as e:
                print(f"  [Improvement 6] LLMJudge check failed: {e}")
                # Keyword fallback on LLMJudge failure
                if impossible and _flag_hallucinations(text, impossible):
                    text = text + "\n[UNCERTAIN: possible hallucination detected]"
        elif impossible:
            # Lightweight keyword fallback (no LLMJudge needed)
            if _flag_hallucinations(text, impossible):
                print(f"  [Improvement 6] Keyword hallucination flag in description {i+1}")
                text = text + "\n[UNCERTAIN: possible hallucination detected]"

        descriptions.append(text)
        if logger:
            img_path = str(image_paths[i]) if image_paths else f"image_{i}"
            logger.log_stage1_description(
                image_idx=i + 1,
                image_path=img_path,
                prompt=prompt,
                response_text=text,
            )
    return descriptions
