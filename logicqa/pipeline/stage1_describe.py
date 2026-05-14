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



def describe_normal_images(
    vlm: VLMBase,
    normal_images: List[Union[Path, Image.Image]],
    normality_definition: str,
    class_name: str = "object",
    image_paths: Optional[List[str]] = None,
    logger: Optional[PipelineLogger] = None,
    llm_judge=None,
) -> List[str]:
    """Stage 1: Generate textual descriptions of normal images."""
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
        # Only flag descriptions with HIGH severity (4+ issues) or confirmed
        # impossible objects — Qwen2.5-3B produces many false positives on
        # count/spatial categories for normal images, so low/medium severity
        # is treated as noise and ignored to avoid corrupting Stage 2.
        if llm_judge is not None:
            try:
                report = llm_judge.detect_hallucinations(
                    description=text,
                    normality_definition=normality_definition,
                    class_name=class_name,
                    impossible_objects=impossible,
                )
                severity = report.severity()
                flag = (
                    report.is_hallucinated
                    and report.overall_confidence >= 0.8
                    and (severity == "high" or len(report.impossible_objects) > 0)
                )
                if flag:
                    print(
                        f"  [Hallucination] description {i+1} severity={severity}: "
                        f"impossible={report.impossible_objects}, "
                        f"counts={report.count_violations}, "
                        f"spatial={report.spatial_violations}, "
                        f"absent={report.absent_required}"
                    )
                    text = text + f"\n[UNCERTAIN: hallucination detected (severity={severity})]"
                elif report.is_hallucinated:
                    print(
                        f"  [Hallucination] description {i+1} severity={severity} "
                        f"(confidence={report.overall_confidence:.2f}) — below flag threshold, ignored"
                    )
            except Exception as e:
                print(f"  [Improvement 6] LLMJudge check failed: {e}")

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
