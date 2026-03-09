"""Stage 2: Summarize normal image descriptions into a normality context."""
from __future__ import annotations

from logicqa.vlm.base import VLMBase
from logicqa.prompts import SUMMARIZE_PROMPT, format_descriptions
from logicqa.logging import PipelineLogger

def summarize_normal_context(
    vlm: VLMBase,
    descriptions: list[str],
    normality_definition: str,
    class_name: str = "object",
    logger: Optional[PipelineLogger] = None,
) -> str:
    """
    Stage 2: Distill multiple normal image descriptions into a single summary.

    Args:
        vlm:                  VLM backend.
        descriptions:         List of descriptions from Stage 1.
        normality_definition: Normality definition string.

    Returns:
        A normality summary string.
    """
    print(" [Stage 2] Summarizing normal image context ...")
    labeled = format_descriptions(descriptions, class_name=class_name)
    prompt = SUMMARIZE_PROMPT.format(
        labeled_descriptions=labeled,
        n_descriptions=len(descriptions),
        normality_definition=normality_definition,
    )
    response = vlm.query(prompt=prompt, image=None)
    text = response.text.strip()

    if logger:
        logger.log_stage2_summary(prompt=prompt, response_text=text)
    return text
