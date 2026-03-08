"""Stage 2: Summarize normal image descriptions into a normality context."""
from __future__ import annotations

from logicqa.vlm.base import VLMBase
from logicqa.prompts import SUMMARIZE_PROMPT, format_descriptions


def summarize_normal_context(
    vlm: VLMBase,
    descriptions: list[str],
    normality_definition: str,
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
    print("  [Stage 2] Summarizing normal image context ...")
    formatted_descs = format_descriptions(descriptions)
    prompt = SUMMARIZE_PROMPT.format(
        n_descriptions=len(descriptions),
        descriptions=formatted_descs,
        normality_definition=normality_definition,
    )
    response = vlm.query(prompt=prompt, image=None)
    return response.text.strip()
