"""Stage 2: Summarize normal image descriptions into a normality context."""
from __future__ import annotations

from logicqa.vlm.base import VLMBase
from logicqa.prompts import SUMMARIZE_PROMPT, format_descriptions
from logicqa.logging import PipelineLogger

from typing import Dict, List, Optional, Union
import re


_HEDGE_PATTERNS = [
    r"if applicable",
    r"depending on",
    r"unless (stated|specified|otherwise)",
    r"no specific .{0,30} (given|provided|mentioned)",
    r"(approximately|about) \d+%",
    r"(may|might|could) be",
    r"seems? to",
    r"beyond fixed counts",
]

def _sanitize_summary_section(text: str) -> str:
    """
    Replace a summary section with N/A if it contains hedge language
    that indicates the model was uncertain rather than factual.
    """
    for pattern in _HEDGE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return "N/A"
    return text

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
        class_name=class_name,
        
    )
    response = vlm.query(prompt=prompt, image=None) # Try add images to promt
    text = response.text.strip()
    sanitized = _sanitize_summary_section(text)
    if logger:
        logger.log_stage2_summary(prompt=prompt, response_text=sanitized)
    
    return sanitized

