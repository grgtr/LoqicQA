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
    r"\[UNCERTAIN",
]

# Fix B: pattern to strip [UNCERTAIN: ...] tags added by Stage 1 hallucination detector
_UNCERTAIN_TAG_RE = re.compile(r"\n?\[UNCERTAIN:[^\]]*\]", re.IGNORECASE)


def _strip_uncertain_tags(description: str) -> str:
    """Strip [UNCERTAIN: ...] suffixes from a Stage 1 description.

    Preserves the useful description text while removing the warning tag so
    Stage 2 receives clean context instead of the raw annotation.
    """
    return _UNCERTAIN_TAG_RE.sub("", description).strip()


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
    # Fix B: strip [UNCERTAIN: ...] tags before building the prompt so that
    # hallucination warnings from Stage 1 don't corrupt the Stage 2 context.
    clean_descriptions = [_strip_uncertain_tags(d) for d in descriptions]
    labeled = format_descriptions(clean_descriptions, class_name=class_name)
    prompt = SUMMARIZE_PROMPT.format(
        labeled_descriptions=labeled,
        n_descriptions=len(descriptions),
        normality_definition=normality_definition,
        class_name=class_name,
        
    )
    response = vlm.query(prompt=prompt, image=None) # Try add images to promt
    text = response.text.strip()
    sanitized = text
    if logger:
        logger.log_stage2_summary(prompt=prompt, response_text=sanitized)
    
    return sanitized

