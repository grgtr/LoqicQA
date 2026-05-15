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

_QUANTITY_PREFIX_RE = re.compile(
    r"^(exactly\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+",
    re.IGNORECASE,
)


def _dedup_key(raw: str) -> str:
    """
    Canonical form of a component name used only for deduplication.

    Steps:
      1. Strip leading quantity words ("Two tangerines" → "tangerines")
      2. Strip parenthetical qualifiers ("Cereal mixture (grains and nuts)" → "cereal mixture")
      3. Strip "with ..." suffixes ("Cereal mixture with almonds" → "cereal mixture")
      4. Strip trailing plural 's' for a language-neutral key ("tangerines" → "tangerine")
      5. Lowercase and strip whitespace
    """
    s = _QUANTITY_PREFIX_RE.sub("", raw)
    s = re.sub(r"\s*\(.*?\)", "", s)
    s = re.split(r"\s+with\s+", s, maxsplit=1)[0]
    s = s.strip().lower()
    # simple singularisation: "tangerines"→"tangerine", "chips"→"chip"
    if s.endswith("ies") and len(s) > 4:
        s = s[:-3] + "y"
    elif s.endswith("s") and not s.endswith("ss") and len(s) > 4:
        s = s[:-1]
    return s


def _expand_raw_component(raw: str) -> List[str]:
    """
    Expand one raw bullet string into individual component display names.

    "Cereal mixture with banana chips and almonds"
        → ["Cereal mixture", "banana chips", "almonds"]
    "Two tangerines (one above the other)"
        → ["tangerines"]          (parenthetical stripped)
    "Banana chips"
        → ["Banana chips"]
    """
    # Base: strip quantity prefix and parenthetical
    base = _QUANTITY_PREFIX_RE.sub("", raw)
    base = re.sub(r"\s*\(.*?\)", "", base)

    # Split on "with" to get base and extras
    parts = re.split(r"\s+with\s+", base, maxsplit=1, flags=re.IGNORECASE)
    items = [parts[0].strip()]

    if len(parts) == 2:
        # "banana chips and almonds" → ["banana chips", "almonds"]
        extras = re.split(r",\s*|\s+and\s+", parts[1], flags=re.IGNORECASE)
        items.extend(e.strip().rstrip(".") for e in extras if e.strip())

    return [i for i in items if len(i) > 2]


def extract_all_components(descriptions: List[str]) -> List[str]:
    """
    Parse '1. Components:' from every Stage 1 description and return a
    deduplicated union of all mentioned objects.

    Handles:
    - Quantity prefixes: "Two tangerines" and "Tangerine" → same component
    - Parenthetical qualifiers: "Cereal mixture (grains and nuts)" → "Cereal mixture"
    - Compound entries: "Cereal mixture with banana chips and almonds"
                        → ["Cereal mixture", "banana chips", "almonds"]
    """
    seen_keys: set = set()
    components: List[str] = []

    for desc in descriptions:
        in_components = False
        for line in desc.splitlines():
            line = line.strip()
            if re.match(r"^1\.\s*(components|Components)", line):
                in_components = True
                continue
            if re.match(r"^\d+\.\s", line) and in_components:
                break
            if in_components and line.startswith("- "):
                raw = line[2:].strip().rstrip(".")
                if len(raw) < 3:
                    continue
                for item in _expand_raw_component(raw):
                    key = _dedup_key(item)
                    if key and key not in seen_keys:
                        seen_keys.add(key)
                        components.append(item)

    return components


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

