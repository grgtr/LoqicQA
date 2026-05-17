"""Stage 1: Describe normal images using VLM.

For each of the N few-shot normal images, call the VLM with a description
prompt and the image, and collect the textual descriptions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from PIL import Image

from logicqa.vlm.base import VLMBase
from logicqa.prompts import (
    DESCRIBE_PROMPT,
    IDENTIFY_COMPONENTS_PROMPT,
    DESCRIBE_COMPONENT_PROMPT,
    DESCRIBE_RELATIONAL_SLOT_PROMPT,
)
from logicqa.logging import PipelineLogger


# ============================================================
# Dataclasses for decomposed descriptions
# ============================================================

@dataclass
class ComponentObs:
    count: str = "N/A"
    position: str = "N/A"
    appearance: str = "N/A"
    rel_size: str = "N/A"

    def as_text(self, component: str) -> str:
        return (
            f"- {component}: count={self.count}, position={self.position}, "
            f"appearance={self.appearance}, relative size={self.rel_size}"
        )


@dataclass
class DecomposedDescription:
    image_path: str
    components: List[str] = field(default_factory=list)
    per_component: Dict[str, ComponentObs] = field(default_factory=dict)
    relational: str = "N/A"


# ============================================================
# Parsing helpers
# ============================================================

def _parse_bullet_list(text: str) -> List[str]:
    items = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            item = line[2:].strip().rstrip(".")
            if len(item) > 2:
                items.append(item)
    return items


def _parse_component_obs(text: str) -> ComponentObs:
    fields: Dict[str, str] = {"count": "N/A", "position": "N/A", "appearance": "N/A", "rel_size": "N/A"}
    # Map lowercase line prefixes → field key
    # Numbered form: "1. Count: exactly two"  or  "1. exactly two"
    # Labeled form:  "Count: exactly two"
    label_map = {
        "1.": "count", "count:": "count",
        "2.": "position", "position:": "position",
        "3.": "appearance", "appearance:": "appearance",
        "4.": "rel_size", "relative size:": "rel_size",
    }
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        for prefix, key in label_map.items():
            if lower.startswith(prefix):
                val = stripped[len(prefix):].strip().strip("*").strip()
                # Strip any trailing label like "Count: ..." left from numbered form
                if ":" in val:
                    colon_idx = val.index(":")
                    # Only strip if what's before the colon is a short label word (≤20 chars)
                    if colon_idx <= 20:
                        val = val[colon_idx + 1:].strip()
                if val:
                    fields[key] = val
                break
    return ComponentObs(**fields)


def _union_components(per_image: List[List[str]], normality_def: str) -> List[str]:
    """Deduplicated union of per-image component lists, anchored by normality_def.

    The anchor only uses short noun phrases (≤ 4 words) extracted from
    normality_def so that full sentences don't pollute the component list.
    The IDENTIFY_COMPONENTS_PROMPT already grounds the model to normality_def,
    so this anchor mainly catches edge cases where the model missed an item.
    """
    from logicqa.pipeline.stage2_summarize import _dedup_key
    seen: set = set()
    result: List[str] = []

    for comps in per_image:
        for c in comps:
            k = _dedup_key(c)
            if k and k not in seen:
                seen.add(k)
                result.append(c)

    # Anchor: extract short noun phrases (≤ 4 words) from normality_def lines
    for line in normality_def.splitlines():
        # Split on common delimiters to isolate candidate phrases
        for phrase in re.split(r"[,;.]|\band\b|\bthe\b|\bof\b|\bon\b|\bin\b|\bwith\b", line, flags=re.IGNORECASE):
            phrase = phrase.strip().lstrip("-").strip()
            # Strip quantity prefixes
            phrase = re.sub(r"^(exactly|at least|no more than|one|two|three|four|five|\d+)\s+", "", phrase, flags=re.IGNORECASE)
            phrase = phrase.strip()
            # Only keep short noun phrases (1–3 words, not spatial/relational terms)
            _SPATIAL_STOP = {"side", "ratio", "position", "left", "right", "center",
                             "top", "bottom", "fixed", "always", "contains", "located"}
            words = phrase.split()
            if 1 <= len(words) <= 3 and len(phrase) > 3:
                if not any(w.lower() in _SPATIAL_STOP for w in words):
                    k = _dedup_key(phrase)
                    if k and k not in seen:
                        seen.add(k)
                        result.append(phrase)

    return result



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


def describe_normal_images_decomposed(
    vlm: VLMBase,
    normal_images: List[Union[Path, Image.Image]],
    normality_definition: str,
    class_name: str = "object",
    image_paths: Optional[List[str]] = None,
    logger: Optional[PipelineLogger] = None,
) -> List[DecomposedDescription]:
    """Stage 1 (decomposed): per-component description of normal images.

    Phase A: identify components in each image (anchored to normality_definition).
    Phase B: compute deduplicated union across all images.
    Phase C: for each image, describe each component individually + relational facts.

    Returns List[DecomposedDescription] — one entry per normal image.
    """
    pil_images: List[Image.Image] = []
    for img in normal_images:
        if isinstance(img, (str, Path)):
            img = Image.open(str(img)).convert("RGB")
        pil_images.append(img)

    n = len(pil_images)
    print(f"  [Stage 1 Decomposed] Phase A: identifying components in {n} images ...")

    # Phase A: identify components per image
    raw_per_image: List[List[str]] = []
    for i, img in enumerate(pil_images):
        prompt = IDENTIFY_COMPONENTS_PROMPT.format(
            class_name=class_name,
            normality_definition=normality_definition,
        )
        response = vlm.query(prompt=prompt, image=img)
        comps = _parse_bullet_list(response.text)
        print(f"    image {i+1}: {comps}")
        raw_per_image.append(comps)
        if logger:
            img_path = str(image_paths[i]) if image_paths else f"image_{i}"
            logger.log_stage1_description(
                image_idx=f"{i+1}.identify",
                image_path=img_path,
                prompt=prompt,
                response_text=response.text,
            )

    # Phase B: union + normality_definition anchor
    all_components = _union_components(raw_per_image, normality_definition)
    all_components_bullet = "\n".join(f"- {c}" for c in all_components)
    print(f"  [Stage 1 Decomposed] Component union ({len(all_components)}): {all_components}")

    # Phase C: per-component describe + relational
    print(f"  [Stage 1 Decomposed] Phase C: per-component calls ({n} images × {len(all_components)+1} calls) ...")
    descriptions: List[DecomposedDescription] = []
    for i, img in enumerate(pil_images):
        img_path = str(image_paths[i]) if image_paths else f"image_{i}"
        per_comp: Dict[str, ComponentObs] = {}

        for c in all_components:
            prompt = DESCRIBE_COMPONENT_PROMPT.format(
                class_name=class_name,
                component=c,
                all_components_bullet=all_components_bullet,
            )
            resp = vlm.query(prompt=prompt, image=img)
            obs = _parse_component_obs(resp.text)
            per_comp[c] = obs
            if logger:
                logger.log_stage1_description(
                    image_idx=f"{i+1}.{c}",
                    image_path=img_path,
                    prompt=prompt,
                    response_text=resp.text,
                )

        rel_prompt = DESCRIBE_RELATIONAL_SLOT_PROMPT.format(
            class_name=class_name,
            all_components_bullet=all_components_bullet,
        )
        rel_resp = vlm.query(prompt=rel_prompt, image=img)
        relational = rel_resp.text.strip()
        if logger:
            logger.log_stage1_description(
                image_idx=f"{i+1}.relational",
                image_path=img_path,
                prompt=rel_prompt,
                response_text=relational,
            )

        descriptions.append(DecomposedDescription(
            image_path=img_path,
            components=raw_per_image[i],
            per_component=per_comp,
            relational=relational,
        ))
        print(f"    image {i+1} done: {list(per_comp.keys())}")

    total_calls = n + n * (len(all_components) + 1)
    print(f"  [Stage 1 Decomposed] Done: {total_calls} VLM calls for {n} images, {len(all_components)} components")
    return descriptions
