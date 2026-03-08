"""Stage 3: Generate and filter main questions.

This stage:
1. Prompts the VLM to generate candidate main questions from the normality summary.
2. Filters out questions with < 80% accuracy on validation normal images.
3. Generates 5 semantically equivalent sub-questions per accepted main question.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from PIL import Image

from logicqa.vlm.base import VLMBase
from logicqa.prompts import (
    GENERATE_QUESTIONS_PROMPT,
    SUBQUESTION_AUGMENT_PROMPT,
    TEST_PROMPT,
)


def _parse_numbered_list(text: str) -> List[str]:
    """Parse a numbered list (1. ... \n 2. ...) from VLM output."""
    lines = text.strip().splitlines()
    questions = []
    for line in lines:
        # Match "1. ", "1) ", or bare lines
        match = re.match(r"^\s*\d+[\.\)]\s*(.+)$", line)
        if match:
            questions.append(match.group(1).strip())
        elif line.strip() and not line.strip().isdigit():
            # Fallback: non-empty line without a number prefix
            # pass
            questions.append(line.strip())
    return questions


def generate_candidate_questions(
    vlm: VLMBase,
    normality_summary: str,
    normality_definition: str,
    n_questions: int = 6,
) -> List[str]:
    """
    Stage 3a: Generate candidate main questions from normality summary.

    Args:
        vlm:                  VLM backend.
        normality_summary:    Output of Stage 2.
        normality_definition: Formal normality definition.
        n_questions:          Number of questions to request.

    Returns:
        List of candidate question strings.
    """
    print("  [Stage 3a] Generating candidate main questions ...")
    prompt = GENERATE_QUESTIONS_PROMPT.format(
        normality_summary=normality_summary,
        normality_definition=normality_definition,
        n_questions=n_questions,
    )
    response = vlm.query(prompt=prompt, image=None)
    questions = _parse_numbered_list(response.text)
    print(f"    Generated {len(questions)} candidate questions.")
    return questions


def _answer_single_question(
    vlm: VLMBase,
    question: str,
    image: Union[Path, Image.Image],
) -> Optional[str]:
    """Ask a single question about one image and return 'Yes'/'No'/None."""
    if isinstance(image, (str, Path)):
        img = Image.open(str(image)).convert("RGB")
    else:
        img = image

    prompt = TEST_PROMPT.format(question=question)
    response = vlm.query(prompt=prompt, image=img)
    return response.answer


def filter_questions_on_normal(
    vlm: VLMBase,
    candidate_questions: List[str],
    normal_images: List[Union[Path, Image.Image]],
    threshold: float = 0.8,
) -> List[str]:
    """
    Stage 3b: Filter candidate questions with < threshold accuracy on normals.

    Normal images should answer 'Yes' to every valid question.
    Questions where accuracy < threshold are dropped (they are biased toward
    the few-shot samples or too noisy).

    Args:
        vlm:                 VLM backend.
        candidate_questions: From Stage 3a.
        normal_images:       Validation normal images (can be the same 3 few-shot).
        threshold:           Minimum accuracy to keep a question (default 0.8).

    Returns:
        Filtered list of main questions.
    """
    if not normal_images:
        return candidate_questions

    print(f"  [Stage 3b] Filtering {len(candidate_questions)} questions "
          f"on {len(normal_images)} normal images (threshold={threshold:.0%}) ...")

    kept = []
    for q in candidate_questions:
        correct = 0
        for img in normal_images:
            answer = _answer_single_question(vlm, q, img)
            if answer == "Yes":
                correct += 1
        accuracy = correct / len(normal_images)
        status = "✓ KEEP" if accuracy >= threshold else "✗ DROP"
        print(f"    [{status}] acc={accuracy:.2f} | {q[:80]}")
        if accuracy >= threshold:
            kept.append(q)

    print(f"  [Stage 3b] Kept {len(kept)}/{len(candidate_questions)} questions.")
    return kept


def generate_sub_questions(
    vlm: VLMBase,
    main_questions: List[str],
    n_variants: int = 5,
) -> Dict[str, List[str]]:
    """
    Stage 3c: Generate sub-question variants for each accepted main question.

    Args:
        vlm:            VLM backend.
        main_questions: Filtered main questions from Stage 3b.
        n_variants:     Number of sub-question variants per main question.

    Returns:
        Dict mapping each main question → list of n_variants sub-questions.
    """
    print(f"  [Stage 3c] Generating {n_variants} sub-questions per main question ...")
    sub_questions: Dict[str, List[str]] = {}
    for i, mq in enumerate(main_questions):
        prompt = SUBQUESTION_AUGMENT_PROMPT.format(
            main_question=mq,
            n_variants=n_variants,
        )
        response = vlm.query(prompt=prompt, image=None)
        variants = _parse_numbered_list(response.text)
        # Ensure we always have exactly n_variants (pad with original if short)
        while len(variants) < n_variants:
            variants.append(mq)
        sub_questions[mq] = variants[:n_variants]
        print(f"    Q{i+1}: {mq[:60]} → {len(variants)} sub-Qs")
    return sub_questions
