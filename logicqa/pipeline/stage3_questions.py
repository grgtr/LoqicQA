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
    build_question_slots,
    build_subquestion_slots
)
from logicqa.logging import PipelineLogger
from logicqa.data.normality_definitions import CLASS_INSPECTION_CONTEXTS


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

def _parse_questions(text: str) -> List[str]:
    lines = text.strip().splitlines()
    questions = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(r"^\(Q\d+\)\s*:?\s*(.+)$", line, re.IGNORECASE)
        if match:
            questions.append(match.group(1).strip())
            continue

        match = re.match(r"^Q\d+[\.:\)]\s*(.+)$", line, re.IGNORECASE)
        if match:
            questions.append(match.group(1).strip())
            continue

        match = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if match:
            q = match.group(1).strip()
            if len(q) > 10:
                questions.append(q)
            continue

        match = re.match(r"^[-•]\s*(.+)$", line)
        if match:
            q = match.group(1).strip()
            if len(q) > 10:
                questions.append(q)
            continue

    if not questions:
        for line in text.strip().splitlines():
            line = line.strip()
            if len(line) > 15 and (
                line.endswith("?") or
                re.match(r"^(Is |Are |Does |Do |Can |Has |Have )", line, re.IGNORECASE)
            ):
                questions.append(line)

    questions = [
        q for q in questions
        if (q.endswith("?") or
        re.match(r"^(Is |Are |Does |Do |Can |Has |Have |Did )", q, re.IGNORECASE)) and 15 < len(q) < 250
    ]
    return questions


def _parse_output_list(text: str) -> List[str]:
    lines = text.strip().splitlines()
    variants = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(r"^Output\s*\d+\s*:\s*(.+)$", line, re.IGNORECASE)
        if match:
            variants.append(match.group(1).strip())
            continue
        match = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if match:
            q = match.group(1).strip()
            if len(q) > 10:
                variants.append(q)
            continue

        match = re.match(r"^[-•]\s*(.+)$", line)
        if match:
            q = match.group(1).strip()
            if len(q) > 10:
                variants.append(q)

    if not variants:
        for line in text.strip().splitlines():
            line = line.strip()
            if len(line) > 15 and (
                line.endswith("?") or
                re.match(r"^(Is |Are |Does |Do |Can |Has |Have )", line, re.IGNORECASE)
            ):
                variants.append(line)
    variants = [
        q for q in variants
        if (q.endswith("?") or
        re.match(r"^(Is |Are |Does |Do |Can |Has |Have |Did )", q, re.IGNORECASE)) and 15 < len(q) < 250
    ]
    return variants


def generate_candidate_questions(
    vlm: VLMBase,
    normality_summary: str,
    normality_definition: str,
    class_name: str = "object",
    n_questions: int = 6,
    logger: Optional[PipelineLogger] = None,
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
        class_name=class_name,
        normality_summary=normality_summary,
        normality_definition=normality_definition,
        n_questions=n_questions,
        question_slots=build_question_slots(n_questions),
    )
    response = vlm.query(prompt=prompt, image=None)

    print(f"   [DEBUG] Raw output:\n{response.text}\n")
    questions = _parse_questions(response.text)
    print(f"   [DEBUG] Parsed questions:\n{questions}\n")
    print(f"    Generated {len(questions)} candidate questions.")
    if logger:
        logger.log_stage3a_questions(
            prompt=prompt,
            response_text=response.text,
            parsed_questions=questions,
        )
    return questions


def _answer_single_question(
    vlm: VLMBase,
    question: str,
    image: Union[Path, Image.Image],
    class_name: str = "object",
    logger: Optional[PipelineLogger] = None,
    gt_label: str = "unknown",
    image_path: Optional[str] = None,
) -> Optional[str]:
    """Ask a single question about one image and return 'Yes'/'No'/None."""
    if isinstance(image, (str, Path)):
        img = Image.open(str(image)).convert("RGB")
    else:
        img = image

    normalized_class_name = class_name.lower().replace(" ", "_")
    class_context = CLASS_INSPECTION_CONTEXTS.get(normalized_class_name, "")
    prompt = TEST_PROMPT.format(question=question, class_name=class_name, class_context=class_context)
    response = vlm.query(prompt=prompt, image=img)
    if logger:
        logger.log_stage3b_filter_answer(
            question=question,
            image_path=image_path,
            gt_label=gt_label,
            prompt=prompt,
            response_text=response.text,
            extracted_answer=response.answer,
        )
    return response.answer


def filter_questions_on_normal(
    vlm: VLMBase,
    candidate_questions: List[str],
    normal_images: List[Union[Path, Image.Image]],
    threshold: float = 0.8,
    class_name: str = "object",
    image_paths: Optional[List[str]] = None,
    logger: Optional[PipelineLogger] = None,
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
    print(f"[DEBUG] stage3b Filtering logger is: {'None' if logger is None else 'not None'}")
    if not normal_images:
        return candidate_questions

    print(f"  [Stage 3b] Filtering {len(candidate_questions)} questions "
          f"on {len(normal_images)} normal images (threshold={threshold:.0%}) ...")

    kept = []
    for q in candidate_questions:
        correct = 0
        for i, img in enumerate(normal_images):
            gt_label = "good"
            answer = _answer_single_question(vlm, q, img, class_name, logger, gt_label, str(image_paths[i]))
            if answer == "Yes":
                correct += 1
        accuracy = correct / len(normal_images)
        status = "KEEP" if accuracy >= threshold else "DROP"
        print(f"    [{status}] acc={accuracy:.2f} | {q}")
        if logger:
            logger.log_stage3b_result(q, accuracy, accuracy >= threshold)
        if accuracy >= threshold:
            kept.append(q)

    print(f"  [Stage 3b] Kept {len(kept)}/{len(candidate_questions)} questions.")
    return kept


def generate_sub_questions(
    vlm: VLMBase,
    main_questions: List[str],
    n_variants: int = 5,
    logger: Optional[PipelineLogger] = None,
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
    subquestion_slots = build_subquestion_slots(n_variants)
    for i, mq in enumerate(main_questions):
        prompt = SUBQUESTION_AUGMENT_PROMPT.format(n_variants=n_variants,
            main_question=mq,
            subquestion_slots=subquestion_slots,
        )
        response = vlm.query(prompt=prompt, image=None)
        print("[DEBUG] generated sub_questions :", response.text)
        variants = _parse_output_list(response.text)
        print(variants)
        # Ensure we always have exactly n_variants (pad with original if short)
        while len(variants) < n_variants:
            variants.append(mq)
        sub_questions[mq] = variants[:n_variants]
        print(f"    Q{i+1}: {mq[:60]} → {len(variants)} sub-Qs")
        if logger:
            logger.log_stage3c_subquestions(
                main_question=mq,
                prompt=prompt,
                response_text=response.text,
                sub_questions=sub_questions[mq],
            )
    return sub_questions
