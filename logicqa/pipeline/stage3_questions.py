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
    BULLET_TO_QUESTION_PROMPT,
    SUBQUESTION_AUGMENT_PROMPT,
    TEST_PROMPT,
    build_question_slots,
    build_subquestion_slots
)
from logicqa.logging import PipelineLogger


def _normalize_text(text: str) -> str:
    """
    Pre-process VLM output before parsing:
    - Replace newlines with spaces
    - Remove/replace special characters that break regex matching
    - Strip surrounding quotes from individual items
    """
    # Replace newlines and carriage returns with a single space
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def _strip_quotes(s: str) -> str:
    """Strip surrounding straight or curly quotes from a string."""
    # Remove surrounding double/single/curly quotes (including nested)
    s = s.strip()
    # Iteratively strip outer quote pairs
    quote_pairs = [('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019")]
    changed = True
    while changed:
        changed = False
        for open_q, close_q in quote_pairs:
            if s.startswith(open_q) and s.endswith(close_q) and len(s) > 2:
                s = s[1:-1].strip()
                changed = True
            # Also handle mismatched: starts with " ends with ?"
            elif s.startswith(open_q) and s.endswith(close_q[0] if len(close_q) > 0 else "") :
                pass
    # Handle case: ends with `?"` or `?'` — quote after question mark
    s = re.sub(r'\?["\'\u201d\u2019]+$', "?", s)
    # Handle case: starts with quote char
    s = re.sub(r'^["\'\u201c\u2018]+', "", s)
    return s.strip()

def _is_valid_question(q: str) -> bool:
    """Check if a string looks like a valid Yes/No question."""
    q = q.strip()
    return (
        15 < len(q) < 300
        and (
            q.endswith("?")
            or re.match(
                r"^(Is |Are |Does |Do |Can |Has |Have |Did |Was |Were )",
                q,
                re.IGNORECASE,
            )
        )
    )


def _is_semantically_valid_question(q: str) -> bool:
    """
    Improvement 1: Programmatic semantic validation of generated questions.

    Rejects questions that are structurally malformed for binary Yes/No anomaly detection:
    - Negated constructions: the answer "Yes" would mean anomaly (inverted polarity)
    - Double-constraint questions via "and"/"or": ambiguous vote if one holds but not the other
    - Missing question mark: often indicates the model output a statement, not a question
    """
    q = q.strip()

    # Must end with "?"
    if not q.endswith("?"):
        return False

    q_lower = q.lower()

    # Reject negated question starters — "Yes" would mean anomaly (wrong polarity)
    negation_starters = (
        "isn't ", "aren't ", "doesn't ", "don't ", "can't ", "won't ",
        "is there no ", "are there no ", "is there not ", "are there not ",
    )
    if any(q_lower.startswith(neg) for neg in negation_starters):
        return False

    # Reject questions that couple two independent logical conditions with " and " or " or "
    # Heuristic: if the question contains " and " AND has two verb-phrase anchors after it,
    # it's likely a double-constraint. A simple count of "and" occurrences is sufficient.
    if q_lower.count(" and ") >= 2:
        return False

    return True

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
    """
    Parse a numbered list of sub-question variants from VLM output.

    Handles all observed VLM response formats:
      - Output1: "question text?"
      - Output1:\n"question text?"
      - 1. question text?
      - 1) question text?
      - - question text?
      - Plain question lines (fallback)

    Special characters are normalised before parsing:
      - \\n  → space
      - Surrounding quotes stripped from each candidate
    """
    variants: List[str] = []

    # ── Step 1: split on OutputN: markers (works even if question is on same line
    #            or on the next line after the marker)
    # First, try to find "OutputN:" anchored tokens to split the response
    output_marker_pattern = re.compile(
        r"Output\s*\d+\s*:", re.IGNORECASE
    )

    # Split text by "OutputN:" markers — handles multi-line values too
    parts = output_marker_pattern.split(text)
    if len(parts) > 1:
        # parts[0] is text before the first marker (usually empty or preamble)
        for part in parts[1:]:
            # Each part is everything between two consecutive "OutputN:" markers
            # Normalize: replace newlines with spaces
            part_clean = _normalize_text(part)
            # Strip surrounding quotes
            part_clean = _strip_quotes(part_clean)
            if part_clean:
                variants.append(part_clean)

    # ── Step 2: if OutputN: split found nothing useful, try line-by-line parsing
    if not variants:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # "1. text" or "1) text"
            m = re.match(r"^\d+[\.\)]\s*(.+)$", line)
            if m:
                q = _strip_quotes(m.group(1).strip())
                if q:
                    variants.append(q)
                continue

            # "- text" or "• text"
            m = re.match(r"^[-•]\s*(.+)$", line)
            if m:
                q = _strip_quotes(m.group(1).strip())
                if q:
                    variants.append(q)
                continue

    # ── Step 3: final fallback — collect any line that looks like a question
    if not variants:
        for raw_line in text.splitlines():
            line = _strip_quotes(raw_line.strip())
            if _is_valid_question(line):
                variants.append(line)

    # ── Step 4: validate and clean every collected candidate
    cleaned: List[str] = []
    for q in variants:
        q = _strip_quotes(q)
        if _is_valid_question(q):
            cleaned.append(q)

    return cleaned


def _parse_stage2_bullets(summary: str) -> List[str]:
    """Extract atomic fact bullet-points from Stage 2's 7-section structured output."""
    bullets = []
    for line in summary.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d+\.\s", line):
            continue
        if line.lower().strip(".:") == "n/a":
            continue
        if line.startswith("- "):
            fact = line[2:].strip().rstrip(".")
            if len(fact) > 10:
                bullets.append(fact)
    seen: set = set()
    return [b for b in bullets if not (b in seen or seen.add(b))]  # type: ignore[func-returns-value]


def _is_semantic_duplicate(q: str, seen_questions: "set[str]", jaccard_threshold: float = 0.65, containment_threshold: float = 0.90) -> bool:
    """Check if q is a near-duplicate of any question already in seen_questions."""
    w_q = set(q.lower().replace("?", "").split())
    if not w_q:
        return False
    for sq in seen_questions:
        w_sq = set(sq.lower().replace("?", "").split())
        if not w_sq:
            continue
        jaccard = len(w_q & w_sq) / len(w_q | w_sq)
        if jaccard >= jaccard_threshold:
            return True
        shorter = w_q if len(w_q) <= len(w_sq) else w_sq
        if shorter and len(w_q & w_sq) / len(shorter) >= containment_threshold:
            return True
    return False


def generate_questions_from_bullets(
    vlm: VLMBase,
    normality_summary: str,
    class_name: str = "object",
    logger: Optional[PipelineLogger] = None,
) -> List[str]:
    """
    Stage 3a (structured mode): generate one Yes/No question per Stage 2 bullet-point.

    Guarantees that every atomic constraint section (Components, Quantities, Spatial,
    Visual, Relational, Symmetry, Per-Slot) is covered rather than letting the VLM
    cluster questions around the most salient constraint.
    """
    bullets = _parse_stage2_bullets(normality_summary)
    print(f"  [Stage 3a/structured] {len(bullets)} bullets parsed from Stage 2 summary.")
    questions: List[str] = []
    seen: set = set()
    for fact in bullets:
        prompt = BULLET_TO_QUESTION_PROMPT.format(class_name=class_name, fact=fact)
        response = vlm.query(prompt=prompt, image=None)
        q = response.text.strip()
        q = _strip_quotes(q)
        is_dup = _is_semantic_duplicate(q, seen)
        if q and not is_dup and _is_valid_question(q) and _is_semantically_valid_question(q):
            questions.append(q)
            seen.add(q)
            print(f"    fact: {fact[:60]} → {q[:80]}")
        else:
            reason = "NEAR-DUP" if is_dup else "INVALID"
            print(f"    fact: {fact[:60]} → {reason} ({q[:60]})")
    print(f"  [Stage 3a/structured] Generated {len(questions)} questions from {len(bullets)} bullets.")
    if logger:
        logger.log_stage3a_questions(
            prompt="(structured-from-bullets)",
            response_text=str(questions),
            parsed_questions=questions,
        )
    return questions


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
    # Improvement 1: filter out structurally malformed questions before any VLM call
    valid = [q for q in questions if _is_semantically_valid_question(q)]
    dropped = [q for q in questions if q not in valid]
    if dropped:
        print(f"   [Improvement 1] Dropped {len(dropped)} malformed questions: {dropped}")
    questions = valid
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
    normality_summary: str = "",
) -> Optional[str]:
    """Ask a single question about one image and return 'Yes'/'No'/None."""
    if isinstance(image, (str, Path)):
        img = Image.open(str(image)).convert("RGB")
    else:
        img = image

    prompt = TEST_PROMPT.format(question=question, class_name=class_name, class_context=normality_summary)
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


def _adaptive_threshold(n_shots: Optional[int], base_threshold: float) -> float:
    """
    Improvement 2: Compute adaptive filtering threshold based on few-shot count.

    More shots → more reliable per-question accuracy estimate → can afford to
    be slightly more lenient without letting noisy questions through.
    Fewer shots → noisier estimate → require higher agreement to be safe.

    Schedule:
        n_shots <= 3  → 1.0  (all images must say Yes — 3 shots is very noisy)
        n_shots <= 6  → 0.8  (4/5 or 5/6 must say Yes)
        n_shots <= 10 → 0.7
        n_shots > 10  → 0.6
    """
    if n_shots is None:
        return base_threshold
    if n_shots <= 3:
        return 1.0
    if n_shots <= 6:
        return 0.8
    if n_shots <= 10:
        return 0.7
    return 0.6


def filter_questions_on_normal(
    vlm: VLMBase,
    candidate_questions: List[str],
    normal_images: List[Union[Path, Image.Image]],
    threshold: float = 0.8,
    class_name: str = "object",
    image_paths: Optional[List[str]] = None,
    logger: Optional[PipelineLogger] = None,
    n_shots: Optional[int] = None,
    normality_summary: str = "",
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
        n_shots:             If provided, overrides threshold with adaptive schedule
                             (Improvement 2): n_shots≤5 → 1.0, ≤8 → 0.8, >8 → 0.7.

    Returns:
        Filtered list of main questions.
    """
    # Improvement 2: adaptive threshold
    threshold = _adaptive_threshold(n_shots, threshold)

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
            answer = _answer_single_question(vlm, q, img, class_name, logger, gt_label, str(image_paths[i]), normality_summary)
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

        n_parsed = len(variants)
        while len(variants) < n_variants:
            variants.append(mq)
        print(f"  Q{i+1}: parsed={n_parsed}, padded={n_variants - n_parsed}, total={len(variants[:n_variants])}")

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
