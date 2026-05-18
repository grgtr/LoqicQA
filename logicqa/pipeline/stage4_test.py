"""Stage 4: Test query images using Main-Qs / Sub-Qs and majority voting.

Pipeline (per image):
    For each Main-Q:
        1. Ask all 5 Sub-Qs → get Yes/No + log-prob per sub-Q.
        2. Majority vote among Sub-Qs → main-Q decision.
        3. Collect the highest log-prob among sub-Qs matching the voted answer.
    Final decision:
        → "anomaly" if ANY Main-Q votes "No", else "normal".
    Anomaly score (for AUROC / F1-max):
        → product of exp(log_prob_i) for each Main-Q_i.
        → Higher score = more anomalous.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from PIL import Image

from logicqa.vlm.base import VLMBase
from logicqa.prompts import (
    TEST_PROMPT,
    LOCALIZATION_PROMPT,
)
from logicqa.pipeline.stage1_describe import describe_image_decomposed
from logicqa.logging import PipelineLogger
from logicqa.pipeline.stage2_summarize import _dedup_key


@dataclass
class SubQResult:
    question: str
    answer: Optional[str]   # "Yes" / "No" / None
    log_prob: Optional[float]


@dataclass
class MainQResult:
    question: str
    voted_answer: str                    # "Yes" or "No" (majority vote)
    sub_results: List[SubQResult] = field(default_factory=list)
    best_log_prob: Optional[float] = None   # highest log-prob matching voted answer


@dataclass
class ImageResult:
    image_path: Optional[str]
    is_anomaly: bool                          # final prediction
    anomaly_score: float                      # for AUROC/F1-max (higher = more anomalous)
    main_q_results: List[MainQResult] = field(default_factory=list)
    explanation: str = ""                     # human-readable explanation


def _dedup_components_for_grounding(components: List[str]) -> List[str]:
    """
    Deduplicate raw component names before grounding using canonical key.

    Prevents sending redundant VLM queries for "Two tangerines", "Tangerine",
    and "Two tangerines (one above the other)" — all map to the same object.
    Returns one display name per unique canonical key (first occurrence wins).
    """
    seen_keys: set = set()
    result: List[str] = []
    for comp in components:
        key = _dedup_key(comp)
        if key and key not in seen_keys:
            seen_keys.add(key)
            result.append(comp)
    return result


def localize_components(
    vlm: VLMBase,
    image: Image.Image,
    components: List[str],
    class_name: str = "object",
) -> Dict[str, str]:
    """Ask the VLM to locate each known component in the image.

    Returns a dict mapping component name → one-line position description.
    Called once per test image before the question loop.
    """
    result: Dict[str, str] = {}
    for comp in components:
        prompt = LOCALIZATION_PROMPT.format(class_name=class_name, component=comp)
        response = vlm.query(prompt=prompt, image=image)
        first_line = response.text.strip().splitlines()[0] if response.text.strip() else "Not found"
        result[comp] = first_line
        print(f"  [Grounding] {comp}: {first_line}")
    return result


def format_grounding_map(grounding_map: Dict[str, str]) -> str:
    """Format grounding results as a structured string for injection into TEST_PROMPT."""
    if not grounding_map:
        return ""
    lines = ["Located objects:"] + [f"- {k}: {v}" for k, v in grounding_map.items()]
    return "\n".join(lines)


def _ask_sub_question(
    vlm: VLMBase,
    question: str,
    image: Image.Image,
    class_name: str = "object",
    main_question: Optional[str] = None,
    sub_q_idx: Optional[int] = None,
    image_path: Optional[str] = None,
    logger: Optional[PipelineLogger] = None,
    normality_summary: str = "",
    grounding_context: str = "",
    current_image_description: str = "",
) -> SubQResult:
    """Ask one sub-question about an image and return the result."""
    prompt = TEST_PROMPT.format(
        question=question,
        class_name=class_name,
        class_context=normality_summary,
        current_image_description=current_image_description or "(not available)",
        grounding_context=grounding_context or "(not available)",
    )
    if hasattr(vlm, "query_with_logprobs"):
        # print("[DEBUG] using query_with_logprobs in stage4_test")
        response = vlm.query_with_logprobs(prompt=prompt, image=image)
    else:
        # print("[DEBUG] using query in stage4_test")
        response = vlm.query(prompt=prompt, image=image)
    if logger:
        logger.log_stage4_sub_question(
            image_path=image_path,
            main_question=main_question,
            sub_question=question,
            sub_q_idx=sub_q_idx,
            prompt=prompt,
            response_text=response.text,
            extracted_answer=response.answer,
            log_prob=response.log_prob,
            extraction_meta=response.extraction_meta
        )
    return SubQResult(
        question=question,
        answer=response.answer,
        log_prob=response.log_prob,
    )


def _majority_vote(answers: List[Optional[str]]) -> str:
    """Return 'Yes' or 'No' by majority vote. Tie → 'No' (conservative)."""
    yes_count = sum(1 for a in answers if a == "Yes")
    no_count = sum(1 for a in answers if a == "No")
    return "Yes" if yes_count > no_count else "No"


# def _compute_anomaly_score(main_q_results: List[MainQResult]) -> float:
#     """
#     Compute the anomaly score from Main-Q log-probabilities.

#     Per the paper's formulation:
#         score_i = exp(best_log_prob_i)   for each main-Q i
#         final_score = product(score_i)

#     A higher score indicates higher anomaly confidence.
#     When a main-Q votes 'No', its exp(log_prob) contributes to the score.
#     When it votes 'Yes' (normal), its contribution is low (near 0).

#     To compute AUROC, we use: anomaly_score = 1 - product(exp(log_p_i for Yes Qs))
#     = probability that at least one constraint is violated.
#     """
#     if not main_q_results:
#         return 0.5

#     log_prob_no_answers = []
#     for mq in main_q_results:
#         lp = mq.best_log_prob
#         if lp is None:
#             lp = -1.0  # default when log-prob unavailable
#         if mq.voted_answer == "No":
#             log_prob_no_answers.append(lp)

#     if not log_prob_no_answers:
#         # All 'Yes' → compute inverse: 1 - product of Yes probabilities
#         log_prob_yes = []
#         for mq in main_q_results:
#             lp = mq.best_log_prob if mq.best_log_prob is not None else -1.0
#             log_prob_yes.append(lp)
#         log_prod_yes = sum(log_prob_yes)
#         prob_all_yes = math.exp(max(log_prod_yes, -30))
#         return 1.0 - min(prob_all_yes, 1.0)

#     # Some 'No' answers → anomaly score from No answers
#     log_sum = sum(log_prob_no_answers)
#     return min(math.exp(max(log_sum, -30)), 1.0)

import numpy as np


def _subq_consistency(sub_results: List[SubQResult]) -> float:
    """
    Improvement 3: Compute consistency score for a set of sub-question answers.

    Measures how confidently the sub-questions agree:
      - 5:0 or 0:5 split → variance=0 → consistency=1.0 (full agreement)
      - 3:2 split        → variance=0.24 → consistency=0.0 (maximum confusion)

    Used to weight each main question's contribution to the anomaly score.
    """
    if not sub_results:
        return 1.0
    yes_count = sum(1 for r in sub_results if r.answer == "Yes")
    p = yes_count / len(sub_results)
    variance = p * (1.0 - p)
    return max(0.0, 1.0 - variance / 0.24)


def _compute_anomaly_score(main_q_results: List[MainQResult]) -> float:
    """
    Count-based anomaly score: fraction of main questions that voted "No".

    Returns a value in [0, 1] that monotonically increases with the number of
    violated constraints. This gives a smooth signal for AUROC computation even
    when all binary predictions are identical.

    Example with 8 questions:
      0 fail → 0.000 (clearly normal)
      1 fail → 0.125 (noisy / likely FP)
      2 fail → 0.250 (borderline)
      4 fail → 0.500 (likely anomaly)
    """
    if not main_q_results:
        return 0.0

    n_no = sum(1 for mq in main_q_results if mq.voted_answer == "No")
    return n_no / len(main_q_results)



def test_image(
    vlm: VLMBase,
    image: Union[Path, Image.Image],
    main_questions: List[str],
    sub_questions: Dict[str, List[str]],
    image_path: Optional[str] = None,
    class_name: str = "object",
    logger: Optional[PipelineLogger] = None,
    gt_label="unknown",
    anomaly_type: Optional[str] = None,
    anomaly_min_failures: int = 2,
    normality_summary: str = "",
    components: Optional[List[str]] = None,
    use_grounded_reasoning: bool = False,
    use_decomposed_description: bool = False,
) -> ImageResult:
    """
    Stage 4: Test a single query image with the generated question checklist.

    Args:
        vlm:            VLM backend.
        image:          Query image (PIL or path).
        main_questions: Filtered main questions from Stage 3.
        sub_questions:  Dict: main_Q → list of 5 sub-Qs from Stage 3.
        image_path:     Optional path string for record-keeping.

    Returns:
        ImageResult with prediction, score, and per-question breakdown.
    """
    print("[DEBUG] Test image, anomaly_type=",anomaly_type)
    if isinstance(image, (str, Path)):
        pil_img = Image.open(str(image)).convert("RGB")
        image_path = image_path or str(image)
    else:
        pil_img = image
    if logger:
        logger.log_stage4_image_start(
            image_idx=0, image_path=image_path or "", gt_label=gt_label, anomaly_type=anomaly_type
        )

    # Pre-description + grounding: describe the test image before the question loop
    current_image_description = ""
    grounding_context = ""
    if use_decomposed_description and components:
        print(f"  [Stage 4 Decomposed] Describing test image ({len(components)} components) ...")
        current_image_description, grounding_context = describe_image_decomposed(
            vlm, pil_img, components, class_name,
            normality_definition=normality_summary,
        )
        print(f"  [Grounding] Context built from per-component describe:\n{grounding_context}")
    elif use_grounded_reasoning and components:
        deduped = _dedup_components_for_grounding(components)
        print(f"  [Grounding] Localizing {len(deduped)} components (deduped from {len(components)}) ...")
        grounding_map = localize_components(vlm, pil_img, deduped, class_name)
        grounding_context = format_grounding_map(grounding_map)
        print(f"  [Grounding] Context built:\n{grounding_context}")

    main_q_results: List[MainQResult] = []
    violating_questions: List[str] = []

    for mq in main_questions:
        sub_qs = sub_questions.get(mq, [mq])
        sub_results: List[SubQResult] = []

        for idx, sq in enumerate(sub_qs):
            sub_result = _ask_sub_question(
                vlm, sq, pil_img,
                class_name=class_name,
                main_question=mq,
                sub_q_idx=idx + 1,
                image_path=image_path or "",
                logger=logger,
                normality_summary=normality_summary,
                grounding_context=grounding_context,
                current_image_description=current_image_description,
            )
            sub_results.append(sub_result)

        # Majority vote
        answers = [r.answer for r in sub_results]
        voted = _majority_vote(answers)

        # Best log-prob among sub-Qs that match the voted answer
        matching_lp = [
            r.log_prob for r in sub_results
            if r.answer == voted and r.log_prob is not None
        ]
        best_lp = max(matching_lp) if matching_lp else None

        mq_result = MainQResult(
            question=mq,
            voted_answer=voted,
            sub_results=sub_results,
            best_log_prob=best_lp,
        )
        main_q_results.append(mq_result)

        if logger:
            logger.log_stage4_main_question_result(mq, voted, answers)

        if voted == "No":
            violating_questions.append(mq)

    is_anomaly = len(violating_questions) >= anomaly_min_failures
    anomaly_score = _compute_anomaly_score(main_q_results)

    # Human-readable explanation
    if is_anomaly:  
        explanation = (
            "Logical anomaly detected. Violated constraints:\n"
            + "\n".join(f"  - {q}" for q in violating_questions)
        )
    else:
        explanation = "Image appears normal. All constraints satisfied."
    if logger:
        logger.log_stage4_image_result(
            image_path=image_path or "",
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            explanation=explanation,
        )
    return ImageResult(
        image_path=image_path,
        is_anomaly=is_anomaly,
        anomaly_score=anomaly_score,
        main_q_results=main_q_results,
        explanation=explanation,
    )
