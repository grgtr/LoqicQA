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

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from PIL import Image

from logicqa.vlm.base import VLMBase
from logicqa.prompts import TEST_PROMPT
from logicqa.data.normality_definitions import CLASS_INSPECTION_CONTEXTS
from logicqa.logging import PipelineLogger


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


def _ask_sub_question(
    vlm: VLMBase,
    question: str,
    image: Image.Image,
    class_name: str = "object",
    main_question: Optional[str] = None,
    sub_q_idx: Optional[int] = None,
    image_path: Optional[str] = None,
    logger: Optional[PipelineLogger] = None,
) -> SubQResult:
    """Ask one sub-question about an image and return the result."""
    normalized_class_name = class_name.lower().replace(" ", "_")
    class_context = CLASS_INSPECTION_CONTEXTS.get(normalized_class_name, "")
    prompt = TEST_PROMPT.format(question=question, class_name=class_name, class_context=class_context)
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

def _compute_anomaly_score(main_q_results: List[MainQResult]) -> float:
    if not main_q_results:
        return 0.5

    S = []
    for mq in main_q_results:
        lp = mq.best_log_prob if mq.best_log_prob is not None else -1.0
        S.append(math.exp(max(lp, -30)))

    median_s = float(np.median(S))
    is_anomaly = any(mq.voted_answer == "No" for mq in main_q_results)

    if is_anomaly:
        return min(median_s, 1.0)
    else:
        return max(1.0 - median_s, 0.0)

def test_image(
    vlm: VLMBase,
    image: Union[Path, Image.Image],
    main_questions: List[str],
    sub_questions: Dict[str, List[str]],
    image_path: Optional[str] = None,
    class_name: str = "object",
    logger: Optional[PipelineLogger] = None,
    gt_label="unknown",
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
    if isinstance(image, (str, Path)):
        pil_img = Image.open(str(image)).convert("RGB")
        image_path = image_path or str(image)
    else:
        pil_img = image
    if logger:
        logger.log_stage4_image_start(
            image_idx=0, image_path=image_path or "", gt_label=gt_label
        )
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

    # Final decision: anomaly if ANY main-Q votes "No"
    is_anomaly = len(violating_questions) > 0
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
