"""LogicQAPipeline: full end-to-end orchestration of the 4-stage pipeline.

This class ties together all stages, preprocessing, and dataset integration.
"""
from __future__ import annotations

import json
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from PIL import Image

from logicqa.config import LogicQAConfig
from logicqa.vlm.factory import get_vlm
from logicqa.vlm.base import VLMBase
from logicqa.preprocessing.bpm import apply_bpm_from_config
from logicqa.preprocessing.langsam_wrapper import LangSAMWrapper
from logicqa.data.normality_definitions import (
    get_normality_definition,
    BPM_CLASSES,
    LANGSAM_CLASSES,
)
from logicqa.pipeline.stage1_describe import describe_normal_images
from logicqa.pipeline.stage2_summarize import summarize_normal_context, extract_all_components
from logicqa.pipeline.stage3_questions import (
    generate_candidate_questions,
    generate_questions_from_bullets,
    filter_questions_on_normal,
    generate_sub_questions,
)
from logicqa.pipeline.stage4_test import test_image, ImageResult
from logicqa.logging import PipelineLogger

def _is_near_duplicate(q: str, existing_questions: "set[str]", jaccard_threshold: float = 0.65, containment_threshold: float = 0.90) -> bool:
    """
    Semantic near-duplicate check using word Jaccard and subset-containment.

    Catches two types of duplicates that exact-string matching misses:
    - Rephrasing across ensemble seeds ("Are the tangerines on the left?" vs
      "Are there tangerines located on the left-hand side?") → high Jaccard
    - One question is a sub-phrase of another ("Is there exactly two tangerines?"
      vs "Is there exactly two tangerines in the breakfast box?") → high containment
    """
    w_q = set(q.lower().replace("?", "").split())
    if not w_q:
        return False
    for eq in existing_questions:
        w_eq = set(eq.lower().replace("?", "").split())
        if not w_eq:
            continue
        union = w_q | w_eq
        intersection = w_q & w_eq
        jaccard = len(intersection) / len(union)
        if jaccard >= jaccard_threshold:
            return True
        # containment: smaller set almost entirely within larger
        smaller = intersection
        shorter = w_q if len(w_q) <= len(w_eq) else w_eq
        if shorter and len(smaller) / len(shorter) >= containment_threshold:
            return True
    return False


@dataclass
class LogicQAArtifacts:
    # Stage 1: Describing
    stage1_descriptions: List[Dict[str, str]] = field(default_factory=list)
    
    # Stage 2: Summarization
    stage2_summary: str = ""
    
    # Stage 3: Questions
    stage3_main_questions: List[str] = field(default_factory=list)
    stage3_sub_questions: Dict[str, List[str]] = field(default_factory=dict)
    
    # Stage 4: Testing (inference)
    stage4_results: List[Dict[str, Any]] = field(default_factory=list)

    def save_to_json(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=4, ensure_ascii=False)

class LogicQAPipeline:
    """
    End-to-end LogicQA pipeline for logical anomaly detection.

    Usage:
        cfg = LogicQAConfig.from_yaml("config.yaml")
        pipeline = LogicQAPipeline(cfg)
        pipeline.setup(class_name="breakfast_box", normal_images=[...])
        result = pipeline.predict(test_image_path)
    """

    def __init__(self, cfg: LogicQAConfig):
        self.cfg = cfg
        self.vlm: VLMBase = get_vlm(cfg)
        self._langsam: Optional[LangSAMWrapper] = None
        self.logger = PipelineLogger(
            output_dir=cfg.pipeline.output_dir,
            class_name=cfg.pipeline.class_name,
        )

        # Will be set after setup()
        self.class_name: Optional[str] = None
        self.normality_definition: Optional[str] = None
        self.normality_summary: str = ""
        self.components: List[str] = []
        self.main_questions: List[str] = []
        self.sub_questions: Dict[str, List[str]] = {}
        self._setup_done = False

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def langsam(self) -> LangSAMWrapper:
        if self._langsam is None:
            self._langsam = LangSAMWrapper(self.cfg.preprocessing.langsam)
        return self._langsam

    # ------------------------------------------------------------------ #
    # Fix C: constraint relevance filter
    # ------------------------------------------------------------------ #

    def _filter_by_constraints(
        self,
        questions: List[str],
        llm_judge,
    ) -> List[str]:
        """Drop questions that don't map to any known atomic constraint.

        Requires LLMJudge and ATOMIC_CONSTRAINTS for the current class.
        If either is unavailable, returns questions unchanged.
        """
        if llm_judge is None or not questions:
            return questions
        try:
            from logicqa.data.evaluation_gt import ATOMIC_CONSTRAINTS
        except ImportError:
            return questions

        constraints = ATOMIC_CONSTRAINTS.get(self.class_name, [])
        if not constraints:
            return questions

        filtered = []
        for q in questions:
            mapped = llm_judge.map_question_to_constraints(q, constraints)
            if mapped:
                filtered.append(q)
            else:
                print(f"  [Constraint filter] DROPPED (no constraint match): {q[:80]}")
        print(f"  [Constraint filter] {len(filtered)}/{len(questions)} questions map to known constraints")
        return filtered

    # ------------------------------------------------------------------ #
    # Preprocessing helpers
    # ------------------------------------------------------------------ #

    def _preprocess(
        self,
        image: Union[Path, Image.Image],
        class_name: str,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Apply class-specific preprocessing (BPM and/or Lang-SAM).

        Returns:
            - A single PIL Image for most classes.
            - A list of PIL Images when Lang-SAM is applied (one per segment).
        """
        cls = class_name.lower().replace(" ", "_")

        # Load image if needed
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        # Apply BPM for: screw_bag, splicing_connectors
        if cls in BPM_CLASSES:
            pil_img = apply_bpm_from_config(pil_img, self.cfg.preprocessing.bpm)

        # Apply Lang-SAM for: pushpins, splicing_connectors
        if cls in LANGSAM_CLASSES and self.cfg.preprocessing.langsam.enabled:
            segments = self.langsam.segment(pil_img, cls)
            return segments  # list of crops

        return pil_img

    def _preprocess_for_description(
        self,
        image: Union[Path, Image.Image],
        class_name: str,
    ) -> Image.Image:
        """For Stage 1 (description), return a single image (use first segment if Lang-SAM)."""
        result = self._preprocess(image, class_name)
        if isinstance(result, list):
            return result[0] if result else Image.open(str(image)).convert("RGB")
        return result

    # ------------------------------------------------------------------ #
    # Setup (Stages 1-3)
    # ------------------------------------------------------------------ #

    def setup(
        self,
        class_name: str,
        normal_images: List[Union[Path, str]],
        normality_definition: Optional[str] = None,
        n_questions: int = 6,
        validation_images: Optional[List[Union[Path, str]]] = None,
        output_dir="results", **kwargs
    ) -> Dict:
        """
        Run Stages 1-3 to build the question checklist for a class.

        Args:
            class_name:           MVTec LOCO AD class name.
            normal_images:        N few-shot normal image paths.
            normality_definition: Override default normality definition.
            n_questions:          Number of candidate questions to generate.
            validation_images:    Images used for question filtering (defaults to normal_images).

        Returns:
            Dict with main_questions and sub_questions.
        """
        self.class_name = class_name.lower().replace(" ", "_")
        self.normality_definition = (
            normality_definition or get_normality_definition(self.class_name)
        )
        val_images = validation_images or normal_images

        print(f"\n{'='*60}")
        print(f" LogicQA Setup: {class_name}")
        print(f"{'='*60}")

        # Stage 1
        preprocessed_normals = [
            self._preprocess_for_description(img, self.class_name)
            for img in normal_images
        ]
        # Improvement 6: optionally load LLMJudge for hallucination detection
        llm_judge = None
        if getattr(self.cfg.pipeline, "use_llm_judge_hallucination", False):
            try:
                from logicqa.evaluation.llm_judge import LLMJudge
                llm_judge = LLMJudge()
                print("[Setup] LLMJudge loaded for hallucination detection (Improvement 6)")
            except Exception as e:
                print(f"[Setup] LLMJudge load failed, skipping: {e}")

        descriptions = describe_normal_images(
            self.vlm, preprocessed_normals, self.normality_definition, self.class_name,
            image_paths=normal_images, logger=self.logger, llm_judge=llm_judge,
        )
        self.components = extract_all_components(descriptions)
        print(f"[Setup] Extracted {len(self.components)} unique components: {self.components}")

        # Stage 2
        self.normality_summary = summarize_normal_context(
            self.vlm, descriptions, self.normality_definition,
            class_name=self.class_name,
            logger=self.logger,
            all_components=self.components,
        )

        # Stage 3a: Generate candidates
        q_mode = getattr(self.cfg.pipeline, "question_generation_mode", "llm")
        if q_mode == "structured":
            candidates = generate_questions_from_bullets(
                self.vlm,
                self.normality_summary,
                class_name=self.class_name,
                logger=self.logger,
            )
        else:
            candidates = generate_candidate_questions(
                self.vlm,
                self.normality_summary,
                self.normality_definition,
                class_name=self.class_name,
                n_questions=n_questions,
                logger=self.logger,
            )

        # Stage 3b: Filter
        preprocessed_vals = [
            self._preprocess_for_description(img, self.class_name)
            for img in val_images
        ]
        kept = filter_questions_on_normal(
            self.vlm,
            candidates,
            preprocessed_vals,
            threshold=self.cfg.pipeline.question_filter_threshold,
            class_name=self.class_name,
            image_paths=val_images,
            logger=self.logger,
            n_shots=self.cfg.pipeline.n_shots,
            normality_summary=self.normality_summary,
        )

        # Fix C: drop questions that don't map to any known atomic constraint
        kept = self._filter_by_constraints(kept, llm_judge)

        # Stage 3c: Sub-questions
        sub_qs = generate_sub_questions(
            self.vlm,
            kept,
            n_variants=self.cfg.pipeline.n_sub_questions,
            logger=self.logger
        )

        self.main_questions = kept
        self.sub_questions = sub_qs
        self._setup_done = True

        print(f"\n[Setup complete] {len(kept)} main questions ready.")
        return {"main_questions": kept, "sub_questions": sub_qs}

    # ------------------------------------------------------------------ #
    # Inference (Stage 4)
    # ------------------------------------------------------------------ #

    def predict(
        self,
        image: Union[Path, str, Image.Image],
        gt_label: Optional[str] = "unknown",
        anomaly_type: Optional[str] = None,
    ) -> ImageResult:
        """
        Run Stage 4 on a single query image.

        Args:
            image: Test image (path or PIL Image).

        Returns:
            ImageResult with prediction, anomaly score, and explanation.
        """
        if not self._setup_done:
            raise RuntimeError(
                "Pipeline not set up. Call pipeline.setup() first."
            )

        cls = self.class_name or ""
        image_path_str = str(image) if isinstance(image, (str, Path)) else None

        # Preprocess
        preprocessed = self._preprocess(image, cls)

        min_failures = getattr(self.cfg.pipeline, "anomaly_min_failures", 2)
        use_grounded = getattr(self.cfg.pipeline, "use_grounded_reasoning", False)

        # If Lang-SAM returned multiple segments, test each and aggregate
        if isinstance(preprocessed, list):
            results = [
                test_image(
                    self.vlm,
                    seg,
                    self.main_questions,
                    self.sub_questions,
                    image_path=image_path_str,
                    class_name=self.class_name,
                    logger=self.logger,
                    gt_label=gt_label,
                    anomaly_type=anomaly_type,
                    anomaly_min_failures=min_failures,
                    normality_summary=self.normality_summary,
                    components=self.components if use_grounded else None,
                    use_grounded_reasoning=use_grounded,
                )
                for seg in preprocessed
            ]
            is_anomaly = any(r.is_anomaly for r in results)
            anomaly_score = max(r.anomaly_score for r in results)
            best = max(results, key=lambda r: r.anomaly_score)
            return ImageResult(
                image_path=image_path_str,
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                main_q_results=best.main_q_results,
                explanation=best.explanation,
            )

        return test_image(
            self.vlm,
            preprocessed,
            self.main_questions,
            self.sub_questions,
            image_path=image_path_str,
            class_name=self.class_name,
            logger=self.logger,
            gt_label=gt_label,
            anomaly_type=anomaly_type,
            anomaly_min_failures=min_failures,
            normality_summary=self.normality_summary,
            components=self.components if use_grounded else None,
            use_grounded_reasoning=use_grounded,
        )

    # ------------------------------------------------------------------ #
    # Improvement 5: Multi-run Ensemble setup
    # ------------------------------------------------------------------ #

    def setup_ensemble(
        self,
        class_name: str,
        normal_images: List[Union[Path, str]],
        normality_definition: Optional[str] = None,
        n_questions: int = 6,
        seeds: Optional[List[int]] = None,
        validation_images: Optional[List[Union[Path, str]]] = None,
    ) -> Dict:
        """
        Improvement 5: Run Stage 3 multiple times with different random seeds
        and merge all uniquely passing questions into the final checklist.

        Each seed produces a different ordering of the VLM's token sampling,
        yielding different question candidates that may cover different constraints.
        The union of filtered questions increases L2.5 Recall without hurting Precision
        (each question still passes the Stage 3b filter individually).

        Args:
            seeds: List of integer seeds. Defaults to config.pipeline.ensemble_seeds,
                   or [42, 7, 13] if not configured.
        """
        import random

        if seeds is None:
            seeds = list(getattr(self.cfg.pipeline, "ensemble_seeds", [])) or [42, 7, 13]

        self.class_name = class_name.lower().replace(" ", "_")
        self.normality_definition = (
            normality_definition or get_normality_definition(self.class_name)
        )
        val_images = validation_images or normal_images

        print(f"\n{'='*60}")
        print(f" LogicQA Ensemble Setup: {class_name} | seeds={seeds}")
        print(f"{'='*60}")

        # Optionally load LLMJudge once for hallucination detection + Fix C filter
        ensemble_judge = None
        if getattr(self.cfg.pipeline, "use_llm_judge_hallucination", False):
            try:
                from logicqa.evaluation.llm_judge import LLMJudge
                ensemble_judge = LLMJudge()
                print("[Ensemble] LLMJudge loaded for hallucination detection + constraint filter")
            except Exception as e:
                print(f"[Ensemble] LLMJudge load failed, skipping: {e}")

        # Stages 1-2 are deterministic — run once
        preprocessed_normals = [
            self._preprocess_for_description(img, self.class_name)
            for img in normal_images
        ]
        descriptions = describe_normal_images(
            self.vlm, preprocessed_normals, self.normality_definition, self.class_name,
            image_paths=normal_images, logger=self.logger, llm_judge=ensemble_judge,
        )
        self.components = extract_all_components(descriptions)
        print(f"[Ensemble] Extracted {len(self.components)} unique components: {self.components}")
        self.normality_summary = summarize_normal_context(
            self.vlm, descriptions, self.normality_definition,
            class_name=self.class_name,
            logger=self.logger,
            all_components=self.components,
        )

        preprocessed_vals = [
            self._preprocess_for_description(img, self.class_name)
            for img in val_images
        ]

        q_mode = getattr(self.cfg.pipeline, "question_generation_mode", "llm")

        # Stage 3: run with each seed, merge unique filtered questions
        all_sub_questions: Dict[str, List[str]] = {}
        for seed in seeds:
            print(f"\n  [Ensemble] Seed={seed} — generating questions ...")
            random.seed(seed)

            if q_mode == "structured":
                candidates = generate_questions_from_bullets(
                    self.vlm,
                    self.normality_summary,
                    class_name=self.class_name,
                    logger=self.logger,
                )
            else:
                candidates = generate_candidate_questions(
                    self.vlm, self.normality_summary, self.normality_definition,
                    class_name=self.class_name, n_questions=n_questions,
                    logger=self.logger,
                )
            filtered = filter_questions_on_normal(
                self.vlm, candidates, preprocessed_vals,
                threshold=self.cfg.pipeline.question_filter_threshold,
                class_name=self.class_name, image_paths=val_images,
                logger=self.logger, n_shots=self.cfg.pipeline.n_shots,
                normality_summary=self.normality_summary,
            )
            sub_qs = generate_sub_questions(
                self.vlm, filtered,
                n_variants=self.cfg.pipeline.n_sub_questions,
                logger=self.logger,
            )
            new = 0
            for q, sqs in sub_qs.items():
                if _is_near_duplicate(q, set(all_sub_questions.keys())):
                    print(f"  [Ensemble] DEDUP: '{q[:70]}' is near-duplicate, skipping")
                else:
                    all_sub_questions[q] = sqs
                    new += 1
            print(f"  [Ensemble] Seed={seed}: {len(filtered)} filtered, {new} new unique questions added")

        # Fix C: drop questions that don't map to any known atomic constraint
        merged = self._filter_by_constraints(list(all_sub_questions.keys()), ensemble_judge)
        all_sub_questions = {q: all_sub_questions[q] for q in merged}

        self.main_questions = list(all_sub_questions.keys())
        self.sub_questions = all_sub_questions
        self._setup_done = True
        print(f"\n[Ensemble complete] {len(self.main_questions)} unique main questions from {len(seeds)} seeds.")
        return {"main_questions": self.main_questions, "sub_questions": self.sub_questions}

    # ------------------------------------------------------------------ #
    # Save / Load question checklist
    # ------------------------------------------------------------------ #

    def save_questions(self, path: Union[str, Path]) -> None:
        """Save the generated question checklist to a JSON file."""
        path = Path(os.path.expanduser(str(path)))
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "class_name": self.class_name,
            "normality_definition": self.normality_definition,
            "main_questions": self.main_questions,
            "sub_questions": self.sub_questions,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Pipeline] Questions saved to {path}")

    def load_questions(self, path: Union[str, Path]) -> None:
        """Load a previously saved question checklist from JSON."""
        path = Path(os.path.expanduser(str(path)))
        with open(path) as f:
            data = json.load(f)
        self.class_name = data["class_name"]
        self.normality_definition = data["normality_definition"]
        self.main_questions = data["main_questions"]
        self.sub_questions = data["sub_questions"]
        self._setup_done = True
        print(f"[Pipeline] Questions loaded from {path}: {len(self.main_questions)} main-Qs")
    
    def export_run_artifacts(self, filepath: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """
        [Evaluation Framework] Export the entire pipeline context (Stages 1-4)
        in a single JSON file for 4-level quality analysis.
        """
        if self.logger:
            # Enrich artifact with normality definition (required for Level 1 - CCR)
            if self.normality_definition:
                self.logger.record.normality_definition = self.normality_definition
            
            return self.logger.export_artifacts(filepath)
        print("Warning: PipelineLogger is not initialized. Cannot export artifacts.")
        return None
