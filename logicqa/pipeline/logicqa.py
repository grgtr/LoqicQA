"""LogicQAPipeline: full end-to-end orchestration of the 4-stage pipeline.

This class ties together all stages, preprocessing, and dataset integration.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

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
from logicqa.pipeline.stage2_summarize import summarize_normal_context
from logicqa.pipeline.stage3_questions import (
    generate_candidate_questions,
    filter_questions_on_normal,
    generate_sub_questions,
)
from logicqa.pipeline.stage4_test import test_image, ImageResult


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

        # Will be set after setup()
        self.class_name: Optional[str] = None
        self.normality_definition: Optional[str] = None
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
        descriptions = describe_normal_images(
            self.vlm, preprocessed_normals, self.normality_definition
        )

        # Stage 2
        summary = summarize_normal_context(
            self.vlm, descriptions, self.normality_definition
        )

        # Stage 3a: Generate candidates
        candidates = generate_candidate_questions(
            self.vlm, summary, self.normality_definition, n_questions=n_questions
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
        )

        # Stage 3c: Sub-questions
        sub_qs = generate_sub_questions(
            self.vlm, kept, n_variants=self.cfg.pipeline.n_sub_questions
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

        # If Lang-SAM returned multiple segments, test each and aggregate
        if isinstance(preprocessed, list):
            results = [
                test_image(self.vlm, seg, self.main_questions, self.sub_questions)
                for seg in preprocessed
            ]
            # Aggregate: anomaly if ANY segment is anomalous; max anomaly score
            is_anomaly = any(r.is_anomaly for r in results)
            anomaly_score = max(r.anomaly_score for r in results)
            best = max(results, key=lambda r: r.anomaly_score)
            best.is_anomaly = is_anomaly
            best.anomaly_score = anomaly_score
            best.image_path = image_path_str
            return best

        return test_image(
            self.vlm,
            preprocessed,
            self.main_questions,
            self.sub_questions,
            image_path=image_path_str,
        )

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
