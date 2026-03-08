#!/usr/bin/env python3
"""
tests/test_pipeline_smoke.py — Smoke test for the full LogicQA pipeline
using a mock VLM (no GPU or API key required).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from logicqa.vlm.base import VLMBase, VLMResponse
from logicqa.pipeline.stage1_describe import describe_normal_images
from logicqa.pipeline.stage2_summarize import summarize_normal_context
from logicqa.pipeline.stage3_questions import (
    generate_candidate_questions,
    filter_questions_on_normal,
    generate_sub_questions,
)
from logicqa.pipeline.stage4_test import test_image
from logicqa.evaluation.metrics import compute_auroc, compute_f1_max


# --------------------------------------------------------------------------- #
# Mock VLM that returns deterministic responses
# --------------------------------------------------------------------------- #

class MockVLM(VLMBase):
    """Deterministic mock VLM for testing without any actual model."""

    def __init__(self, answer="Yes", description="A normal image with all items."):
        self.answer = answer
        self.description = description
        self.call_count = 0

    def query(self, prompt, image=None):
        self.call_count += 1
        # Stage 1: return description
        if "describe" in prompt.lower() or "analyze" in prompt.lower():
            return VLMResponse(text=self.description, answer=None, log_prob=None)
        # Stage 2: summary
        if "summariz" in prompt.lower():
            return VLMResponse(text="Summary: all items present.", answer=None, log_prob=None)
        # Stage 3a: generate questions
        if "checklist" in prompt.lower() or "question" in prompt.lower():
            questions = "\n".join([
                "1. Is exactly one juice bottle present?",
                "2. Is the bottle cap on?",
                "3. Is the bottle upright?",
            ])
            return VLMResponse(text=questions, answer=None, log_prob=None)
        # Stage 3b: sub-question augmentation
        if "rephrasing" in prompt.lower() or "variants" in prompt.lower() or "equivalent" in prompt.lower():
            variants = "\n".join([
                "1. Does the image show exactly one juice bottle?",
                "2. Is there a single juice bottle visible?",
                "3. Can you see one and only one juice bottle?",
                "4. Is the count of juice bottles exactly one?",
                "5. Is a single juice bottle present in the image?",
            ])
            return VLMResponse(text=variants, answer=None, log_prob=None)
        # Stage 4: binary Yes/No
        return VLMResponse(text=f"{self.answer}\nBecause the image looks normal.", answer=self.answer, log_prob=-0.1)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def dummy_image() -> Image.Image:
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


@pytest.fixture
def normal_vlm() -> MockVLM:
    return MockVLM(answer="Yes")


@pytest.fixture
def anomaly_vlm() -> MockVLM:
    return MockVLM(answer="No")


NORMALITY_DEF = "A juice bottle must be present, upright, with cap on."

# --------------------------------------------------------------------------- #
# Stage 1 tests
# --------------------------------------------------------------------------- #

def test_stage1_returns_one_description_per_image(dummy_image, normal_vlm):
    images = [dummy_image, dummy_image, dummy_image]
    descs = describe_normal_images(normal_vlm, images, NORMALITY_DEF)
    assert len(descs) == 3
    assert all(isinstance(d, str) and len(d) > 0 for d in descs)


# --------------------------------------------------------------------------- #
# Stage 2 tests
# --------------------------------------------------------------------------- #

def test_stage2_returns_string(normal_vlm):
    descs = ["desc1", "desc2", "desc3"]
    summary = summarize_normal_context(normal_vlm, descs, NORMALITY_DEF)
    assert isinstance(summary, str) and len(summary) > 0


# --------------------------------------------------------------------------- #
# Stage 3 tests
# --------------------------------------------------------------------------- #

def test_stage3_generates_questions(normal_vlm):
    questions = generate_candidate_questions(
        normal_vlm, "Summary text.", NORMALITY_DEF, n_questions=3
    )
    assert len(questions) >= 1


def test_stage3_filter_drops_low_accuracy_questions(dummy_image, anomaly_vlm):
    """Questions where VLM says 'No' to normal images should be dropped."""
    candidates = ["Is there exactly one bottle?", "Is the cap on?"]
    # anomaly_vlm always answers "No" → accuracy=0 → all should be dropped
    kept = filter_questions_on_normal(anomaly_vlm, candidates, [dummy_image], threshold=0.8)
    assert len(kept) == 0


def test_stage3_filter_keeps_good_questions(dummy_image, normal_vlm):
    """Questions where VLM says 'Yes' to normal images should be kept."""
    candidates = ["Is there exactly one bottle?", "Is the cap on?"]
    kept = filter_questions_on_normal(normal_vlm, candidates, [dummy_image], threshold=0.8)
    assert len(kept) == 2


def test_stage3_generates_sub_questions(normal_vlm):
    main_qs = ["Is there exactly one juice bottle?"]
    sub_qs = generate_sub_questions(normal_vlm, main_qs, n_variants=5)
    assert "Is there exactly one juice bottle?" in sub_qs
    assert len(sub_qs["Is there exactly one juice bottle?"]) == 5


# --------------------------------------------------------------------------- #
# Stage 4 tests
# --------------------------------------------------------------------------- #

def test_stage4_normal_image_classified_normal(dummy_image, normal_vlm):
    main_qs = ["Is there exactly one bottle?"]
    sub_qs = {"Is there exactly one bottle?": [
        "Is there exactly one bottle?" * 1,
        "Can you see one bottle?",
        "Is the bottle count one?",
        "Is a single bottle present?",
        "One bottle only?",
    ]}
    result = test_image(normal_vlm, dummy_image, main_qs, sub_qs)
    assert result.is_anomaly is False
    assert result.anomaly_score < 0.5


def test_stage4_anomaly_image_classified_anomaly(dummy_image, anomaly_vlm):
    main_qs = ["Is there exactly one bottle?"]
    sub_qs = {"Is there exactly one bottle?": [
        "Is there exactly one bottle?",
        "Can you see one bottle?",
        "Is the bottle count one?",
        "Is a single bottle present?",
        "One bottle only?",
    ]}
    result = test_image(anomaly_vlm, dummy_image, main_qs, sub_qs)
    assert result.is_anomaly is True
    assert result.anomaly_score > 0.0


# --------------------------------------------------------------------------- #
# Evaluation metrics tests
# --------------------------------------------------------------------------- #

def test_auroc_perfect():
    scores = [0.9, 0.8, 0.1, 0.2]
    labels = [1, 1, 0, 0]
    assert compute_auroc(scores, labels) == pytest.approx(1.0)


def test_auroc_random():
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [1, 0, 1, 0]
    auroc = compute_auroc(scores, labels)
    assert 0.0 <= auroc <= 1.0


def test_f1_max():
    scores = [0.9, 0.8, 0.1, 0.2]
    labels = [1, 1, 0, 0]
    f1 = compute_f1_max(scores, labels)
    assert f1 == pytest.approx(1.0, abs=0.01)
