"""Abstract VLM interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from PIL import Image

import re

@dataclass
class VLMResponse:
    """Structured response from a VLM query."""
    text: str                  # Full generated text (e.g., "Yes" or "No" + reasoning)
    answer: Optional[str]      # Extracted binary answer: "Yes" or "No" (or None if unclear)
    log_prob: Optional[float]  # Log-probability of the first answer token (if available)


class VLMBase(ABC):
    """Abstract base class for all VLM backends used in LogicQA."""

    @abstractmethod
    def query(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, Path, str]] = None,
    ) -> VLMResponse:
        """
        Send a text (+ optional image) prompt to the VLM.

        Args:
            prompt: Text prompt string.
            image:  Either a PIL Image, a file path, or None (text-only).

        Returns:
            VLMResponse with generated text, extracted answer, and log-prob.
        """

    # def _extract_answer(self, text: str) -> Optional[str]:
    #     """
    #     Heuristically extract 'Yes' or 'No' from a VLM response.
    #     Returns None if neither is found.
    #     """
    #     normalized = text.strip().lower()
    #     # Check for explicit tokens at start of response
    #     if normalized.startswith("yes"):
    #         return "Yes"
    #     if normalized.startswith("no"):
    #         return "No"
    #     # Fallback: search anywhere in the text
    #     if " yes" in normalized or "\nyes" in normalized:
    #         return "Yes"
    #     if " no" in normalized or "\nno" in normalized:
    #         return "No"
    #     return None

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract Yes/No from '- Result: Yes' / '- Result: No' marker."""
        lower = text.lower()
        if "- result: yes" in lower:
            return "Yes"
        if "- result: no" in lower:
            return "No"
        # Fallback для свободного текста
        if lower.strip().startswith("yes"):
            return "Yes"
        if lower.strip().startswith("no"):
            return "No"
        return None

