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

    # def _extract_answer(self, text: str) -> Optional[str]:
    #     """Extract Yes/No from '- Result: Yes' / '- Result: No' marker."""
    #     lower = text.strip().lower()

    #     yes_patterns = [
    #         "- result: yes",
    #         "result: yes",
    #         "**result: yes**",
    #         "result:yes",
    #     ]
    #     no_patterns = [
    #         "- result: no",
    #         "result: no",
    #         "**result: no**",
    #         "result:no",
    #     ]
    #     for pattern in yes_patterns:
    #         if pattern in lower:
    #             return "Yes"
    #     for pattern in no_patterns:
    #         if pattern in lower:
    #             return "No"
    #     first = lower.strip().split()[0] if lower.strip() else ""
    #     if first == "yes":
    #         return "Yes"
    #     if first == "no":
    #         return "No"
    #     match = re.search(r'\b(sey|on)\b', lower[::-1])
    #     if match:
    #         word = match.group(1)
    #         return "Yes" if word == "yes" else "No"
    #     return None

    # def _extract_answer(self, text: str) -> Optional[str]:
    #     """
    #     Extract 'Yes' or 'No' from the VLM output.
    #     Handles Markdown, variable spacing, and newlines 
    #     (e.g., 'Result: Yes', '## Result ##\nYes', 'Result = No').
    #     Searches for the last occurrence to handle Chain-of-Thought reasoning.
    #     """
    #     text_lower = text.lower()
        
    #     # 1. Ищем паттерны: слово "result", затем любые пробелы, переносы строк, 
    #     # символы маркдауна/пунктуации (:, =, #, *, -) и ответ (yes или no)
    #     # \s* захватывает пробелы и \n. [:=#*-]* захватывает разделители.
    #     pattern = r"result\s*[:=#*-]*\s*(yes|no)"
    #     matches = list(re.finditer(pattern, text_lower))
        
    #     if matches:
    #         # Берем самое последнее совпадение (финальный вывод)
    #         last_match = matches[-1]
    #         extracted_word = last_match.group(1) # 'yes' или 'no'
    #         print("[DEBUG] _extract_answer() matches not empty")
    #         return "Yes" if extracted_word == "yes" else "No"
            
    #     # 2. Фолбэк: если слова "Result" нет, просто ищем последнее "yes" или "no"
    #     # (Это исправляет баг с перевернутой строкой lower[::-1])
    #     matches = list(re.finditer(r"\b(yes|no)\b", text_lower))
    #     if matches:
    #         last_match = matches[-1]
    #         extracted_word = last_match.group(1)
    #         print("[DEBUG] _extract_answer() no Result word")
    #         return "Yes" if extracted_word == "yes" else "No"
            
    #     return None

    # def _extract_answer(self, text: str) -> Optional[str]:
    #     """
    #     Extract answer from text, looking for the last occurrence of 'result'
    #     and then the first word 'yes' or 'no' after it.
    #     Returns the found word in the original case or None.
    #     """
    #     if not isinstance(text, str) or not text:
    #         return None

    #     # 1. Find the position of the last occurrence of 'result' (case-insensitive)
    #     lower_text = text.lower()
    #     pos = lower_text.rfind('result')
    #     if pos == -1:
    #         return None

    #     # 2. Take the tail of the original text from the found position to the end
    #     tail = text[pos:]

    #     # 3. Search for the word yes/no in this tail (case-insensitive, with word boundaries)
    #     match = re.search(r'\b(yes|no)\b', tail, re.IGNORECASE)
    #     if match:
    #         return "Yes" if match.group(0).lower() == "yes" else "No"
    #     return None

    def _extract_answer(self, text: str) -> Optional[str]:
        import re
        text_lower = text.lower()
        
        # Сначала ищем жесткий паттерн (Result: Yes / No) с любыми разделителями
        pattern = r"result\s*[:=#*-]*\s*(yes|no)"
        matches = list(re.finditer(pattern, text_lower))
        
        if matches:
            # Берем самое последнее совпадение в тексте
            return "Yes" if matches[-1].group(1) == "yes" else "No"
            
        # Если слова Result нет, просто ищем последнее yes или no
        matches = list(re.finditer(r"\b(yes|no)\b", text_lower))
        if matches:
            return "Yes" if matches[-1].group(1) == "yes" else "No"
            
        return None
