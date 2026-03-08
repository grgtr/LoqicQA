"""Gemini-1.5-Flash VLM backend via google-generativeai."""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from logicqa.vlm.base import VLMBase, VLMResponse
from logicqa.config import GeminiConfig


class GeminiBackend(VLMBase):
    """Gemini-1.5-Flash backend via google-generativeai SDK."""

    def __init__(self, cfg: GeminiConfig):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")

        self.cfg = cfg
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("Set GOOGLE_API_KEY environment variable.")

        genai.configure(api_key=api_key)
        self.generation_config = genai.types.GenerationConfig(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            max_output_tokens=512,
        )
        self.model = genai.GenerativeModel(cfg.model_name)

    def query(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, Path, str]] = None,
    ) -> VLMResponse:
        parts = []
        if image is not None:
            if isinstance(image, (str, Path)):
                pil_img = Image.open(str(image)).convert("RGB")
            else:
                pil_img = image.convert("RGB")
            parts.append(pil_img)
        parts.append(prompt)

        response = self.model.generate_content(
            parts,
            generation_config=self.generation_config,
        )
        generated_text = response.text or ""
        answer = self._extract_answer(generated_text)
        # Gemini SDK does not expose token-level log-probs in standard API;
        # we fall back to None. Use GPT-4o for accurate log-prob scoring.
        return VLMResponse(text=generated_text, answer=answer, log_prob=None)
