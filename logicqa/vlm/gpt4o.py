"""GPT-4o VLM backend via OpenAI API."""
from __future__ import annotations

import base64
import io
import math
import os
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from logicqa.vlm.base import VLMBase, VLMResponse
from logicqa.config import GPT4oConfig


def _encode_image_base64(image: Union[Image.Image, Path, str]) -> str:
    """Encode image as base64 string for OpenAI API."""
    if isinstance(image, (str, Path)):
        with open(str(image), "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class GPT4oBackend(VLMBase):
    """GPT-4o backend via OpenAI API with logprobs support."""

    def __init__(self, cfg: GPT4oConfig):
        try:
            import openai
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        self.cfg = cfg
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set OPENAI_API_KEY environment variable.")
        self.client = openai.OpenAI(api_key=api_key)

    def query(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, Path, str]] = None,
    ) -> VLMResponse:
        messages = []
        if image is not None:
            b64 = _encode_image_base64(image)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            })
        else:
            messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages,
            temperature=self.cfg.temperature,
            logprobs=True,
            top_logprobs=5,
            max_tokens=512,
        )

        choice = response.choices[0]
        generated_text = choice.message.content or ""
        answer = self._extract_answer(generated_text)

        # Extract log-prob of the first token
        log_prob = None
        if choice.logprobs and choice.logprobs.content:
            log_prob = choice.logprobs.content[0].logprob

        return VLMResponse(text=generated_text, answer=answer, log_prob=log_prob)
