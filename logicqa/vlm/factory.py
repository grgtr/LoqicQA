"""Factory function to instantiate VLM backends."""
from __future__ import annotations

from logicqa.vlm.base import VLMBase
from logicqa.config import LogicQAConfig


def get_vlm(cfg: LogicQAConfig) -> VLMBase:
    """
    Instantiate and return the VLM backend specified in the config.

    Args:
        cfg: Full LogicQAConfig object.

    Returns:
        A concrete VLMBase subclass instance.
    """
    backend = cfg.vlm.backend.lower()

    if backend == "internvl":
        from logicqa.vlm.internvl import InternVLBackend
        return InternVLBackend(cfg.vlm.internvl)

    elif backend == "gpt4o":
        from logicqa.vlm.gpt4o import GPT4oBackend
        return GPT4oBackend(cfg.vlm.gpt4o)

    elif backend in ("gemini", "gemini-flash", "gemini1.5"):
        from logicqa.vlm.gemini import GeminiBackend
        return GeminiBackend(cfg.vlm.gemini)

    else:
        raise ValueError(
            f"Unknown VLM backend: '{backend}'. "
            f"Choose from: 'internvl', 'gpt4o', 'gemini'."
        )
