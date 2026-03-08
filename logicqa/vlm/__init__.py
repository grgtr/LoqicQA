"""VLM package — abstract interface + concrete backends."""
from logicqa.vlm.base import VLMBase, VLMResponse
from logicqa.vlm.factory import get_vlm

__all__ = ["VLMBase", "VLMResponse", "get_vlm"]
