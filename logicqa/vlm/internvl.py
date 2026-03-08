"""InternVL-2.5 local VLM backend."""
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from logicqa.vlm.base import VLMBase, VLMResponse
from logicqa.config import InternVLConfig

# --------------------------------------------------------------------------- #
# InternVL image pre-processing helpers (from official InternVL repo)
# --------------------------------------------------------------------------- #
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def _split_model(model_name: str) -> dict:
    """
    Returns device_map for even distribution of layers across GPUs.
    Used instead of device_map='auto' to avoid meta tensor issue.
    """
    import transformers
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True
    )
    num_layers = config.llm_config.num_hidden_layers

    n_gpus = torch.cuda.device_count()
    assert n_gpus > 0, "No available GPUs"

    # vision model + embedding on GPU 0
    # lm_head + norm on last GPU
    # LLM layers are divided equally
    layers_per_gpu = math.ceil(num_layers / n_gpus)
    device_map = {"vision_model": 0, "mlp1": 0, "language_model.model.embed_tokens": 0}

    for i in range(num_layers):
        gpu_id = min(i // layers_per_gpu, n_gpus - 1)
        device_map[f"language_model.model.layers.{i}"] = gpu_id

    last_gpu = n_gpus - 1
    device_map["language_model.model.norm"] = last_gpu
    device_map["language_model.lm_head"] = last_gpu

    return device_map

def _build_transform(input_size: int = 448):
    """Build the transform used by InternVL."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    """
    Dynamic resolution tiling following InternVL's preprocessing convention.
    Splits high-res images into tiles to stay within patch budget.
    """
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h

    # Determine best (rows, cols) tile arrangement
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1)
         for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    best_ratio = min(
        target_ratios,
        key=lambda r: abs(r[0] / r[1] - aspect_ratio)
        if abs(r[0] / r[1] - aspect_ratio) > 0 else float("inf"),
    )
    target_w = image_size * best_ratio[1]
    target_h = image_size * best_ratio[0]
    resized = image.resize((target_w, target_h), Image.BICUBIC)

    tiles = []
    rows, cols = best_ratio
    for r in range(rows):
        for c in range(cols):
            box = (c * image_size, r * image_size,
                   (c + 1) * image_size, (r + 1) * image_size)
            tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size), Image.BICUBIC))
    return tiles


def _load_image(
    image_input: Union[Image.Image, Path, str],
    max_num: int = 12,
    image_size: int = 448,
):
    """Load image from PIL, path, or string path and return pixel_values tensor."""
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(str(image_input)).convert("RGB")
    else:
        pil_img = image_input.convert("RGB")

    transform = _build_transform(image_size)
    tiles = _dynamic_preprocess(pil_img, max_num=max_num, image_size=image_size)
    pixel_values = torch.stack([transform(t) for t in tiles])  # (N, 3, H, W)
    return pixel_values


# --------------------------------------------------------------------------- #
# InternVL-2.5 Backend
# --------------------------------------------------------------------------- #

class InternVLBackend(VLMBase):
    """
    Local InternVL-2.5 VLM backend.

    Supports multi-GPU inference via ``device_map="auto"`` (accelerate).
    Set ``CUDA_VISIBLE_DEVICES`` to control which GPUs are used.

    Log-probability is computed as the mean log-prob of the first generated
    token ('Yes'/'No'), using scores returned by ``model.generate``.
    """

    def __init__(self, cfg: InternVLConfig):
        self.cfg = cfg
        print(f"[InternVL] Loading model: {cfg.model_name}")
        n_gpus = torch.cuda.device_count()
        print("Number of GPUs:", n_gpus)
        assert n_gpus > 0

        _orig_linspace = torch.linspace
        def _safe_linspace(*args, **kwargs):
            kwargs['device'] = 'cpu'
            return _orig_linspace(*args, **kwargs)
        torch.linspace = _safe_linspace

        try:
            self.model = AutoModel.from_pretrained(
                cfg.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
                use_flash_attn=False,
                trust_remote_code=True,
            ).eval()
        finally:
            torch.linspace = _orig_linspace
        if not hasattr(self.model, 'all_tied_weights_keys'):
            tied = getattr(self.model, '_tied_weights_keys', []) or []
            self.model.all_tied_weights_keys = {k: None for k in tied}

        self.model = self.model.eval()
        if n_gpus == 1:
            self.model = self.model.cuda()
        else:
            from accelerate import dispatch_model
            device_map = _split_model(cfg.model_name)
            self.model = dispatch_model(self.model, device_map=device_map)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        print(f"[InternVL] Model loaded. GPUs used: {n_gpus}")

    # ------------------------------------------------------------------ #

    def query(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, Path, str]] = None,
    ) -> VLMResponse:
        """
        Query InternVL-2.5.

        Args:
            prompt: The text prompt (question or instruction).
            image:  Optional image (PIL, path, or str).

        Returns:
            VLMResponse with answer text, binary answer, and log-prob.
        """
        generation_config = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            repetition_penalty=self.cfg.repetition_penalty,
        )

        if image is not None:
            pixel_values = _load_image(image).to(
                dtype=torch.bfloat16,
                device=next(self.model.parameters()).device,
            )
            # InternVL uses <image> token in the conversation
            conv_prompt = f"<image>\n{prompt}"
            response_obj = self.model.chat(
                self.tokenizer,
                pixel_values,
                conv_prompt,
                generation_config,
                history=None,
                return_history=False,
            )
        else:
            pixel_values = None
            response_obj = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config,
                history=None,
                return_history=False,
            )

        # InternVL.chat with return_dict_in_generate may return (text, history)
        # or a GenerateOutput depending on version; handle both.
        if isinstance(response_obj, tuple):
            generated_text = response_obj[0]
        else:
            generated_text = response_obj

        answer = self._extract_answer(generated_text)
        log_prob = self._compute_log_prob_from_text(generated_text, answer)

        return VLMResponse(text=generated_text, answer=answer, log_prob=log_prob)

    # ------------------------------------------------------------------ #

    def query_with_logprobs(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, Path, str]] = None,
    ) -> VLMResponse:
        """
        Query InternVL using model.generate directly to capture token log-probs.
        This is more accurate than _compute_log_prob_from_text but slower.
        """
        generation_config = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            repetition_penalty=self.cfg.repetition_penalty,
            output_scores=True,
            return_dict_in_generate=True,
        )

        if image is not None:
            pixel_values = _load_image(image).to(
                dtype=torch.bfloat16,
                device=next(self.model.parameters()).device,
            )
        else:
            pixel_values = None

        # Build input_ids manually
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            next(self.model.parameters()).device
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                pixel_values=pixel_values,
                **generation_config,
            )

        # Decode
        seq = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(seq, skip_special_tokens=True)
        answer = self._extract_answer(generated_text)

        # Extract log-prob of the first answer token
        log_prob = None
        if outputs.scores:
            first_token_scores = outputs.scores[0][0]  # (vocab_size,)
            first_token_log_probs = torch.nn.functional.log_softmax(
                first_token_scores.float(), dim=-1
            )
            first_token_id = seq[0].item()
            log_prob = first_token_log_probs[first_token_id].item()

        return VLMResponse(text=generated_text, answer=answer, log_prob=log_prob)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_log_prob_from_text(text: str, answer: Optional[str]) -> Optional[float]:
        """
        Fallback: estimate log-prob from answer confidence heuristics.
        Returns None when answer extraction fails.
        """
        if answer is None:
            return None
        # If the answer token appears very early → high confidence (closer to 0)
        normalized = text.strip().lower()
        if normalized.startswith("yes") or normalized.startswith("no"):
            return -0.05  # high confidence placeholder
        return -1.0      # lower confidence when answer buried in text
