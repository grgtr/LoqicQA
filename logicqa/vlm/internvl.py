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

import importlib.util
import sys

import re

# --------------------------------------------------------------------------- #
# InternVL image pre-processing helpers (from official InternVL repo)
# --------------------------------------------------------------------------- #
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def _apply_chat_patch(model) -> None:
    original_chat = model.__class__.chat

    def patched_chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        **kwargs,
    ):
        want_scores = generation_config.pop("output_scores", False)
        generation_config.pop("return_dict_in_generate", False)

        if not want_scores:
            return original_chat(
                self, tokenizer, pixel_values, question,
                generation_config, history=history,
                return_history=return_history, **kwargs,
            )

        original_generate = self.generate
        captured = {}

        def capturing_generate(*args, **gen_kwargs):
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True
            outputs = original_generate(*args, **gen_kwargs)
            
            # Save scores and real IDs of generated tokens
            captured["scores"] = getattr(outputs, "scores", None)
            captured["sequences"] = getattr(outputs, "sequences", None)
            
            return outputs.sequences

        self.generate = capturing_generate
        try:
            result = original_chat(
                self, tokenizer, pixel_values, question,
                generation_config, history=history,
                return_history=return_history, **kwargs,
            )
        finally:
            self.generate = original_generate

        text = result[0] if isinstance(result, tuple) else result
        history_out = (
            result[1] if isinstance(result, tuple) and len(result) > 1 else None
        )
        return text, history_out, captured.get("scores"), captured.get("sequences")

    model.__class__.chat = patched_chat
    print("[Patch] chat() successfully patched — scores will be returned")


# def _load_patched_internvl():
#     """Load patched modeling_internvl_chat instead of cached."""
#     patch_path = Path(__file__).parent / "modeling_internvl_chat_patched.py"
#     if not patch_path.exists():
#         return  # patch not installed — work as usual

#     spec = importlib.util.spec_from_file_location(
#         "modeling_internvl_chat", str(patch_path)
#     )
#     module = importlib.util.module_from_spec(spec)
#     # Register under original name to intercept import
#     sys.modules["modeling_internvl_chat"] = module
#     spec.loader.exec_module(module)
#     print("[Patch] Loaded patched modeling_internvl_chat")

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
        # _load_patched_internvl()
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

        _apply_chat_patch(self.model)

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
        # log_prob = self._compute_log_prob_from_text(generated_text, answer)

        return VLMResponse(text=generated_text, answer=answer)

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
        pad_token_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 2
        # generation_config = dict(
        #     max_new_tokens=self.cfg.max_new_tokens,
        #     do_sample=self.cfg.do_sample,
        #     temperature=self.cfg.temperature,
        #     top_p=self.cfg.top_p,
        #     repetition_penalty=self.cfg.repetition_penalty,
        #     output_scores=True,
        #     return_dict_in_generate=True,
        #     pad_token_id=pad_token_id,
        # )
        generation_config = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            repetition_penalty=self.cfg.repetition_penalty,
            pad_token_id=pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

        if image is not None:
            pixel_values = _load_image(image).to(
                dtype=torch.bfloat16,
                device=next(self.model.parameters()).device,
            )
            conv_prompt = f"<image>\n{prompt}"
            result = self.model.chat(
                self.tokenizer, pixel_values, conv_prompt,
                generation_config, history=None, return_history=True,
            )
        else:
            result = self.model.chat(
                self.tokenizer, None, prompt,
                generation_config, history=None, return_history=True,
            )
        # print(f"[DEBUG] image is not None: {image is not None}, result_len={len(result)}")
        if isinstance(result, tuple) and len(result) >= 3:
            # generated_text, _, scores, sequences = result
            generated_text = result[0]
            scores = result[2]
            sequences = result[3] if len(result) >= 4 else None
            # print(f"[DEBUG] scores type={type(scores)}, "
            # f"len={len(scores) if scores is not None else 0}")
            # if scores:
            #     print(f"[DEBUG] scores[0].shape={scores[0].shape}") 
            # print(f"[DEBUG] scores={scores}")
            # print()
            # print(f"[DEBUG] sequences={sequences}")
        else:
            generated_text = result[0] if isinstance(result, tuple) else result
            scores = None
            sequences = None

        answer = self._extract_answer(generated_text)
        print(f"[DEBUG] extracted_answer={answer}")
        log_prob, meta = self._extract_answer_log_prob(generated_text, answer, scores, sequences)
    
        return VLMResponse(text=generated_text, answer=answer, log_prob=log_prob, extraction_meta=meta)

    # ------------------------------------------------------------------ #

    def _extract_answer_log_prob(
        self, text: str, answer: Optional[str], scores, sequences=None
    ) -> Optional[float]:
        if answer is None:
            return None, {}

        if scores is None or len(scores) == 0:
            return self._compute_log_prob_from_text(text, answer), {"fallback": True}

        text_lower = text.lower()
        
        # 1. Используем ТУ ЖЕ САМУЮ регулярку (СТРОГО БЕЗ [::-1])
        matches = list(re.finditer(r"result\s*[:=#*-]*\s*(yes|no)", text_lower))
        if not matches:
            matches = list(re.finditer(r"\b(yes|no)\b", text_lower))

        if not matches:
            return self._compute_log_prob_from_text(text, answer)

        # Берем последнее совпадение и находим реальный индекс символа "y" или "n"
        last_match = matches[-1]
        target_char_index = last_match.start(1)

        # 2. Получаем массив реальных сгенерированных токенов
        if sequences is not None:
            gen_len = len(scores)
            token_ids = sequences[0, -gen_len:].tolist()
        else:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            gen_len = len(token_ids)

        # 3. Ищем токен через прогрессивное декодирование (работает с любыми токенизаторами)
        answer_pos = None
        for i in range(gen_len):
            prefix_str = self.tokenizer.decode(
                token_ids[:i+1], 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            if len(prefix_str) > target_char_index:
                answer_pos = i
                break

        if answer_pos is None:
            return self._compute_log_prob_from_text(text, answer), {"fallback": True}

        # 4. Вытаскиваем вероятности
        raw_scores = scores[answer_pos]
        if raw_scores.dim() == 2:
            raw_scores = raw_scores[0]
            
        target_token_id = token_ids[answer_pos]

        log_probs = torch.nn.functional.log_softmax(raw_scores.float(), dim=-1)
        lp = log_probs[target_token_id].item()
        prob = float(torch.exp(torch.tensor(lp)))
        meta = {
            "regex_found": text[target_char_index:target_char_index+3],
            "pos": target_char_index,
            "mapped_to_token_id": target_token_id,
            "decoded_token": self.tokenizer.decode([target_token_id]),
            "step": answer_pos,
            "raw_prob": prob
        }
        # print(f"[DEBUG] Regex found '{text[target_char_index:target_char_index+3]}' at pos {target_char_index}")
        # print(f"[DEBUG] Mapped to token {target_token_id} ('{self.tokenizer.decode([target_token_id])}') at step {answer_pos}")
        # print(f"[DEBUG] log_prob={lp:.4f} (prob={torch.exp(torch.tensor(lp)):.4f})")
        return lp, meta

    # def _extract_answer_log_prob(
    #     self,
    #     text: str,
    #     answer: Optional[str],
    #     scores,   # tuple of (vocab_size,) tensors | None
    #     sequences=None
    # ) -> Optional[float]:
    #     """
    #     Extract log_prob of answer token ("Yes"/"No") from scores.

    #     scores[i] — logits for i-th generated token (before softmax).
    #     We look for the last occurrence of Yes/No in the generated sequence,
    #     and take log_softmax of the corresponding scores[i].
    #     """
    #     if answer is None:
    #         return None
        
    #     # Fallback if patch did not return scores
    #     if scores is None or len(scores) == 0:
    #         print("[DEBUG] SCORES IS NONE Fallback to _compute_log_prob_from_text")
    #         return self._compute_log_prob_from_text(text, answer)
        
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
    #     target_char_index = match.start()

    #     # 2. Получаем массив реальных сгенерированных токенов
    #     if sequences is not None:
    #         gen_len = len(scores)
    #         token_ids = sequences[0, -gen_len:].tolist()
    #     else:
    #         token_ids = self.tokenizer.encode(text, add_special_tokens=False)
    #         gen_len = len(token_ids)

    #     # 3. Ищем нужный токен через прогрессивное декодирование
    #     answer_pos = None
    #     for i in range(gen_len):
    #         # Декодируем токены от начала до i-го включительно
    #         # clean_up_tokenization_spaces=False гарантирует сохранение всех пробелов
    #         prefix_str = self.tokenizer.decode(
    #             token_ids[:i+1], 
    #             skip_special_tokens=True, 
    #             clean_up_tokenization_spaces=False
    #         )
            
    #         # Как только декодированный текст перекрыл начальную позицию нашего слова,
    #         # значит текущий токен (i) и содержит начало этого слова!
    #         if len(prefix_str) > target_char_index:
    #             answer_pos = i
    #             break

    #     if answer_pos is None:
    #         print("[DEBUG] Ошибка маппинга символа в токен (fallback)")
    #         return self._compute_log_prob_from_text(text, answer)

    #     # 4. Вытаскиваем вероятности из массива scores по найденному индексу
    #     raw_scores = scores[answer_pos]
    #     if raw_scores.dim() == 2:
    #         raw_scores = raw_scores[0]
            
    #     target_token_id = token_ids[answer_pos]

    #     log_probs = torch.nn.functional.log_softmax(raw_scores.float(), dim=-1)
    #     lp = log_probs[target_token_id].item()

    #     print(f"[DEBUG] Regex found '{text[target_char_index:target_char_index+3]}' at pos {target_char_index}")
    #     print(f"[DEBUG] Mapped to token {target_token_id} ('{self.tokenizer.decode([target_token_id])}') at step {answer_pos}")
    #     print(f"[DEBUG] log_prob={lp:.4f} (prob={torch.exp(torch.tensor(lp)):.4f})")

    #     return lp
        
        # def get_all_ids(word: str) -> set:
        #     ids = set()
        #     for variant in [word, f" {word}", word.lower(), f" {word.lower()}"]:
        #         encoded = self.tokenizer.encode(variant, add_special_tokens=False)
        #         if len(encoded) == 1:
        #             ids.add(encoded[0])
        #     return ids

        # yes_ids = get_all_ids("Yes")
        # no_ids  = get_all_ids("No")
        # print(f"[DEBUG] yes_ids={yes_ids}, no_ids={no_ids}")
        # target_ids = yes_ids if answer == "Yes" else no_ids
        # # Tokenize generated text (without special tokens)
        # # token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        # if sequences is not None:
        #     # sequences has shape (batch_size, seq_len)
        #     # The last len(scores) tokens are exactly the generated tokens
        #     gen_len = len(scores)
        #     token_ids = sequences[0, -gen_len:].tolist()
        # else:
        #     # Fallback: old logic if sequences are not found
        #     token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        # # ID tokens for "Yes" and "No"
        # # yes_ids = set(self.tokenizer.encode("Yes", add_special_tokens=False))
        # # no_ids  = set(self.tokenizer.encode("No",  add_special_tokens=False))
        # # target_ids = yes_ids if answer == "Yes" else no_ids

        # # Find the last occurrence of the answer token in the generated sequence
        # answer_pos = None
        # max_pos = min(len(token_ids), len(scores))
        # for i in range(max_pos - 1, -1, -1):
        #     if token_ids[i] in target_ids:
        #         answer_pos = i
        #         break

        # if answer_pos is None:
        #     print("[DEBUG] Answer token not found in generated text, Fallback to _compute_log_prob_from_text")
        #     print(f"[DEBUG] text tail: {repr(text[-100:])}")
        #     print(f"[DEBUG] token_ids tail: {token_ids[-20:]}")
        #     print(f"[DEBUG] target_ids: {target_ids}")
        #     return self._compute_log_prob_from_text(text, answer)

        # # scores[i] has shape (batch_size, vocab_size) or (vocab_size,)
        # raw_scores = scores[answer_pos]
        # if raw_scores.dim() == 2:
        #     raw_scores = raw_scores[0]  # remove batch dimension

        # log_probs = torch.nn.functional.log_softmax(raw_scores.float(), dim=-1)
        # lp = log_probs[token_ids[answer_pos]].item()
        # print(f"[DEBUG] answer_pos={answer_pos}, token_id={token_ids[answer_pos]}, decoded='{self.tokenizer.decode([token_ids[answer_pos]])}'")
        # print(f"[DEBUG] log_prob={lp:.4f} (prob={torch.exp(torch.tensor(lp)):.4f})")
        # return lp

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
