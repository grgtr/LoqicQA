import json
import re
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class HallucinationReport:
    impossible_objects: List[str] = field(default_factory=list)
    count_violations: List[Dict] = field(default_factory=list)
    spatial_violations: List[Dict] = field(default_factory=list)
    absent_required: List[str] = field(default_factory=list)
    self_contradictions: List[str] = field(default_factory=list)
    overall_confidence: float = 0.0
    is_hallucinated: bool = False

    def severity(self) -> str:
        total = (
            len(self.impossible_objects)
            + len(self.count_violations)
            + len(self.spatial_violations)
            + len(self.absent_required)
            + len(self.self_contradictions)
        )
        if total == 0:
            return "none"
        if total == 1:
            return "low"
        if total <= 3:
            return "medium"
        return "high"

class LLMJudge:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "cuda"):
        print(f"[LLMJudge] Loading {model_id} on {device}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device
        )
        self.model.eval()

    def _generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response

    def _parse_json(self, text: str) -> dict:
        """Parse a JSON object from model output. Always returns a dict."""
        # Code block
        try:
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                result = json.loads(match.group(1))
                if isinstance(result, dict):
                    return result
        except Exception:
            pass
        # Direct parse
        try:
            result = json.loads(text.strip())
            if isinstance(result, dict):
                return result
        except Exception:
            pass
        # Strip preamble: find first { and try from there
        start = text.find('{')
        if start >= 0:
            try:
                result = json.loads(text[start:])
                if isinstance(result, dict):
                    return result
            except Exception:
                pass
        print(f"[LLMJudge] Failed to parse JSON from: {text[:200]}...")
        return {}

    def _parse_list(self, text: str) -> list:
        """Parse a JSON array from model output. Always returns a list."""
        # Code block
        try:
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                result = json.loads(match.group(1))
                if isinstance(result, list):
                    return result
        except Exception:
            pass
        # Direct parse
        try:
            result = json.loads(text.strip())
            if isinstance(result, list):
                return result
        except Exception:
            pass
        # Strip preamble: find first [ and try from there
        start = text.find('[')
        if start >= 0:
            try:
                result = json.loads(text[start:])
                if isinstance(result, list):
                    return result
            except Exception:
                pass
        print(f"[LLMJudge] Failed to parse JSON list from: {text[:200]}...")
        return []


    def evaluate_ccr(self, description: str, constraints: List[str]) -> Dict[str, bool]:
        # Ask for a list of covered constraints only — much shorter output than full dict.
        system_prompt = (
            "You are an objective evaluator. Given a visual description and a list of constraints, "
            "output ONLY a valid JSON list containing the EXACT strings of constraints that are "
            "explicitly mentioned or clearly implied in the description. "
            "Copy strings exactly as given. Output [] if none are covered. No other text."
        )
        user_prompt = f"Constraints:\n{json.dumps(constraints, indent=2)}\n\nDescription:\n{description}"

        response = self._generate(system_prompt, user_prompt, max_new_tokens=1024)
        covered = self._parse_list(response)
        covered_set = set(covered)
        return {c: (c in covered_set) for c in constraints}

    def extract_count(self, text: str, object_name: str) -> Optional[int]:
        # system_prompt = (
        #     "Extract the exact quantity of the specified object mentioned in the text. "
        #     "Output ONLY a valid JSON dictionary with a single key 'count' and an integer value. "
        #     "If the count is not mentioned, return null for the value."
        # )
        # user_prompt = f"Object to count: {object_name}\n\nText:\n{text}"
        
        # response = self._generate(system_prompt, user_prompt)
        # data = self._parse_json(response)
        # return data.get("count")
        system_prompt = (
            "You are a strict data extractor. Read the text and find the exact numerical "
            f"count mentioned for the object: '{object_name}'. "
            "Output ONLY a valid JSON dictionary with a single key 'count' and an integer value. "
            "If no number is mentioned for this object, use null. Examples of numbers: 'two' -> 2, '0' -> 0. "
            "Do not output markdown formatting, just raw JSON."
        )
        user_prompt = f"Text:\n{text}"
        
        response = self._generate(system_prompt, user_prompt)
        data = self._parse_json(response)
        
        count = data.get("count")
        if isinstance(count, str) and count.isdigit():
            return int(count)
        if isinstance(count, int):
            return count
        return None

    def check_spatial_violation(self, text: str, object_name: str, expected_relation: str) -> bool:
        system_prompt = (
            "Determine if the text explicitly states that the spatial relation or position "
            "of the object violates expectations or is abnormal. "
            "Output ONLY a valid JSON dictionary with a single key 'violation_detected' (boolean)."
        )
        user_prompt = f"Object: {object_name}\nExpected relation: {expected_relation}\n\nText:\n{text}"
        
        response = self._generate(system_prompt, user_prompt)
        data = self._parse_json(response)
        return data.get("violation_detected", False)

    def map_question_to_constraints(self, question: str, constraints: List[str]) -> List[str]:

        system_prompt = (
            "You are a strict logical evaluator. Given a question and a list of official constraints, "
            "determine which constraints this question is trying to test. "
            "A question might test 0, 1, or multiple constraints.\n"
            "Output ONLY a valid JSON list containing the exact strings of the matched constraints. "
            "If the question is irrelevant (tests none of the constraints), output an empty list []."
        )
        user_prompt = f"Constraints:\n{json.dumps(constraints, indent=2)}\n\nQuestion:\n{question}"

        response = self._generate(system_prompt, user_prompt)
        matched = self._parse_list(response)
        return [c for c in matched if c in constraints]

    def detect_hallucinations(
        self,
        description: str,
        normality_definition: str,
        class_name: str,
        impossible_objects: Optional[List[str]] = None,
    ) -> HallucinationReport:
        """
        Detect diverse hallucination types in a VLM-generated description.

        Covers 5 categories via a single LLM call:
          - impossible_objects: objects that cannot appear in this class
          - count_violations: quantities contradicting the normality definition
          - spatial_violations: positions/locations contradicting the definition
          - absent_required: required objects not mentioned in the description
          - self_contradictions: internally inconsistent claims
        """
        impossible_str = json.dumps(impossible_objects or [], ensure_ascii=False)
        system_prompt = (
            f"You are a hallucination detector for visual descriptions of a '{class_name}'.\n"
            "Given a normality definition (ground truth) and a generated description, "
            "identify ALL inconsistencies in exactly these 5 categories:\n"
            f"- \"impossible_objects\": objects mentioned that CANNOT exist in a {class_name} "
            f"(forbidden list: {impossible_str})\n"
            "- \"count_violations\": list of {{\"mentioned\": \"...\", \"expected\": \"...\"}} "
            "where quantities contradict the normality definition\n"
            "- \"spatial_violations\": list of {{\"object\": \"...\", \"mentioned_position\": \"...\", "
            "\"expected\": \"...\"}} where positions contradict the normality definition\n"
            "- \"absent_required\": objects required by the normality definition but NOT mentioned\n"
            "- \"self_contradictions\": internally inconsistent claims within the description\n"
            "Also add:\n"
            "- \"overall_confidence\": float 0-1 (how confident you are in your findings)\n"
            "- \"is_hallucinated\": true if ANY category is non-empty, else false\n\n"
            "Output ONLY valid JSON. Use empty lists [] if no issues found in a category."
        )
        user_prompt = (
            f"Normality definition:\n{normality_definition}\n\n"
            f"Description:\n{description}"
        )

        response = self._generate(system_prompt, user_prompt)
        data = self._parse_json(response)

        impossible = data.get("impossible_objects", [])
        counts = data.get("count_violations", [])
        spatial = data.get("spatial_violations", [])
        absent = data.get("absent_required", [])
        contradictions = data.get("self_contradictions", [])
        confidence = float(data.get("overall_confidence", 0.0))

        is_hallucinated = bool(
            impossible or counts or spatial or absent or contradictions
        )
        # Honour explicit LLM override if no issues found by categories
        if not is_hallucinated:
            is_hallucinated = bool(data.get("is_hallucinated", False))

        return HallucinationReport(
            impossible_objects=impossible if isinstance(impossible, list) else [],
            count_violations=counts if isinstance(counts, list) else [],
            spatial_violations=spatial if isinstance(spatial, list) else [],
            absent_required=absent if isinstance(absent, list) else [],
            self_contradictions=contradictions if isinstance(contradictions, list) else [],
            overall_confidence=confidence,
            is_hallucinated=is_hallucinated,
        )

