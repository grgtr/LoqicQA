import json
import re
import torch
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
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
                max_new_tokens=256,
                do_sample=False
            )
        
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response

    # def _parse_json(self, text: str) -> dict:
    #     try:
    #         match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    #         if match:
    #             return json.loads(match.group(1))
    #         return json.loads(text)
    #     except Exception as e:
    #         print(f"[LLMJudge] Failed to parse JSON: {e}\nText: {text}")
    #         return {}
    def _parse_json(self, text: str) -> dict:
        try:
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return json.loads(text)
        except Exception:
            try:
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
            except Exception:
                pass
            print(f"[LLMJudge] Failed to parse JSON from: {text[:200]}...")
            return {}


    def evaluate_ccr(self, description: str, constraints: List[str]) -> Dict[str, bool]:
        system_prompt = (
            "You are an objective evaluator. You will be given a visual description of an object "
            "and a list of constraints. Determine if each constraint is explicitly mentioned "
            "or clearly implied in the description. Output ONLY a valid JSON dictionary where "
            "keys are the exact constraints and values are true or false."
        )
        user_prompt = f"Constraints:\n{json.dumps(constraints, indent=2)}\n\nDescription:\n{description}"
        
        response = self._generate(system_prompt, user_prompt)
        return self._parse_json(response)

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
        """
        Уровень 2.5: Сопоставляет сгенерированный вопрос с реальными правилами.
        Возвращает список правил, которые этот вопрос проверяет.
        """
        system_prompt = (
            "You are a strict logical evaluator. Given a question and a list of official constraints, "
            "determine which constraints this question is trying to test. "
            "A question might test 0, 1, or multiple constraints.\n"
            "Output ONLY a valid JSON list containing the exact strings of the matched constraints. "
            "If the question is irrelevant (tests none of the constraints), output an empty list []."
        )
        user_prompt = f"Constraints:\n{json.dumps(constraints, indent=2)}\n\nQuestion:\n{question}"
        
        response = self._generate(system_prompt, user_prompt)
        data = self._parse_json(response)
        
        if isinstance(data, list):
            # Фильтруем, чтобы вернуть только точные совпадения из GT
            return [c for c in data if c in constraints]
        return []

