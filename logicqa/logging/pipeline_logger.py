"""Pipeline logger — сохраняет все текстовые ответы модели по стадиям."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class PipelineLogger:
    """
    Логирует ответы модели на каждом этапе пайплайна в results/{run_dir}/.

    Структура директории:
        results/
          breakfast_box_20250309_143022/
            stage1_descriptions.json
            stage2_summary.json
            stage3_filtering.json
            stage3_subquestions.json
            stage4_responses.json
            pipeline.log
    """

    def __init__(self, output_dir: str | Path, class_name: str):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(os.path.expanduser(str(output_dir))) / f"{class_name}_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._log_path = self.run_dir / "pipeline.log"
        self._stage1: List[Dict] = []
        self._stage2: Optional[Dict] = None
        self._stage3_filter: List[Dict] = []
        self._stage3_subqs: Dict[str, List[str]] = {}
        self._stage4: List[Dict] = []

        self._log(f"=== PipelineLogger started: {class_name} ===")
        self._log(f"Run dir: {self.run_dir}")

    # ------------------------------------------------------------------ #
    # Внутренние утилиты
    # ------------------------------------------------------------------ #

    def _log(self, msg: str) -> None:
        """Дописать строку в pipeline.log."""
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _save_json(self, filename: str, data: Any) -> None:
        path = self.run_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #
    # Stage 1
    # ------------------------------------------------------------------ #

    def log_stage1_description(
        self,
        image_idx: int,
        image_path: str,
        prompt: str,
        response_text: str,
    ) -> None:
        entry = {
            "image_idx": image_idx,
            "image_path": image_path,
            "prompt": prompt,
            "response": response_text,
        }
        self._stage1.append(entry)
        self._save_json("stage1_descriptions.json", self._stage1)

        self._log(f"\n--- Stage 1 | Image {image_idx}: {image_path} ---")
        self._log(f"[PROMPT]\n{prompt}")
        self._log(f"[RESPONSE]\n{response_text}")

    # ------------------------------------------------------------------ #
    # Stage 2
    # ------------------------------------------------------------------ #

    def log_stage2_summary(
        self,
        prompt: str,
        response_text: str,
    ) -> None:
        self._stage2 = {
            "prompt": prompt,
            "response": response_text,
        }
        self._save_json("stage2_summary.json", self._stage2)

        self._log("\n--- Stage 2 | Summary ---")
        self._log(f"[PROMPT]\n{prompt}")
        self._log(f"[RESPONSE]\n{response_text}")

    # ------------------------------------------------------------------ #
    # Stage 3a — генерация вопросов
    # ------------------------------------------------------------------ #

    def log_stage3a_questions(
        self,
        prompt: str,
        response_text: str,
        parsed_questions: List[str],
    ) -> None:
        entry = {
            "prompt": prompt,
            "response": response_text,
            "parsed_questions": parsed_questions,
        }
        self._save_json("stage3a_questions.json", entry)

        self._log("\n--- Stage 3a | Generated questions ---")
        self._log(f"[PROMPT]\n{prompt}")
        self._log(f"[RESPONSE]\n{response_text}")
        self._log(f"[PARSED] {parsed_questions}")

    # ------------------------------------------------------------------ #
    # Stage 3b — фильтрация
    # ------------------------------------------------------------------ #

    def log_stage3b_filter_answer(
        self,
        question: str,
        image_idx: int,
        image_path: str,
        prompt: str,
        response_text: str,
        extracted_answer: Optional[str],
    ) -> None:
        entry = {
            "question": question,
            "image_idx": image_idx,
            "image_path": image_path,
            "prompt": prompt,
            "response": response_text,
            "extracted_answer": extracted_answer,
        }
        self._stage3_filter.append(entry)
        self._save_json("stage3b_filtering.json", self._stage3_filter)

        self._log(
            f"\n--- Stage 3b | Filter | Q: {question[:60]}... "
            f"| img {image_idx} ---"
        )
        self._log(f"[PROMPT]\n{prompt}")
        self._log(f"[RESPONSE]\n{response_text}")
        self._log(f"[EXTRACTED] {extracted_answer}")

    def log_stage3b_result(
        self,
        question: str,
        accuracy: float,
        kept: bool,
    ) -> None:
        self._log(
            f"[FILTER RESULT] '{question[:70]}' "
            f"acc={accuracy:.2f} → {'KEEP' if kept else 'DROP'}"
        )

    # ------------------------------------------------------------------ #
    # Stage 3c — генерация саб-вопросов
    # ------------------------------------------------------------------ #

    def log_stage3c_subquestions(
        self,
        main_question: str,
        prompt: str,
        response_text: str,
        sub_questions: List[str],
    ) -> None:
        self._stage3_subqs[main_question] = {
            "prompt": prompt,
            "response": response_text,
            "sub_questions": sub_questions,
        }
        self._save_json("stage3c_subquestions.json", self._stage3_subqs)

        self._log(f"\n--- Stage 3c | Sub-Qs for: {main_question[:70]} ---")
        self._log(f"[PROMPT]\n{prompt}")
        self._log(f"[RESPONSE]\n{response_text}")
        self._log(f"[SUB-QUESTIONS] {sub_questions}")

    # ------------------------------------------------------------------ #
    # Stage 4 — тестирование
    # ------------------------------------------------------------------ #

    def log_stage4_image_start(
        self,
        image_idx: int,
        image_path: str,
        gt_label: str,
    ) -> None:
        self._log(
            f"\n{'='*60}\n"
            f"Stage 4 | [{image_idx}] {image_path} | gt={gt_label}\n"
            f"{'='*60}"
        )

    def log_stage4_sub_question(
        self,
        image_path: str,
        main_question: str,
        sub_question: str,
        sub_q_idx: int,
        prompt: str,
        response_text: str,
        extracted_answer: Optional[str],
        log_prob: Optional[float],
        extraction_meta: Optional[Dict[str, Any]],
    ) -> None:
        entry = {
            "image_path": image_path,
            "main_question": main_question,
            "sub_question": sub_question,
            "sub_q_idx": sub_q_idx,
            "prompt": prompt,
            "response": response_text,
            "extracted_answer": extracted_answer,
            "log_prob": log_prob,
            "extraction_meta": extraction_meta,
        }
        self._stage4.append(entry)
        self._save_json("stage4_responses.json", self._stage4)

        self._log(
            f"\n  [Sub-Q {sub_q_idx}] MQ: {main_question[:50]}...\n"
            f"  SQ: {sub_question}"
        )
        self._log(f"  [PROMPT]\n{prompt}")
        self._log(f"  [RESPONSE]\n{response_text}")
        self._log(
            f"  [ANSWER] {extracted_answer} | log_prob={log_prob}"
        )
        self._log(
            f"  [META] {extraction_meta}"
        )

    def log_stage4_main_question_result(
        self,
        main_question: str,
        voted_answer: str,
        sub_answers: List[Optional[str]],
    ) -> None:
        yes_count = sum(1 for a in sub_answers if a == "Yes")
        no_count  = sum(1 for a in sub_answers if a == "No")
        self._log(
            f"  [VOTE] '{main_question[:60]}' → {voted_answer} "
            f"(Yes={yes_count}, No={no_count})"
        )

    def log_stage4_image_result(
        self,
        image_path: str,
        is_anomaly: bool,
        anomaly_score: float,
        explanation: str,

    ) -> None:
        self._log(
            f"  [RESULT] anomaly={is_anomaly} "
            f"score={anomaly_score:.4f}\n"
            f"  {explanation}"
        )
    def log(self, message: str) -> None:
        self._log(message)
        
