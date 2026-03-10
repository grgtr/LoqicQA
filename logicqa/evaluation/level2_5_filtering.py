# logicqa/evaluation/level2_5_filtering.py
from typing import List, Dict
from logicqa.evaluation.llm_judge import LLMJudge

def evaluate_filtering(stage3a_questions: Dict, stage3b_filtering: List[Dict], gt_constraints: List[str], judge: LLMJudge) -> Dict:
    """
    Оценивает качество фильтрации вопросов (Stage 3).
    """
    if not stage3a_questions or not gt_constraints:
        return {"Precision": 0.0, "Recall": 0.0}

    # 1. Получаем список всех сгенерированных вопросов (кандидатов)
    candidate_questions = stage3a_questions.get("parsed_questions", [])
    
    # 2. Выясняем, какие вопросы прошли фильтр (Kept), а какие упали (Dropped)
    # Если вопрос есть в stage3b_filtering и accuracy >= threshold (у вас это 0.8), он прошел.
    # Но проще взять ключи из stage3c_subqs (там лежат только те, что прошли фильтр)
    # Так как мы передаем только stage3a и 3b, давайте восстановим статус:
    
    # Считаем, сколько раз VLM ответила Yes для каждого вопроса
    q_stats = {q: {"yes": 0, "total": 0} for q in candidate_questions}
    for log in stage3b_filtering:
        q = log.get("question")
        ans = log.get("extracted_answer")
        if q in q_stats:
            q_stats[q]["total"] += 1
            if ans == "Yes":
                q_stats[q]["yes"] += 1
                
    kept_questions = []
    dropped_questions = []
    
    for q, stats in q_stats.items():
        if stats["total"] > 0 and (stats["yes"] / stats["total"]) >= 0.8: # Threshold из статьи
            kept_questions.append(q)
        else:
            dropped_questions.append(q)
            
    # Если логов фильтрации нет, считаем, что все кандидаты сохранились
    if not q_stats:
        kept_questions = candidate_questions

    # 3. Вычисляем Precision (Отсутствие мусора среди Kept)
    tp_questions = 0
    fp_questions = 0
    
    # Храним, какие правила были покрыты
    covered_constraints = set()

    for q in kept_questions:
        matched_constraints = judge.map_question_to_constraints(q, gt_constraints)
        if matched_constraints:
            tp_questions += 1
            covered_constraints.update(matched_constraints)
        else:
            fp_questions += 1 # Вопрос сохранен, но он мусорный (не бьется с ГОСТом)

    precision = tp_questions / (tp_questions + fp_questions) if (tp_questions + fp_questions) > 0 else 0.0

    # 4. Вычисляем Recall (Какую долю ГОСТа мы покрыли итоговым списком)
    # Если констрейнт покрыт - это TP. Если не покрыт - FN.
    recall = len(covered_constraints) / len(gt_constraints) if gt_constraints else 0.0
    print({
        "Filter_Precision": precision,
        "Filter_Recall": recall,
        "Total_Candidates": len(candidate_questions),
        "Kept": len(kept_questions),
        "Dropped": len(dropped_questions)
    }
    )
    return {
        "Filter_Precision": precision,
        "Filter_Recall": recall,
        "Total_Candidates": len(candidate_questions),
        "Kept": len(kept_questions),
        "Dropped": len(dropped_questions)
    }
