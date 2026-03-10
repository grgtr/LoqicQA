# logicqa/evaluation/level2_5_filtering.py
from typing import List, Dict
from logicqa.evaluation.llm_judge import LLMJudge

def evaluate_filtering(stage3a_questions: Dict, stage3b_filtering: List[Dict], gt_constraints: List[str], judge: LLMJudge) -> Dict:

    if not stage3a_questions or not gt_constraints:
        return {"Precision": 0.0, "Recall": 0.0}

    # 1. Get list of all generated questions (candidates)
    candidate_questions = stage3a_questions.get("parsed_questions", [])
    
    # 2. Determine which questions passed the filter (Kept) and which fell (Dropped)
    # If the question is in stage3b_filtering and accuracy >= threshold (you have 0.8), it passed.
    # But it's easier to take keys from stage3c_subqs (only those that passed the filter)
    # Since we only pass stage3a and 3b, let's restore the status:
    
    # Count how many times VLM answered Yes for each question
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
        if stats["total"] > 0 and (stats["yes"] / stats["total"]) >= 0.6:
            kept_questions.append(q)
        else:
            dropped_questions.append(q)
            
    if not q_stats:
        kept_questions = candidate_questions

    tp_questions = 0
    fp_questions = 0
    
    covered_constraints = set()

    for q in kept_questions:
        matched_constraints = judge.map_question_to_constraints(q, gt_constraints)
        if matched_constraints:
            tp_questions += 1
            covered_constraints.update(matched_constraints)
        else:
            fp_questions += 1

    precision = tp_questions / (tp_questions + fp_questions) if (tp_questions + fp_questions) > 0 else 0.0

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
