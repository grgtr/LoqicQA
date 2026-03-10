# logicqa/evaluation/level2_attributes.py
from typing import List, Dict, Any
from logicqa.evaluation.llm_judge import LLMJudge

def is_relevant_question(question: str, gt_object: str) -> bool:
    """
    Простая эвристика: связан ли вопрос с объектом, в котором кроется аномалия?
    """
    q_lower = question.lower()
    obj_words = gt_object.lower().split()
    
    # Ищем пересечения ключевых слов (длиной > 3 символов, чтобы исключить предлоги)
    for word in obj_words:
        if len(word) > 3 and word in q_lower:
            return True
            
    # Синонимы для специфичных случаев
    synonyms = {
        "compartments": ["side", "placed", "located", "where"],
        "cereals and toppings": ["mix", "granola", "cereals", "almonds", "bananas"],
        "fruits (nectarines and tangerines)": ["fruit", "nectarine", "tangerine"]
    }
    
    for word in synonyms.get(gt_object, []):
        if word in q_lower:
            return True
            
    return False

def evaluate_attributes(stage4_responses: List[Dict], attribute_gt: Dict[str, Dict[str, Any]], class_name: str, judge: LLMJudge) -> Dict:
    class_gt = attribute_gt.get(class_name, {})
    
    count_errors = []
    spatial_correct = 0
    spatial_total = 0

    for resp in stage4_responses:
        anomaly_type = resp.get("anomaly_type")
        vlm_text = resp.get("response", "")
        question = resp.get("sub_question", "")
        extracted_answer = resp.get("extracted_answer", "")
        
        # 1. ОБРАБОТКА АНОМАЛЬНЫХ ИЗОБРАЖЕНИЙ (Sensitivity / MACE)
        if anomaly_type and anomaly_type in class_gt:
            gt_info = class_gt[anomaly_type]
            
            # Пропускаем ответы, если вопрос вообще не про тот объект, который сломан
            if not is_relevant_question(question, gt_info["object"]):
                continue

            if gt_info["type"] == "count":
                # Уровень 2: Абсолютный подсчет
                predicted_count = judge.extract_count(vlm_text, gt_info["object"])
                if predicted_count is not None:
                    actual_count = gt_info["actual"]
                    error = abs(predicted_count - actual_count)
                    count_errors.append(error)
                    
            elif gt_info["type"] in ["spatial", "condition"]:
                # Для пространственных аномалий нам достаточно знать:
                # 1) Вопрос был релевантным (прошли фильтр выше).
                # 2) VLM ответила "No" (зафиксировала нарушение).
                # Это надежнее, чем просить LLM парсить текстовый лог.
                spatial_total += 1
                if extracted_answer == "No":
                    spatial_correct += 1
                    
        # 2. ОБРАБОТКА НОРМАЛЬНЫХ ИЗОБРАЖЕНИЙ (Specificity)
        # Если картинка нормальная, VLM не должна была найти нарушений атрибутов
        elif anomaly_type is None or anomaly_type == "good":
            # Если VLM ответила "No" на нормальной картинке - это ложная тревога атрибутивного уровня
            if "count" in question.lower() or "exactly" in question.lower():
                # Это ошибка MACE = 1 (модель ошиблась на ровно 1 объект)
                if extracted_answer == "No":
                    count_errors.append(1) 
                else:
                    count_errors.append(0)
                    
            if "located" in question.lower() or "placed" in question.lower() or "side" in question.lower():
                spatial_total += 1
                if extracted_answer == "Yes":
                    spatial_correct += 1 # Модель верно подтвердила, что позиция нормальная

    mace = sum(count_errors) / len(count_errors) if count_errors else None
    spatial_acc = spatial_correct / spatial_total if spatial_total > 0 else None
    print({
        "MACE": mace,
        "count_samples_evaluated": len(count_errors),
        "spatial_relation_accuracy": spatial_acc,
        "spatial_samples_evaluated": spatial_total
    })
    return {
        "MACE": mace,
        "count_samples_evaluated": len(count_errors),
        "spatial_relation_accuracy": spatial_acc,
        "spatial_samples_evaluated": spatial_total
    }

# from typing import List, Dict, Any
# from logicqa.evaluation.llm_judge import LLMJudge

# def evaluate_attributes(stage4_responses: List[Dict], attribute_gt: Dict[str, Dict[str, Any]], class_name: str, judge: LLMJudge) -> Dict:
#     """
#     Analyzes model responses for correct counting (MACE) and spatial positioning.
#     """
#     class_gt = attribute_gt.get(class_name, {})
#     # print("[DEBUG] class_gt: ", class_gt)
#     count_errors = []
#     spatial_correct = 0
#     spatial_total = 0
#     print("[DEBUG len(stage4_responses):]", len(stage4_responses))
#     # print("[DEBUG stage4_responses:]", stage4_responses)
#     count_continue = 0
#     skipped_anomaly_type = set()
#     count_anomaly_type_none = 0
#     for resp in stage4_responses:
#         # Get anomaly type from meta added in Stage 2
#         meta = resp.get("extraction_meta", {})
#         anomaly_type = resp.get("anomaly_type") # Should be passed to logger
#         if not anomaly_type:
#             count_anomaly_type_none += 1
#         if not anomaly_type or anomaly_type not in class_gt:
#             # print("[DEBUG] anomaly_type not in class_gt")
#             # print("[DEBUG] anomaly_type: ", anomaly_type)
#             count_continue += 1
#             skipped_anomaly_type.add(anomaly_type)
#             continue
            
#         gt_info = class_gt[anomaly_type]
#         vlm_text = resp.get("response", "") # Full text response from model before extracting Yes/No
#         # print("[DEBUG] gt_info")
#         # print(gt_info)
#         # print("[DEBUG] vlm_text")
#         # print(vlm_text)
#         if gt_info["type"] == "count":
#             # Level 2: Absolute counting
#             predicted_count = judge.extract_count(vlm_text, gt_info["object"])
#             if predicted_count is not None:
#                 actual_count = gt_info["actual"]
#                 error = abs(predicted_count - actual_count)
#                 count_errors.append(error)
                
#         elif gt_info["type"] == "spatial":
#             # Level 2: Spatial relations
#             detected = judge.check_spatial_violation(vlm_text, gt_info["object"], gt_info["expected_relation"])
#             spatial_total += 1
#             if detected == gt_info.get("violated", True):
#                 spatial_correct += 1
#     print("[DEBUG count_anomaly_type_none]:", count_anomaly_type_none)
#     print("[DEBUG skipped_anomaly_type]:", skipped_anomaly_type)
#     print("[DEBUG count_continue]:", count_continue)
#     mace = sum(count_errors) / len(count_errors) if count_errors else None
#     spatial_acc = spatial_correct / spatial_total if spatial_total > 0 else None
#     print({
#         "MACE": mace,
#         "count_samples_evaluated": len(count_errors),
#         "spatial_relation_accuracy": spatial_acc,
#         "spatial_samples_evaluated": spatial_total
#     })
#     return {
#         "MACE": mace,
#         "count_samples_evaluated": len(count_errors),
#         "spatial_relation_accuracy": spatial_acc,
#         "spatial_samples_evaluated": spatial_total
#     }
