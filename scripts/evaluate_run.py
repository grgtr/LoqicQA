"""
python3 scripts/evaluate_run.py --levels 1,2,3,4 --judge_model "Qwen/Qwen2.5-3B-Instruct" --device cuda --artifacts results/breakfast_box_20260310_143718/run_artifacts.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import numpy as np

from logicqa.data.evaluation_gt import ATOMIC_CONSTRAINTS, ATTRIBUTE_GT
from logicqa.evaluation.level3_reasoning import calculate_subq_consistency
from logicqa.evaluation.level2_5_filtering import evaluate_filtering

def calculate_level4_metrics(image_results):
    """Calculate AUROC and F1-max for Level 4."""
    y_true = []
    y_scores = []
    y_pred_binary = []

    for res in image_results:
        y_true.append(1 if res.get("is_anomaly") else 0)
        y_scores.append(res.get("anomaly_score", 0.0))
        # Binary prediction (if anomaly_score > 0.5 or based on 'is_anomaly')
        y_pred_binary.append(1 if res.get("is_anomaly") else 0)

    if len(set(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_scores)
        # Calculate F1-max
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
        f1_max = np.max(f1_scores)
    else:
        auroc, f1_max = 0.0, 0.0

    accuracy = sum([1 for t, p in zip(y_true, y_pred_binary) if t == p]) / len(y_true) if y_true else 0.0
    
    return {"AUROC": auroc, "F1-max": f1_max, "Accuracy": accuracy}

def print_report(class_name: str, reports: dict):
    print(f"\n{'='*60}")
    print(f" HIERARCHICAL EVALUATION REPORT: {class_name.upper()}")
    print(f"{'='*60}")

    if "level4" in reports:
        print("\n[LEVEL 4: TASK] (Is the anomaly detected correctly?)")
        m = reports["level4"]
        print(f" ├─ AUROC:    {m['AUROC']:.2%}")
        print(f" ├─ F1-max:   {m['F1-max']:.2%}")
        print(f" └─ Accuracy: {m['Accuracy']:.2%}")

    if "level3" in reports:
        print("\n[LEVEL 3: REASONING] (Check consistency of reasoning)")
        m = reports["level3"]
        print(f" └─ Sub-Q Consistency Score: {m['consistency']:.2%} (higher = less hallucinations)")

    if "filtering" in reports:
        print("\n[STAGE 3: FILTERING] (Релевантность вопросов)")
        m = reports["filtering"]
        print(f" ├─ Filter Precision (Нет мусора среди Kept): {m['Filter_Precision']:.2%}")
        print(f" ├─ Filter Recall (Покрытие ГОСТа после фильтра): {m['Filter_Recall']:.2%}")
        print(f" └─ Statistics: {m['Total_Candidates']} generated -> {m['Kept']} kept, {m['Dropped']} dropped")


    if "level2" in reports:
        print("\n[LEVEL 2: ATTRIBUTES] (Check detected attributes of objects)")
        m = reports["level2"]
        if m.get('MACE') is not None:
            print(f" ├─ Mean Absolute Count Error (MACE): {m['MACE']:.2f} objects")
        else:
            print(f" ├─ MACE: N/A (no quantitative anomalies detected)")
            
        if m.get('spatial_relation_accuracy') is not None:
            print(f" └─ Spatial Relation Accuracy:      {m['spatial_relation_accuracy']:.2%}")
        else:
            print(f" └─ Spatial Relation Accuracy:      N/A")

    if "level1" in reports:
        print("\n[LEVEL 1: PERCEPTION] (Is the scene perceived correctly?)")
        m = reports["level1"]
        print(f" ├─ Constraint Coverage Rate (CCR): {m['mean_ccr']:.2%}")
        print(f" └─ Average CLIPScore:              {m['mean_clip']:.2f}%")

    print(f"\n{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Run 4-level evaluation framework")
    parser.add_argument("--artifacts", type=str, required=True, help="Path to run_artifacts.json")
    parser.add_argument("--levels", type=str, default="1,2,3,4", help="Levels to evaluate (comma-separated, e.g. '1,2,3,4')")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HF model ID for LLM-Judge")
    parser.add_argument("--device", type=str, default="cuda", help="Device for Judge and CLIP")
    args = parser.parse_args()

    # 1. Load artifacts
    artifacts_path = Path(args.artifacts)
    if not artifacts_path.exists():
        print(f"[Error] Artifacts file not found: {artifacts_path}")
        sys.exit(1)

    print(f"Loading artifacts from {artifacts_path}...")
    with open(artifacts_path, "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    class_name = artifacts.get("class_name", "unknown")
    levels = [lvl.strip() for lvl in args.levels.split(",")]
    report = {}

    judge = None
    perception_eval = None

    # =================================================================
    # Level 4 (Task)
    # =================================================================
    if "4" in levels:
        print("\n--- Evaluating Level 4 (Task) ---")
        img_results = artifacts.get("stage4_image_results", [])
        report["level4"] = calculate_level4_metrics(img_results)

    # =================================================================
    # Level 3 (Reasoning)
    # =================================================================
    if "3" in levels:
        print("\n--- Evaluating Level 3 (Reasoning) ---")
        subq_responses = artifacts.get("stage4_subq_responses", [])
        cons_score = calculate_subq_consistency(subq_responses)
        report["level3"] = {"consistency": cons_score}

    # =================================================================
    # Level 2 (Attributes)
    # =================================================================
    if "2.5" in levels or "f" in levels:
        print("\n--- Evaluating Stage 3 Filtering Quality ---")
        if judge is None:
            from logicqa.evaluation.llm_judge import LLMJudge
            judge = LLMJudge(model_id=args.judge_model, device=args.device)
            
        stage3a = artifacts.get("stage3a_questions", {})
        stage3b = artifacts.get("stage3b_filtering", [])
        gt_constraints = ATOMIC_CONSTRAINTS.get(class_name, [])
        
        report["filtering"] = evaluate_filtering(stage3a, stage3b, gt_constraints, judge)
        

    if "2" in levels:
        print("\n--- Evaluating Level 2 (Attributes) ---")
        if judge is None:
            from logicqa.evaluation.llm_judge import LLMJudge
            judge = LLMJudge(model_id=args.judge_model, device=args.device)
            
        from logicqa.evaluation.level2_attributes import evaluate_attributes
        subq_responses = artifacts.get("stage4_subq_responses", [])
        img_results = artifacts.get("stage4_image_results", [])
        
        # Using anomaly_type from image_results in subq_responses for Level 2
        img_to_anomaly = {r.get("image_path"): r.get("anomaly_type") for r in img_results}
        for resp in subq_responses:
            path = resp.get("image_path")
            if path in img_to_anomaly:
                resp["anomaly_type"] = img_to_anomaly[path]

        report["level2"] = evaluate_attributes(subq_responses, ATTRIBUTE_GT, class_name, judge)

    # =================================================================
    # Level 1 (Perception)
    # =================================================================
    if "1" in levels:
        print("\n--- Evaluating Level 1 (Perception) ---")
        if judge is None:
            from logicqa.evaluation.llm_judge import LLMJudge
            judge = LLMJudge(model_id=args.judge_model, device=args.device)
            
        from logicqa.evaluation.level1_perception import PerceptionEvaluator
        perception_eval = PerceptionEvaluator(device=args.device)
        
        descriptions = artifacts.get("stage1_descriptions", [])
        gt_constraints = ATOMIC_CONSTRAINTS.get(class_name, [])
        
        ccr_scores = []
        clip_scores = []
        
        for desc in descriptions:
            text = desc.get("response", "")
            img_path = desc.get("image_path", "")
            
            # CCR
            ccr_res = perception_eval.calculate_ccr(text, gt_constraints, judge)
            ccr_scores.append(ccr_res["ccr_score"])
            # print(f"\n[Debug CCR] description: {text[:100]}...")
            for constraint, is_covered in ccr_res["details"].items():
                if not is_covered:
                    print(f"  Skipped: {constraint}")
            
            # CLIP
            if os.path.exists(img_path):
                clip = perception_eval.calculate_clip_score(img_path, text)
                clip_scores.append(clip)
            else:
                print(f"[Warning] Image not found for CLIPScore: {img_path}")

        report["level1"] = {
            "mean_ccr": sum(ccr_scores) / len(ccr_scores) if ccr_scores else 0.0,
            "mean_clip": sum(clip_scores) / len(clip_scores) if clip_scores else 0.0
        }

    print_report(class_name, report)

if __name__ == "__main__":
    main()
