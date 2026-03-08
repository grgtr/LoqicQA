#!/usr/bin/env python3
"""
run_pipeline.py — Run the LogicQA pipeline on one MVTec LOCO AD class.

Example:
    python scripts/run_pipeline.py \\
        --class_name breakfast_box \\
        --data_dir ~/dataset-ninja/ \\
        --vlm internvl \\
        --n_shots 3 \\
        --output_dir results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is in path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from logicqa.config import LogicQAConfig
from logicqa.pipeline.logicqa import LogicQAPipeline
from logicqa.data.mvtec_loco import MVTecLOCODataset
from logicqa.evaluation.metrics import print_evaluation_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LogicQA logical anomaly detection pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--class_name", "-c",
        required=True,
        help="MVTec LOCO AD class name (e.g., breakfast_box, juice_bottle).",
    )
    parser.add_argument(
        "--data_dir", "-d",
        default="~/dataset-ninja/",
        help="Root directory containing the MVTec LOCO AD dataset.",
    )
    parser.add_argument(
        "--vlm",
        default="internvl",
        choices=["internvl", "gpt4o", "gemini"],
        help="VLM backend to use.",
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        default=3,
        help="Number of few-shot normal images for setup.",
    )
    parser.add_argument(
        "--n_questions",
        type=int,
        default=6,
        help="Number of candidate main questions to generate.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (optional).",
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="results",
        help="Output directory for results JSON.",
    )
    parser.add_argument(
        "--questions_file",
        default=None,
        help="Load pre-generated questions from JSON (skip setup). "
             "If not set, runs full setup.",
    )
    parser.add_argument(
        "--save_questions",
        action="store_true",
        help="Save generated questions to JSON for reuse.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for few-shot sampling.",
    )
    parser.add_argument(
        "--max_test",
        type=int,
        default=None,
        help="Limit test set size (for quick experiments).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Load config
    # ------------------------------------------------------------------ #
    if args.config:
        cfg = LogicQAConfig.from_yaml(args.config)
    else:
        cfg = LogicQAConfig.default()

    # Override from CLI args
    cfg.vlm.backend = args.vlm
    cfg.pipeline.n_shots = args.n_shots
    cfg.dataset.data_dir = args.data_dir
    cfg.dataset.download_if_missing = True

    # ------------------------------------------------------------------ #
    # Load dataset
    # ------------------------------------------------------------------ #
    print(f"[Dataset] Loading MVTec LOCO AD class: {args.class_name}")
    dataset = MVTecLOCODataset(
        data_dir=args.data_dir,
        class_name=args.class_name,
        download_if_missing=True,
    )

    # ------------------------------------------------------------------ #
    # Build pipeline
    # ------------------------------------------------------------------ #
    pipeline = LogicQAPipeline(cfg)

    if args.questions_file:
        # Skip setup — load pre-generated questions
        pipeline.load_questions(args.questions_file)
    else:
        # Sample few-shot normal images and run setup (Stages 1-3)
        normal_images = dataset.sample_train_normal(n=args.n_shots, seed=args.seed)
        print(f"[Setup] Using {len(normal_images)} normal images: "
              f"{[p.name for p in normal_images]}")

        pipeline.setup(
            class_name=args.class_name,
            normal_images=normal_images,
            n_questions=args.n_questions,
        )

        if args.save_questions:
            out_dir = Path(os.path.expanduser(args.output_dir))
            out_dir.mkdir(parents=True, exist_ok=True)
            q_path = out_dir / f"{args.class_name}_questions.json"
            pipeline.save_questions(q_path)

    # ------------------------------------------------------------------ #
    # Test (Stage 4)
    # ------------------------------------------------------------------ #
    test_samples = dataset.get_test_images()
    if args.max_test:
        test_samples = test_samples[: args.max_test]

    print(f"\n[Test] Running inference on {len(test_samples)} test images ...")

    results_list = []
    predictions = []
    labels = []
    scores = []

    for i, sample in enumerate(test_samples):
        print(f"  [{i+1}/{len(test_samples)}] {sample.path.name} "
              f"(label={sample.label})", end=" ", flush=True)

        result = pipeline.predict(sample.path)
        pred_label = "anomaly" if result.is_anomaly else "normal"
        gt_label = 1 if sample.is_anomaly else 0

        print(f"→ pred={pred_label}, score={result.anomaly_score:.4f}")

        results_list.append({
            "path": str(sample.path),
            "gt_label": sample.label,
            "gt_is_anomaly": sample.is_anomaly,
            "pred_is_anomaly": result.is_anomaly,
            "anomaly_score": result.anomaly_score,
            "explanation": result.explanation,
            "main_q_results": [
                {
                    "question": mq.question,
                    "voted_answer": mq.voted_answer,
                    "best_log_prob": mq.best_log_prob,
                }
                for mq in result.main_q_results
            ],
        })
        predictions.append(result.is_anomaly)
        labels.append(gt_label)
        scores.append(result.anomaly_score)

    # ------------------------------------------------------------------ #
    # Evaluate & save
    # ------------------------------------------------------------------ #
    metrics = print_evaluation_summary(
        class_name=args.class_name,
        anomaly_scores=scores,
        predictions=predictions,
        labels=labels,
    )

    out_dir = Path(os.path.expanduser(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{args.class_name}_results.json"

    with open(results_path, "w") as f:
        json.dump(
            {
                "class_name": args.class_name,
                "vlm": args.vlm,
                "n_shots": args.n_shots,
                "metrics": metrics,
                "results": results_list,
            },
            f,
            indent=2,
        )
    print(f"\n[Done] Results saved to {results_path}")


if __name__ == "__main__":
    main()
