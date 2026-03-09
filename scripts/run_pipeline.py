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

def select_test_samples(
    all_samples: List[ImageSample],
    cfg: LogicQAConfig,
) -> List[ImageSample]:
    """
    Select test samples based on config.testing.mode:
      - "all"      : все сэмплы класса
      - "specific" : только файлы из testing.specific_samples
      - "random"   : N случайных, с чередованием если testing.interleave=True
    """
    testing = cfg.testing
    mode = getattr(testing, "mode", "all")

    if mode == "all":
        selected = all_samples
        print(f"[TestSelect] mode=all → {len(selected)} samples")

    elif mode == "specific":
        names = set(getattr(testing, "specific_samples", []))
        selected = [s for s in all_samples if s.path.name in names]
        missing = names - {s.path.name for s in selected}
        if missing:
            print(f"[TestSelect] WARNING: not found in dataset: {missing}")
        print(f"[TestSelect] mode=specific → {len(selected)} samples")

    elif mode == "random":
        n = getattr(testing, "random_count", 10)
        seed = getattr(testing, "random_seed", 42)
        interleave = getattr(testing, "interleave", True)
        selected = _sample_random(all_samples, n=n, seed=seed, interleave=interleave)
        print(f"[TestSelect] mode=random (n={n}, interleave={interleave}) "
              f"→ {len(selected)} samples")

    else:
        raise ValueError(
            f"Unknown testing.mode: '{mode}'. Choose from: all, specific, random"
        )

    return selected


def _sample_random(
    samples: List[ImageSample],
    n: int,
    seed: int = 42,
    interleave: bool = True,
) -> List[ImageSample]:
    """
    Sample n images. If interleave=True, чередуем normal/anomaly пока возможно,
    затем добираем из оставшегося типа.
    """
    rng = random.Random(seed)

    normals = [s for s in samples if not s.is_anomaly]
    anomalies = [s for s in samples if s.is_anomaly]
    rng.shuffle(normals)
    rng.shuffle(anomalies)

    if not interleave:
        pool = normals + anomalies
        rng.shuffle(pool)
        selected = pool[:n]
        return selected

    # Чередование: N, A, N, A, ... пока оба списка не пустые
    selected: List[ImageSample] = []
    ni, ai = 0, 0
    turn = 0  # 0 = normal first, 1 = anomaly first
    while len(selected) < n:
        if turn == 0:
            if ni < len(normals):
                selected.append(normals[ni]); ni += 1
            elif ai < len(anomalies):
                selected.append(anomalies[ai]); ai += 1
            else:
                break
        else:
            if ai < len(anomalies):
                selected.append(anomalies[ai]); ai += 1
            elif ni < len(normals):
                selected.append(normals[ni]); ni += 1
            else:
                break
        turn = 1 - turn

    if len(selected) < n:
        print(f"[TestSelect] WARNING: requested {n} samples but only "
              f"{len(selected)} available "
              f"({len(normals)} normal, {len(anomalies)} anomaly)")
    return selected


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
        default=None,
        help="Number of few-shot normal images for setup.",
    )
    parser.add_argument(
        "--n_questions",
        type=int,
        default=None,
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
        default=None,
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
    if args.n_shots is not None:
        cfg.pipeline.n_shots = args.n_shots
    if args.n_questions is not None:
        cfg.pipeline.n_questions = args.n_questions
    if args.seed is not None:
        cfg.testing.random_seed = args.seed
    cfg.dataset.data_dir = args.data_dir
    cfg.dataset.download_if_missing = True
    cfg.pipeline.output_dir = args.output_dir
    cfg.pipeline.class_name = args.class_name
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
        print(f"[Setup] Loaded questions from {args.questions_file}")
    else:
        # Sample few-shot normal images and run setup (Stages 1-3)
        # print("[DEBUG] n_shots", args.n_shots)
        normal_images = dataset.sample_train_normal(n=cfg.pipeline.n_shots, seed=args.seed)
        print(f"[Setup] Using {len(normal_images)} normal images: "
              f"{[p.name for p in normal_images]}")

        pipeline.setup(
            class_name=args.class_name,
            normal_images=normal_images,
            n_questions=cfg.pipeline.n_questions,
        )

        if args.save_questions:
            out_dir = Path(os.path.expanduser(args.output_dir))
            out_dir.mkdir(parents=True, exist_ok=True)
            q_path = out_dir / f"{args.class_name}_questions.json"
            pipeline.save_questions(q_path)

    # ------------------------------------------------------------------ #
    # Test (Stage 4)
    # ------------------------------------------------------------------ #
    # test_samples = dataset.get_test_images()
    # if args.max_test:
    #     test_samples = test_samples[: args.max_test]

    all_test = dataset.get_test_images()
    test_samples = select_test_samples(all_test, cfg)

    if not test_samples:
        print("[ERROR] No test samples selected. Check config.testing settings.")
        sys.exit(1)

    print(f"\n[Test] Running inference on {len(test_samples)} test images ...")

    results_list = []
    predictions = []
    labels = []
    scores = []

    for i, sample in enumerate(test_samples):
        # print(f"  [{i+1}/{len(test_samples)}] {sample.path.name} "
        #       f"(label={sample.label})", end=" ", flush=True)
        tag = "ANOMALY" if sample.is_anomaly else "normal "
        print(f"  [{i+1:03d}/{len(test_samples):03d}] [{tag}] {sample.path.name}",
              end=" ", flush=True)
        result = pipeline.predict(sample.path, gt_label=sample.label)
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
    eval_scores, eval_labels, eval_preds = [], [], []
    for r in results_list:
        if r["gt_label"] == "structural_anomaly":
            continue
        eval_scores.append(r["anomaly_score"])
        eval_labels.append(1 if r["gt_is_anomaly"] else 0)
        eval_preds.append(r["pred_is_anomaly"])

    metrics = print_evaluation_summary(
        class_name=args.class_name,
        anomaly_scores=eval_scores,
        predictions=eval_preds,
        labels=eval_labels,
    )


    out_dir = Path(os.path.expanduser(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{args.class_name}_results.json"

    with open(results_path, "w") as f:
        json.dump(
            {
                "class_name": args.class_name,
                "vlm": args.vlm,
                "n_shots": cfg.pipeline.n_shots,
                "metrics": metrics,
                "results": results_list,
            },
            f,
            indent=2,
        )
    print(f"\n[Done] Results saved to {results_path}")


if __name__ == "__main__":
    main()
