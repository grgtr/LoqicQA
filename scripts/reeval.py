"""
Re-evaluate LogicQA results with a different anomaly_min_failures threshold.

Usage:
    python scripts/reeval.py <results_json> [--threshold N] [--show-errors]

Example:
    python scripts/reeval.py results/decomposed_bb50_r25/breakfast_box_20260519_233659/breakfast_box_results.json --threshold 1
"""

import argparse
import json
import sys
from pathlib import Path


def compute_auroc(labels, scores):
    """Compute AUROC via trapezoidal rule."""
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = 0
    prev_tp = prev_fp = 0
    auc = 0.0
    prev_score = None
    for score, label in pairs:
        if score != prev_score and prev_score is not None:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_fp, prev_tp = fp, tp
        if label:
            tp += 1
        else:
            fp += 1
        prev_score = score
    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
    return auc / (n_pos * n_neg)


def compute_f1_max(labels, scores):
    """Sweep thresholds to find max F1."""
    thresholds = sorted(set(scores), reverse=True)
    best_f1 = 0.0
    best_thresh = thresholds[0] if thresholds else 0.5
    for t in thresholds:
        preds = [1 if s >= t else 0 for s in scores]
        tp = sum(p and l for p, l in zip(preds, labels))
        fp = sum(p and not l for p, l in zip(preds, labels))
        fn = sum(not p and l for p, l in zip(preds, labels))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_f1, best_thresh


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate LogicQA results at a different threshold.")
    parser.add_argument("results_json", help="Path to breakfast_box_results.json")
    parser.add_argument("--threshold", "-t", type=int, default=1,
                        help="New anomaly_min_failures (default: 1)")
    parser.add_argument("--show-errors", action="store_true",
                        help="Print FP/FN details")
    args = parser.parse_args()

    path = Path(args.results_json)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    threshold = args.threshold
    original_threshold = data.get("config", {}).get("anomaly_min_failures", "?")

    labels, scores, preds = [], [], []
    fps, fns = [], []

    for r in results:
        gt = int(r["gt_is_anomaly"])

        # count "No" votes from main_q_results
        mq_results = r.get("main_q_results", [])
        n_no = sum(1 for mq in mq_results if mq.get("voted_answer", "").strip().lower() == "no")
        n_total = len(mq_results)

        score = n_no / n_total if n_total > 0 else 0.0
        pred = int(n_no >= threshold)

        labels.append(gt)
        scores.append(score)
        preds.append(pred)

        name = Path(r["path"]).name
        if not gt and pred:
            fps.append((name, score, n_no, n_total))
        elif gt and not pred:
            fns.append((name, score, n_no, n_total))

    tp = sum(p and l for p, l in zip(preds, labels))
    tn = sum(not p and not l for p, l in zip(preds, labels))
    fp = sum(p and not l for p, l in zip(preds, labels))
    fn = sum(not p and l for p, l in zip(preds, labels))

    n_total = len(labels)
    n_normal = sum(1 for l in labels if not l)
    n_anomaly = sum(labels)

    auroc = compute_auroc(labels, scores)
    f1_max, f1_thresh = compute_f1_max(labels, scores)
    bin_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    bin_rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    bin_f1   = 2 * bin_prec * bin_rec / (bin_prec + bin_rec) if (bin_prec + bin_rec) > 0 else 0.0
    accuracy = (tp + tn) / n_total

    print()
    print(f"Results: {path}")
    print(f"Threshold: anomaly_min_failures = {threshold}  (original in file: {original_threshold})")
    print()
    print("=" * 50)
    print(" Evaluation: breakfast_box")
    print("=" * 50)
    print(f"  Total:    {n_total} ({n_normal} normal, {n_anomaly} anomaly)")
    print(f"  AUROC:    {auroc:.4f}  (paper target: ~0.876)")
    print(f"  F1-max:   {f1_max:.4f}  (paper target: ~0.870)  [at score≥{f1_thresh:.4f}]")
    print(f"  Bin-F1:   {bin_f1:.4f}")
    print("=" * 50)
    print()
    print("=" * 60)
    print("CONFUSION MATRIX (Logical Anomaly Detection)")
    print("=" * 60)
    print(f"{'':18s} | {'Predicted Normal':16s} | {'Predicted Anomaly':17s} |")
    print("-" * 60)
    print(f" {'Actual Normal':16s} | {tn:<16} | {fp:<17} |")
    print(f" {'Actual Anomaly':16s} | {fn:<16} | {tp:<17} |")
    print("=" * 60)
    print(f"Metrics Summary:")
    print(f" - Accuracy:  {accuracy*100:.2f}%")
    print(f" - Precision: {bin_prec*100:.2f}%")
    print(f" - Recall:    {bin_rec*100:.2f}%")
    print("=" * 60)

    if args.show_errors or fps or fns:
        print()
        if fps:
            print(f"=== FALSE POSITIVES ({len(fps)}) — normal → anomaly ===")
            for name, score, n_no, n_tot in fps:
                print(f"  {name}  No={n_no}/{n_tot}  score={score:.3f}")
        else:
            print("=== FALSE POSITIVES: none ===")

        print()
        if fns:
            print(f"=== FALSE NEGATIVES ({len(fns)}) — anomaly missed ===")
            for name, score, n_no, n_tot in fns:
                print(f"  {name}  No={n_no}/{n_tot}  score={score:.3f}")
        else:
            print("=== FALSE NEGATIVES: none ===")

    print()


if __name__ == "__main__":
    main()
