#!/usr/bin/env python3
"""
Backfill empty image_path fields in stage4_final_results.json and
stage4_responses.json for classes where LangSAM was used (pushpins,
splicing_connectors).

The paths are recovered by loading the dataset test image list (same
deterministic order as the pipeline) and zipping with JSON entries by index.

Usage:
    python3 backfill_image_paths.py --class_name pushpins \
        --run_dir results_baseline_model/pushpins/pushpins_20260510_215359 \
        --data_dir /home/chikibriki/LoqicQA/dataset-ninja/
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logicqa.data.mvtec_loco import MVTecLOCODataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_dir", default="/home/chikibriki/LoqicQA/dataset-ninja/")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print first 5 mappings without writing files")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    final_path = run_dir / "stage4_final_results.json"
    resp_path  = run_dir / "stage4_responses.json"

    # ------------------------------------------------------------------ #
    # Load dataset test images (same order as pipeline)
    # ------------------------------------------------------------------ #
    dataset = MVTecLOCODataset(
        data_dir=args.data_dir,
        class_name=args.class_name,
        download_if_missing=False,
    )
    test_samples = dataset.get_test_images()
    image_paths = [str(s.path) for s in test_samples]
    print(f"Dataset: {len(image_paths)} test images for '{args.class_name}'")

    # ------------------------------------------------------------------ #
    # Backfill stage4_final_results.json (1 entry per image)
    # ------------------------------------------------------------------ #
    with open(final_path) as f:
        final_results = json.load(f)

    if len(final_results) != len(image_paths):
        print(f"WARNING: final_results has {len(final_results)} entries "
              f"but dataset has {len(image_paths)} images — sizes differ!")

    for i, (entry, path) in enumerate(zip(final_results, image_paths)):
        if args.dry_run and i < 5:
            print(f"  final[{i}]: '' → '{path}'")
        entry["image_path"] = path

    # ------------------------------------------------------------------ #
    # Backfill stage4_responses.json (N responses per image)
    # ------------------------------------------------------------------ #
    with open(resp_path) as f:
        responses = json.load(f)

    n_images = len(image_paths)
    n_responses = len(responses)

    if n_responses % n_images != 0:
        print(f"WARNING: {n_responses} responses not evenly divisible by "
              f"{n_images} images — will assign by chunk, remainder left as-is")

    responses_per_image = n_responses // n_images
    print(f"Responses per image: {responses_per_image} "
          f"({n_responses} total / {n_images} images)")

    for i, path in enumerate(image_paths):
        start = i * responses_per_image
        end   = start + responses_per_image
        for entry in responses[start:end]:
            if args.dry_run and i == 0:
                print(f"  resp[{start}]: '' → '{path}'")
            entry["image_path"] = path

    # ------------------------------------------------------------------ #
    # Write back (or dry-run report)
    # ------------------------------------------------------------------ #
    if args.dry_run:
        print("\n[dry_run] No files written.")
        return

    with open(final_path, "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"Written: {final_path}")

    with open(resp_path, "w") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
    print(f"Written: {resp_path}")

    # Verify
    empty_final = sum(1 for r in final_results if not r.get("image_path"))
    empty_resp  = sum(1 for r in responses  if not r.get("image_path"))
    print(f"\nVerification — empty image_path remaining: "
          f"final={empty_final}, responses={empty_resp}")


if __name__ == "__main__":
    main()
