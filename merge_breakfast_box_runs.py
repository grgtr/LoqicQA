#!/usr/bin/env python3
"""Merge two breakfast_box pipeline runs into a single output directory."""

import json
import os
from pathlib import Path

RUN1 = Path("results/breakfast_box_20260510_110252")
RUN2 = Path("results_baseline_model/breakfast_box_resume/breakfast_box_20260510_190556")
OUT  = Path("results_baseline_model/breakfast_box")

OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
# stage4_final_results.json  — list of per-image dicts
# ------------------------------------------------------------------ #
with open(RUN1 / "stage4_final_results.json") as f:
    results1 = json.load(f)
with open(RUN2 / "stage4_final_results.json") as f:
    results2 = json.load(f)

# Deduplicate by image_path (run1 takes precedence for overlapping images)
seen = {r["image_path"] for r in results1}
unique2 = [r for r in results2 if r["image_path"] not in seen]

merged_results = results1 + unique2
merged_results.sort(key=lambda r: r["image_path"])

out_final = OUT / "stage4_final_results.json"
with open(out_final, "w") as f:
    json.dump(merged_results, f, indent=2, ensure_ascii=False)
print(f"stage4_final_results.json: {len(results1)} + {len(unique2)} = {len(merged_results)} images → {out_final}")

# ------------------------------------------------------------------ #
# stage4_responses.json  — list of per-sub-question dicts
# ------------------------------------------------------------------ #
with open(RUN1 / "stage4_responses.json") as f:
    resp1 = json.load(f)
with open(RUN2 / "stage4_responses.json") as f:
    resp2 = json.load(f)

# Deduplicate by (image_path, main_question, sub_q_idx)
def resp_key(r):
    return (r.get("image_path", ""), r.get("main_question", ""), r.get("sub_q_idx", 0))

seen_resp = {resp_key(r) for r in resp1}
unique_resp2 = [r for r in resp2 if resp_key(r) not in seen_resp]

merged_resp = resp1 + unique_resp2
merged_resp.sort(key=lambda r: (r.get("image_path", ""), r.get("sub_q_idx", 0)))

out_resp = OUT / "stage4_responses.json"
with open(out_resp, "w") as f:
    json.dump(merged_resp, f, indent=2, ensure_ascii=False)
print(f"stage4_responses.json:     {len(resp1)} + {len(unique_resp2)} = {len(merged_resp)} rows → {out_resp}")

# ------------------------------------------------------------------ #
# pipeline.log  — concatenate with separator
# ------------------------------------------------------------------ #
out_log = OUT / "pipeline.log"
with open(out_log, "w") as out:
    for run_dir in [RUN1, RUN2]:
        log_file = run_dir / "pipeline.log"
        if log_file.exists():
            out.write(f"\n{'='*60}\n")
            out.write(f"# Source: {run_dir}\n")
            out.write(f"{'='*60}\n\n")
            out.write(log_file.read_text())
print(f"pipeline.log:              concatenated 2 logs → {out_log}")

# ------------------------------------------------------------------ #
# Quick sanity summary
# ------------------------------------------------------------------ #
gt_labels = [r["gt_label"] for r in merged_results]
from collections import Counter
print(f"\nLabel distribution in merged results:")
for label, count in sorted(Counter(gt_labels).items()):
    print(f"  {label}: {count}")
print(f"\nTotal images: {len(merged_results)}")
