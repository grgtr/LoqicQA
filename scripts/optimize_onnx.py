"""Optimize an ONNX model using ONNX Runtime transformer optimizer.

Fuses attention patterns, layer norms, and activations into optimized kernels.
Must be run AFTER scripts/export_onnx.py.

Usage:
    python scripts/optimize_onnx.py [--input INPUT] [--output OUTPUT]

Input:
    deploy/triton/model_repository/qwen_onnx/1/model.onnx
Output:
    deploy/triton/model_repository/qwen_onnx_optimized/1/model.onnx
"""
import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path


DEFAULT_INPUT = Path("deploy/triton/model_repository/qwen_onnx/1/model.onnx")
DEFAULT_OUTPUT = Path("deploy/triton/model_repository/qwen_onnx_optimized/1/model.onnx")

# Qwen2.5-3B architecture constants (from config.json)
NUM_HEADS = 16
HIDDEN_SIZE = 2048


def inline_external_data(input_path: Path, tmp_dir: Path) -> Path:
    """If the model uses external data (model.onnx + model.onnx.data),
    copy it to a temp dir so ORT optimizer can find both files side by side."""
    import onnx
    from onnx.external_data_helper import load_external_data_for_model

    data_file = input_path.with_suffix(".onnx.data")
    if not data_file.exists():
        return input_path  # already self-contained

    print(f"  Detected external data format — loading weights into memory (~6 GB RAM)...")
    # Load model with weights fully in memory
    model = onnx.load(str(input_path), load_external_data=True)

    # Save to temp dir using external data format (avoids 2 GB protobuf limit)
    tmp_onnx = tmp_dir / "model.onnx"
    print(f"  Writing to temp dir {tmp_dir} ...")
    onnx.save_model(
        model,
        str(tmp_onnx),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=1024,
    )
    size_gb = (tmp_onnx.stat().st_size + (tmp_dir / "model.onnx.data").stat().st_size) / 1e9
    print(f"  Temp model ready: {size_gb:.2f} GB")
    return tmp_onnx


def save_with_external_data(opt_model, output_path: Path) -> None:
    """Save optimized model; use external data if > 2 GB to avoid protobuf 2 GB limit."""
    import onnx

    # First try to save as-is (may work if model fits in protobuf)
    try:
        opt_model.save_model_to_file(str(output_path))
        size = output_path.stat().st_size
        if size > 1000:  # non-empty
            return
    except Exception:
        pass

    # Fallback: save with external data
    print("  Saving with external data format (model > 2 GB)...")
    model_proto = opt_model.model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model_proto,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=1024,
    )


def main():
    parser = argparse.ArgumentParser(description="Optimize ONNX model with ORT transformer optimizer")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-type", default="gpt2",
                        help="ORT model type: gpt2 or gpt_neox (default: gpt2)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input ONNX not found: {args.input}")
        print("Run scripts/export_onnx.py first.")
        sys.exit(1)

    try:
        from onnxruntime.transformers import optimizer
        import onnx
    except ImportError as e:
        print(f"ERROR: {e}. Run: pip install onnxruntime-gpu onnx")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    data_file = args.input.with_suffix(".onnx.data")
    size_in = (args.input.stat().st_size + (data_file.stat().st_size if data_file.exists() else 0)) / 1e9
    print(f"Optimizing {args.input} ({size_in:.2f} GB total)")
    print(f"  model_type={args.model_type}, num_heads={NUM_HEADS}, hidden_size={HIDDEN_SIZE}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # ORT optimizer requires a single self-contained ONNX file
        model_to_optimize = inline_external_data(args.input, tmp_path)

        t0 = time.time()
        opt_model = optimizer.optimize_model(
            str(model_to_optimize),
            model_type=args.model_type,
            num_heads=NUM_HEADS,
            hidden_size=HIDDEN_SIZE,
            opt_level=2,
            use_gpu=True,
            only_onnxruntime=False,
        )
        elapsed = time.time() - t0
        print(f"Optimization done in {elapsed:.1f}s")

        # get_node_count() was removed in newer ORT; use len(nodes()) instead
        try:
            n = len(opt_model.nodes())
            print(f"Nodes: {n}")
        except Exception:
            pass

        print(f"Saving to {args.output}...")
        save_with_external_data(opt_model, args.output)

    # Report final size
    out_data = args.output.with_suffix(".onnx.data")
    total = args.output.stat().st_size + (out_data.stat().st_size if out_data.exists() else 0)
    print(f"Done. Output size: {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
