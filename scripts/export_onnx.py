"""Export Qwen2.5-3B-Instruct to ONNX format.

Three modes (fastest → slowest):
  --mode dynamo   PyTorch 2.x dynamo exporter (default, 2-4x faster than tracing)
  --mode optimum  Hugging Face Optimum (causal-lm-with-past, KV-cache aware)
  --mode simple   torch.onnx.export tracing without past_key_values (fastest, smallest file)

Usage:
    python scripts/export_onnx.py [--mode dynamo|optimum|simple] [--dtype fp16|fp32]

Output:
    deploy/triton/model_repository/qwen_onnx/1/model.onnx
"""
import argparse
import sys
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_OUT = Path("deploy/triton/model_repository/qwen_onnx/1")

# Qwen2.5-3B from config.json
NUM_HEADS = 16
HIDDEN_SIZE = 2048
VOCAB_SIZE = 151936


def export_dynamo(out_dir: Path, dtype_str: str, device: str) -> Path:
    """PyTorch 2.x dynamo export — fastest, avoids Python-level tracing."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # fp16 is not supported on CPU — fall back to fp32
    if device == "cpu" and dtype_str == "fp16":
        print("  NOTE: fp16 not supported on CPU, using fp32")
        dtype_str = "fp32"
    torch_dtype = torch.float16 if dtype_str == "fp16" else torch.float32

    print(f"[dynamo] Loading model on {device} ({dtype_str})...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        # Eager attention avoids FlashAttention ops that dynamo can't trace
        attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # Dummy inputs (short sequence to keep tracing fast)
    dummy_text = "Hello"
    inputs = tokenizer(dummy_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    out_path = out_dir / "model.onnx"
    print(f"[dynamo] Exporting to {out_path}...")
    t1 = time.perf_counter()

    # PyTorch >= 2.5: torch.onnx.export(..., dynamo=True)
    # PyTorch 2.1-2.4: torch.onnx.dynamo_export(...)
    with torch.no_grad():
        if hasattr(torch.onnx, "dynamo_export"):
            # PyTorch 2.1–2.4
            onnx_program = torch.onnx.dynamo_export(model, input_ids, attention_mask=attention_mask)
            onnx_program.save(str(out_path))
        else:
            # PyTorch >= 2.5
            torch.onnx.export(
                model,
                (input_ids, attention_mask),
                str(out_path),
                dynamo=True,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_shapes={
                    "input_ids":      {0: torch.export.Dim("batch"), 1: torch.export.Dim("seq")},
                    "attention_mask": {0: torch.export.Dim("batch"), 1: torch.export.Dim("seq")},
                },
            )
    print(f"  Exported in {time.perf_counter() - t1:.1f}s")
    return out_path


def export_simple(out_dir: Path, dtype_str: str, device: str) -> Path:
    """Classic tracing export without KV-cache (causal-lm only, simplest graph).

    Fastest file size; Triton model.py wrapper handles autoregressive loop.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cpu" and dtype_str == "fp16":
        print("  NOTE: fp16 not supported on CPU, using fp32")
        dtype_str = "fp32"
    torch_dtype = torch.float16 if dtype_str == "fp16" else torch.float32

    print(f"[simple] Loading model on {device} ({dtype_str})...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    print(f"  RAM for 3B fp32: ~12 GB | fp16: ~6 GB")

    inputs = tokenizer("Hello", return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    out_path = out_dir / "model.onnx"
    print(f"[simple] Tracing and exporting to {out_path}...")
    t1 = time.perf_counter()

    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            str(out_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits":         {0: "batch", 1: "sequence"},
            },
            opset_version=18,
            do_constant_folding=True,
        )
    print(f"  Exported in {time.perf_counter() - t1:.1f}s")
    return out_path


def export_optimum(out_dir: Path, dtype_str: str) -> Path:
    """Hugging Face Optimum export with KV-cache (causal-lm-with-past).

    Produces decoder_model_merged.onnx with past_key_values support.
    Slowest but most complete for autoregressive generation in Triton ONNX backend.
    """
    import subprocess

    print(f"[optimum] Exporting via optimum-cli ({dtype_str})...")
    t0 = time.perf_counter()
    cmd = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", MODEL_ID,
        "--task", "causal-lm-with-past",
        "--dtype", dtype_str,
        "--attn-implementation", "eager",
        str(out_dir),
    ]
    print(f"  CMD: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"  Optimum export done in {time.perf_counter() - t0:.1f}s")

    # Rename merged model to model.onnx for Triton
    merged = out_dir / "decoder_model_merged.onnx"
    out_path = out_dir / "model.onnx"
    if merged.exists() and not out_path.exists():
        merged.rename(out_path)
        print(f"  Renamed decoder_model_merged.onnx → model.onnx")
    elif not out_path.exists():
        candidates = list(out_dir.glob("*.onnx"))
        if candidates:
            candidates[0].rename(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export Qwen2.5-3B to ONNX")
    parser.add_argument("--mode", choices=["dynamo", "simple", "optimum"], default="simple",
                        help="Export mode (default: simple — most compatible)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16",
                        help="Model dtype (fp16 recommended; CPU forces fp32)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                        help="Device to load model on (default: cuda if available, else cpu)")
    args = parser.parse_args()

    import torch
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "cuda":
            free_gb = (torch.cuda.get_device_properties(0).total_memory
                       - torch.cuda.memory_reserved(0)) / 1e9
            if free_gb < 7:
                print(f"WARNING: Only {free_gb:.1f} GB GPU free — switching to CPU export")
                print("  (GPU is occupied by other processes)")
                args.device = "cpu"

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {MODEL_ID} → ONNX")
    print(f"  mode={args.mode}  dtype={args.dtype}  device={args.device}  output={args.output}")
    print()

    t_total = time.perf_counter()

    if args.mode == "dynamo":
        out_path = export_dynamo(args.output, args.dtype, args.device)
    elif args.mode == "simple":
        out_path = export_simple(args.output, args.dtype, args.device)
    else:
        try:
            import optimum
        except ImportError:
            print("ERROR: optimum not installed. Run: pip install 'optimum[onnxruntime]'")
            sys.exit(1)
        out_path = export_optimum(args.output, args.dtype)

    total = time.perf_counter() - t_total
    if out_path.exists():
        # New torch.onnx uses external data format: weights go to model.onnx.data
        data_file = out_path.with_suffix(".onnx.data")
        total_bytes = out_path.stat().st_size
        if data_file.exists():
            total_bytes += data_file.stat().st_size
            print(f"\nDone in {total:.1f}s.")
            print(f"  model.onnx      : {out_path.stat().st_size / 1e6:.1f} MB  (graph)")
            print(f"  model.onnx.data : {data_file.stat().st_size / 1e9:.2f} GB (weights)")
        else:
            print(f"\nDone in {total:.1f}s. model.onnx size: {total_bytes / 1e9:.2f} GB")
    else:
        print(f"\nExport finished in {total:.1f}s. Check {args.output}")

    print("\nMode comparison (approximate times for Qwen2.5-3B on A100):")
    print("  dynamo : ~8-15 min  (fastest, torch.export-based)")
    print("  simple : ~15-25 min (classic tracing, no KV cache)")
    print("  optimum: ~30-60 min (tracing + KV cache graph)")


if __name__ == "__main__":
    main()
