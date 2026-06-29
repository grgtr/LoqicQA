"""Apply INT8 dynamic quantization to Qwen2.5-3B-Instruct via torch.quantization.

Saves the quantized model state dict + config to the Triton model repository.

Usage:
    python scripts/quantize_torch.py [--output OUTPUT_DIR]

Output:
    deploy/triton/model_repository/qwen_quantized/1/quantized_model.pt
    deploy/triton/model_repository/qwen_quantized/1/config.json  (+ tokenizer files)
"""
import argparse
import sys
import time
from pathlib import Path


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_OUT = Path("deploy/triton/model_repository/qwen_quantized/1")


def main():
    parser = argparse.ArgumentParser(description="INT8 dynamic quantization for Qwen2.5-3B")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: torch / transformers not installed")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} in float32 for quantization...")
    print("(This takes ~5-10 minutes and uses ~12 GB RAM)")
    t0 = time.time()

    # Load in float32 — required for dynamic quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",  # quantize on CPU, GPU not needed
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    print("Applying INT8 dynamic quantization to Linear layers...")
    t1 = time.time()
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    print(f"Quantization done in {time.time() - t1:.1f}s")

    out_path = args.output / "quantized_model.pt"
    print(f"Saving quantized state dict to {out_path}...")
    torch.save(quantized.state_dict(), out_path)

    # Save config and tokenizer for the Triton model.py to reload architecture
    model.config.save_pretrained(str(args.output))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(str(args.output))

    size_gb = out_path.stat().st_size / 1e9
    print(f"Done. quantized_model.pt size: {size_gb:.2f} GB (was ~{size_gb * 2:.1f} GB bf16)")
    print(f"All files saved to {args.output}")


if __name__ == "__main__":
    main()
