"""End-to-end Triton Inference Server test for all Qwen model variants."""
import argparse
import json
import time
from pathlib import Path

TRITON_URL = "localhost:8111"
PREPROCESS_URL = "http://localhost:8080"
MODELS = ["qwen_original", "qwen_quantized", "qwen_onnx", "qwen_onnx_optimized"]
TEST_PROMPT = "Answer in one sentence: What is the capital of France?"


def test_preprocessing(preprocess_url: str) -> dict:
    import requests

    result = {"status": "skipped", "latency_ms": None}
    try:
        r = requests.get(f"{preprocess_url}/health", timeout=5)
        if r.status_code != 200:
            result["status"] = f"health_failed ({r.status_code})"
            return result
    except Exception as e:
        result["status"] = f"unreachable: {e}"
        return result

    try:
        t0 = time.perf_counter()
        r = requests.post(
            f"{preprocess_url}/preprocess",
            json={"prompt": TEST_PROMPT, "system": "You are a helpful assistant."},
            timeout=10,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        result["status"] = "ok" if r.status_code == 200 else f"error_{r.status_code}"
        result["latency_ms"] = round(elapsed_ms, 2)
        if r.status_code == 200:
            data = r.json()
            result["token_count"] = data.get("token_count")
    except Exception as e:
        result["status"] = f"error: {e}"

    return result


def test_python_backend_model(client, model_name: str, prompt: str) -> dict:
    """Test qwen_original / qwen_quantized (TEXT_INPUT → TEXT_OUTPUT)."""
    import numpy as np
    import tritonclient.http as httpclient

    result = {"status": "not_ready", "latency_ms": None, "output_len": None, "output_preview": ""}
    try:
        if not client.is_model_ready(model_name):
            result["status"] = "not_ready"
            return result

        input_data = np.array([[prompt.encode("utf-8")]], dtype=object)
        infer_input = httpclient.InferInput("TEXT_INPUT", [1, 1], "BYTES")
        infer_input.set_data_from_numpy(input_data)

        t0 = time.perf_counter()
        response = client.infer(model_name, [infer_input])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        output = response.as_numpy("TEXT_OUTPUT")[0][0].decode("utf-8")
        result.update(status="ok", latency_ms=round(elapsed_ms, 2),
                      output_len=len(output), output_preview=output[:120])
    except Exception as e:
        result["status"] = f"error: {e}"
    return result


def test_onnx_model(client, model_name: str, prompt: str) -> dict:
    """Test qwen_onnx / qwen_onnx_optimized.

    The model was exported with seq_len=1 (single-step autoregressive inference),
    so we send exactly 1 token (BOS) to verify the forward pass works.
    """
    import numpy as np
    import tritonclient.http as httpclient
    from transformers import AutoTokenizer

    result = {"status": "not_ready", "latency_ms": None, "output_len": None, "output_preview": ""}
    try:
        if not client.is_model_ready(model_name):
            result["status"] = "not_ready"
            return result

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct", local_files_only=True
        )
        bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id or 0
        input_ids = np.array([[bos_id]], dtype=np.int64)       # shape [1, 1]
        attention_mask = np.ones((1, 1), dtype=np.int64)       # shape [1, 1]

        inp_ids = httpclient.InferInput("input_ids", list(input_ids.shape), "INT64")
        inp_ids.set_data_from_numpy(input_ids)
        inp_mask = httpclient.InferInput("attention_mask", list(attention_mask.shape), "INT64")
        inp_mask.set_data_from_numpy(attention_mask)

        t0 = time.perf_counter()
        response = client.infer(model_name, [inp_ids, inp_mask])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        logits = response.as_numpy("logits")
        next_token_id = int(logits[0, -1, :].argmax())
        next_token = tokenizer.decode([next_token_id])
        result.update(status="ok", latency_ms=round(elapsed_ms, 2),
                      output_len=1, output_preview=f"next_token={next_token!r} (logits shape={logits.shape})")
    except Exception as e:
        result["status"] = f"error: {e}"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triton", default=TRITON_URL)
    parser.add_argument("--preprocess", default=PREPROCESS_URL)
    parser.add_argument("--prompt", default=TEST_PROMPT)
    parser.add_argument("--output", type=Path, default=Path("test_report.json"))
    args = parser.parse_args()

    report = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "prompt": args.prompt, "preprocessing": {}, "models": {}}

    print("=" * 60)
    print("Testing preprocessing service...")
    preprocess_result = test_preprocessing(args.preprocess)
    report["preprocessing"] = preprocess_result
    print(f"  Status: {preprocess_result['status']}  "
          f"Latency: {preprocess_result.get('latency_ms')} ms  "
          f"Tokens: {preprocess_result.get('token_count')}")

    print("Testing Triton models...")
    try:
        import tritonclient.http as httpclient
    except ImportError:
        print("ERROR: pip install tritonclient[http]")
        import sys; sys.exit(1)

    try:
        client = httpclient.InferenceServerClient(url=args.triton, verbose=False)
        assert client.is_server_live(), f"Triton not reachable at {args.triton}"
        print(f"  Triton server live at {args.triton}")
    except Exception as e:
        print(f"ERROR: {e}")
        import sys; sys.exit(1)

    onnx_models = {"qwen_onnx", "qwen_onnx_optimized"}
    for model_name in MODELS:
        print(f"\n  Testing {model_name}...")
        if model_name in onnx_models:
            result = test_onnx_model(client, model_name, args.prompt)
        else:
            result = test_python_backend_model(client, model_name, args.prompt)
        report["models"][model_name] = result
        print(f"    Status:  {result['status']}")
        if result["latency_ms"]:
            print(f"    Latency: {result['latency_ms']} ms")
            print(f"    Output:  {result['output_preview']!r}")

    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'Status':<20} {'Latency (ms)':>14} {'Output len':>12}")
    print("-" * 75)
    for m, r in report["models"].items():
        lat = f"{r['latency_ms']:.1f}" if r["latency_ms"] else "N/A"
        olen = str(r["output_len"]) if r["output_len"] else "N/A"
        print(f"{m:<25} {r['status']:<20} {lat:>14} {olen:>12}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
