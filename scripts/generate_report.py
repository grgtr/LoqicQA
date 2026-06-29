"""Collect Prometheus metrics and merge with Triton test results to generate a deployment report.

Queries Prometheus for key inference metrics, reads test_report.json,
and writes deployment_report.json with a complete performance summary.

Usage:
    python scripts/generate_report.py [--prometheus PROMETHEUS_URL] [--test-report PATH]
"""
import argparse
import json
import time
from pathlib import Path


PROMETHEUS_URL = "http://localhost:9090"
MODELS = ["qwen_original", "qwen_quantized", "qwen_onnx", "qwen_onnx_optimized"]


def query_prometheus(base_url: str, promql: str) -> list:
    import urllib.request
    import urllib.parse

    url = f"{base_url}/api/v1/query?query={urllib.parse.quote(promql)}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("data", {}).get("result", [])
    except Exception as e:
        return []


def get_model_metrics(base_url: str, model: str) -> dict:
    metrics = {}

    # Average latency (microseconds → milliseconds)
    latency_res = query_prometheus(
        base_url,
        f'rate(nv_inference_request_duration_us{{model="{model}"}}[5m]) / '
        f'rate(nv_inference_request_success{{model="{model}"}}[5m])',
    )
    if latency_res:
        val = float(latency_res[0]["value"][1])
        metrics["avg_latency_ms"] = round(val / 1000, 2)

    # Throughput (req/s)
    throughput_res = query_prometheus(
        base_url,
        f'rate(nv_inference_request_success{{model="{model}"}}[5m])',
    )
    if throughput_res:
        metrics["throughput_rps"] = round(float(throughput_res[0]["value"][1]), 4)

    # Queue latency
    queue_res = query_prometheus(
        base_url,
        f'rate(nv_inference_queue_duration_us{{model="{model}"}}[5m]) / '
        f'rate(nv_inference_request_success{{model="{model}"}}[5m])',
    )
    if queue_res:
        val = float(queue_res[0]["value"][1])
        metrics["avg_queue_ms"] = round(val / 1000, 2)

    # Error rate
    err_res = query_prometheus(
        base_url,
        f'rate(nv_inference_request_failure{{model="{model}"}}[5m])',
    )
    if err_res:
        metrics["error_rate"] = round(float(err_res[0]["value"][1]), 6)

    return metrics


def get_gpu_metrics(base_url: str) -> dict:
    metrics = {}

    util_res = query_prometheus(base_url, "nv_gpu_utilization")
    if util_res:
        metrics["gpu_utilization_pct"] = round(float(util_res[0]["value"][1]), 1)

    mem_used_res = query_prometheus(base_url, "nv_gpu_memory_used_bytes")
    if mem_used_res:
        metrics["gpu_memory_used_gb"] = round(float(mem_used_res[0]["value"][1]) / 1e9, 2)

    mem_total_res = query_prometheus(base_url, "nv_gpu_memory_total_bytes")
    if mem_total_res:
        metrics["gpu_memory_total_gb"] = round(float(mem_total_res[0]["value"][1]) / 1e9, 2)

    return metrics


def get_preprocessing_metrics(base_url: str) -> dict:
    metrics = {}

    rps_res = query_prometheus(
        base_url, 'rate(preprocessing_requests_total{status="success"}[5m])'
    )
    if rps_res:
        metrics["success_rps"] = round(float(rps_res[0]["value"][1]), 4)

    err_res = query_prometheus(
        base_url, 'rate(preprocessing_requests_total{status="error"}[5m])'
    )
    if err_res:
        metrics["error_rps"] = round(float(err_res[0]["value"][1]), 4)

    p50_res = query_prometheus(
        base_url,
        "histogram_quantile(0.50, rate(preprocessing_latency_seconds_bucket[5m]))",
    )
    if p50_res:
        metrics["p50_latency_ms"] = round(float(p50_res[0]["value"][1]) * 1000, 2)

    p95_res = query_prometheus(
        base_url,
        "histogram_quantile(0.95, rate(preprocessing_latency_seconds_bucket[5m]))",
    )
    if p95_res:
        metrics["p95_latency_ms"] = round(float(p95_res[0]["value"][1]) * 1000, 2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Generate LogicQA deployment report")
    parser.add_argument("--prometheus", default=PROMETHEUS_URL)
    parser.add_argument("--test-report", type=Path, default=Path("test_report.json"))
    parser.add_argument("--output", type=Path, default=Path("deployment_report.json"))
    args = parser.parse_args()

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prometheus_url": args.prometheus,
        "models": {},
        "gpu": {},
        "preprocessing": {},
    }

    # Load Triton test results if available
    if args.test_report.exists():
        with open(args.test_report) as f:
            test_data = json.load(f)
        report["test_results"] = test_data
        print(f"Loaded test results from {args.test_report}")
    else:
        print(f"No test report found at {args.test_report} — skipping")

    # Query Prometheus
    print(f"Querying Prometheus at {args.prometheus}...")
    for model in MODELS:
        m = get_model_metrics(args.prometheus, model)
        report["models"][model] = m
        if m:
            print(f"  {model}: {m}")
        else:
            print(f"  {model}: no data (model may not have been queried yet)")

    report["gpu"] = get_gpu_metrics(args.prometheus)
    print(f"  GPU: {report['gpu']}")

    report["preprocessing"] = get_preprocessing_metrics(args.prometheus)
    print(f"  Preprocessing: {report['preprocessing']}")

    # Summary table
    print("\nModel Performance Summary:")
    print(f"{'Model':<25} {'Avg Latency (ms)':>18} {'Throughput (rps)':>18}")
    print("-" * 65)
    for m, metrics in report["models"].items():
        lat = f"{metrics.get('avg_latency_ms', 'N/A')}"
        tput = f"{metrics.get('throughput_rps', 'N/A')}"
        print(f"{m:<25} {lat:>18} {tput:>18}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nDeployment report saved to {args.output}")


if __name__ == "__main__":
    main()
