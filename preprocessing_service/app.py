"""Preprocessing microservice for Qwen2.5-3B-Instruct (text-only).

Formats raw user prompts into the chat template expected by Qwen before
sending to Triton Inference Server.

Endpoints:
    POST /preprocess   — JSON {prompt, system?} → {text, token_count}
    GET  /health       — {"status": "ok"}
    GET  /metrics      — Prometheus text format
"""
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

app = FastAPI(title="Qwen Preprocessing Service", version="1.0.0")

tokenizer = None


@app.on_event("startup")
def load_tokenizer():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


REQUEST_COUNT = Counter(
    "preprocessing_requests_total", "Total requests", ["status"]
)
REQUEST_LATENCY = Histogram(
    "preprocessing_latency_seconds", "Request latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
)
TOKEN_COUNT = Histogram(
    "preprocessing_token_count", "Token count per request",
    buckets=[16, 32, 64, 128, 256, 512, 1024],
)


class PreprocessRequest(BaseModel):
    prompt: str
    system: str = "You are a helpful assistant."


class PreprocessResponse(BaseModel):
    text: str        # formatted chat string ready for Triton
    token_count: int


@app.post("/preprocess", response_model=PreprocessResponse)
def preprocess(req: PreprocessRequest):
    t0 = time.perf_counter()
    try:
        messages = [
            {"role": "system", "content": req.system},
            {"role": "user",   "content": req.prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        token_count = len(tokenizer.encode(text))
        TOKEN_COUNT.observe(token_count)
        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(time.perf_counter() - t0)
        return PreprocessResponse(text=text, token_count=token_count)
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
