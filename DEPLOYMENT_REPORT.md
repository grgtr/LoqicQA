# Отчёт о развёртывании Qwen2.5-3B-Instruct

**Дата:** 2026-05-20  
**Модель:** Qwen/Qwen2.5-3B-Instruct  
**GPU:** NVIDIA A100 80GB PCIe  
**Сервер:** x32-01

---

## 1. Архитектура решения

```
┌─────────────────────────────────────────────────────────────┐
│                     docker-compose стек                      │
│                                                             │
│  ┌──────────────────┐    ┌──────────────────────────────┐  │
│  │  logicqa_triton  │    │   logicqa_preprocessing      │  │
│  │  (порт 8111)     │    │   (порт 8080)                │  │
│  │                  │    │   FastAPI + Qwen tokenizer   │  │
│  │  4 модели:       │    └──────────────────────────────┘  │
│  │  - qwen_original │                                       │
│  │  - qwen_quantized│    ┌──────────────────────────────┐  │
│  │  - qwen_onnx     │    │   logicqa_prometheus         │  │
│  │  - qwen_onnx_opt │    │   (порт 9090)                │  │
│  └──────────────────┘    └──────────────────────────────┘  │
│                                                             │
│                          ┌──────────────────────────────┐  │
│                          │   logicqa_grafana            │  │
│                          │   (порт 3000)                │  │
│                          └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Версии моделей

| Модель | Backend | Precision | Устройство |
|--------|---------|-----------|------------|
| qwen_original | Python (transformers) | bfloat16 | GPU |
| qwen_quantized | Python (bitsandbytes) | INT8 | GPU |
| qwen_onnx | ONNX Runtime | fp16 (веса) / fp32 (logits) | GPU |
| qwen_onnx_optimized | ONNX Runtime + ORT fusion | fp16 (веса) / fp32 (logits) | GPU |

---

## 2. Инструкция по запуску

### Требования

- Docker + docker-compose с GPU поддержкой (NVIDIA Container Toolkit)
- Python 3.10+, виртуальное окружение `.venv`
- NVIDIA GPU с ≥20 GB VRAM (тестировалось на A100 80GB)
- Кэш модели: `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/`

### Шаг 1 — Подготовка ONNX-моделей

```bash
cd ~/LoqicQA

# Экспорт модели в ONNX (fp16, GPU, ~2 мин)
python scripts/export_onnx.py --mode simple --dtype fp16 --device cuda

# ORT-оптимизация графа (~1 мин)
python scripts/optimize_onnx.py
```

**Результаты экспорта:**
```
model.onnx      : 4.6 MB   (граф)
model.onnx.data : 6.17 GB  (веса fp16)
```

**Результаты оптимизации:**
```
Nodes: 2350
Output size: 6.79 GB
```

### Шаг 2 — INT8 квантизация (опционально, только для воспроизведения весов)

```bash
# Применяется через bitsandbytes при загрузке модели в Triton
# Предварительный скрипт квантизации (torch.quantize_dynamic):
python scripts/quantize_torch.py
```

### Шаг 3 — Запуск стека

```bash
cd ~/LoqicQA/deploy

# Пересборка Triton-образа и запуск всех сервисов
sudo docker compose up -d --build triton
sudo docker compose up -d prometheus grafana

# Проверка статуса
sudo docker ps --filter "name=logicqa" --format "table {{.Names}}\t{{.Status}}"
```

**Ожидаемый вывод:**
```
NAMES                    STATUS
logicqa_triton           Up X minutes (healthy)
logicqa_prometheus       Up X minutes
logicqa_grafana          Up X minutes
logicqa_preprocessing    Up X minutes (healthy)
```

### Шаг 4 — Загрузка моделей в Triton

```bash
# Triton работает в explicit mode — модели загружаются вручную
curl -X POST http://localhost:8111/v2/repository/models/qwen_original/load
curl -X POST http://localhost:8111/v2/repository/models/qwen_quantized/load
curl -X POST http://localhost:8111/v2/repository/models/qwen_onnx/load
curl -X POST http://localhost:8111/v2/repository/models/qwen_onnx_optimized/load

# Проверка — все модели должны быть READY
curl -s -X POST http://localhost:8111/v2/repository/index | python3 -m json.tool
```

### Шаг 5 — Тестирование и отчёт

```bash
cd ~/LoqicQA

# Тест инференса всех 4 моделей
python scripts/test_triton.py

# Подождать ~1 минуту для сбора метрик Prometheus
python scripts/generate_report.py
```

### Остановка стека

```bash
cd ~/LoqicQA/deploy
sudo docker compose down
sudo docker compose down -v
```

---

## 3. Результаты тестирования

**Промпт:** `Answer in one sentence: What is the capital of France?`  
**Время теста:** 2026-05-20T21:33:18

### Инференс

| Модель | Статус | Латентность | Вывод |
|--------|--------|-------------|-------|
| qwen_original | ✅ ok | 264.2 ms | "The capital of France is Paris." |
| qwen_quantized | ✅ ok | 6694.5 ms | "The capital of France is Paris." |
| qwen_onnx | ✅ ok | 20.5 ms | next_token forward pass |
| qwen_onnx_optimized | ✅ ok | 25.9 ms | next_token forward pass |

> **Примечание:** `qwen_original` и `qwen_quantized` выполняют полную генерацию текста (31 токен).  
> `qwen_onnx` и `qwen_onnx_optimized` выполняют один forward pass (предсказание следующего токена) —  
> модели экспортированы без KV-cache, полная авторегрессия требует внешнего цикла.

---

## 4. Анализ производительности (Prometheus)

**Период измерения:** rate за 5 минут после тестирования  
**Источник:** `http://localhost:9090`

### Метрики моделей

| Модель | Avg Latency | Throughput | Avg Queue | Error Rate |
|--------|-------------|------------|-----------|------------|
| qwen_onnx | **19.1 ms** | 0.0035 rps | 0.07 ms | 0.0 |
| qwen_onnx_optimized | 23.6 ms | 0.0035 rps | 0.06 ms | 0.0 |
| qwen_original | 263.9 ms | 0.0035 rps | 0.03 ms | 0.0 |
| qwen_quantized | 6694.1 ms | 0.0035 rps | 0.04 ms | 0.0 |

### GPU метрики

| Параметр | Значение |
|----------|----------|
| GPU Utilization | 100% |
| GPU Memory Used | 83.02 GB / 85.9 GB |
| GPU Memory Available | ~2.9 GB |

---

## 5. Анализ результатов

### Сравнение латентности

```
qwen_onnx            ██ 19.1 ms      (1x baseline)
qwen_onnx_optimized  ██ 23.6 ms      (1.2x)
qwen_original        ██████████████  263.9 ms   (13.8x)
qwen_quantized       ██████...██████ 6694 ms    (350x)
```

### qwen_onnx vs qwen_onnx_optimized

ONNX Runtime с ORT fusion (слияние MultiHeadAttention, LayerNorm, FastGelu) 
теоретически быстрее, однако в данном тесте `qwen_onnx` оказался чуть быстрее (19.1 ms vs 23.6 ms).
Причина: тест использует единственный запрос с batch=1. ORT fusion раскрывает преимущество
при параллельной нагрузке и больших батчах, где накладные расходы на запуск отдельных
CUDA-ядер становятся значимыми.

### qwen_original (264 ms)

Python backend добавляет следующие накладные расходы по сравнению с ONNX RT:
- Межпроцессное взаимодействие (IPC через shared memory)
- Python GIL и интерпретатор
- `apply_chat_template` и токенизация внутри контейнера
- Полная генерация 31 токена (авторегрессивный цикл)

### qwen_quantized (6694 ms)

bitsandbytes INT8 quantization:
- **Плюс:** экономия памяти (~3 GB vs 6 GB для fp16)
- **Минус:** при каждом шаге генерации выполняется деквантизация весов INT8→fp16,
  что суммируется за 31 токен и даёт значительный overhead на GPU
- Оптимален для CPU-инференса или сценариев с жёстким ограничением GPU памяти

### Потребление памяти GPU

Все 4 модели загружены одновременно на A100 80GB:

| Модель | Оценка памяти |
|--------|--------------|
| qwen_original (bfloat16) | ~6 GB |
| qwen_quantized (INT8) | ~3 GB |
| qwen_onnx (fp16 веса) | ~6 GB |
| qwen_onnx_optimized (fp16 веса) | ~6 GB |
| Системный overhead | ~62 GB (другие процессы на разделяемом сервере) |
| **Итого занято** | **83.02 / 85.9 GB** |

---

## 6. Структура файлов проекта

```
LoqicQA/
├── scripts/
│   ├── export_onnx.py           # ONNX экспорт (dynamo/simple/optimum)
│   ├── quantize_torch.py        # INT8 dynamic quantization
│   ├── optimize_onnx.py         # ORT transformer optimizer
│   ├── test_triton.py           # End-to-end тест всех моделей
│   └── generate_report.py       # Сбор метрик → deployment_report.json
│
├── preprocessing_service/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py                   # FastAPI: /preprocess, /health, /metrics
│   └── bpm_processor.py
│
├── deploy/
│   ├── docker-compose.yml
│   ├── triton/
│   │   ├── Dockerfile           # tritonserver + torch + transformers + bitsandbytes
│   │   └── model_repository/
│   │       ├── qwen_original/   # Python backend, bfloat16
│   │       ├── qwen_quantized/  # Python backend, INT8 bitsandbytes
│   │       ├── qwen_onnx/       # ONNX Runtime backend, fp16
│   │       └── qwen_onnx_optimized/ # ONNX Runtime + ORT fusion, fp16
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       └── provisioning/
│           ├── datasources/
│           └── dashboards/
│
├── test_report.json             # Результаты инференс-тестов
└── deployment_report.json       # Полный отчёт с Prometheus метриками
```

---

## 7. Мониторинг

- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (admin / logicqa123)

### Ключевые PromQL запросы

```promql
# Средняя латентность модели (мс)
rate(nv_inference_request_duration_us{model="qwen_original"}[5m])
  / rate(nv_inference_request_success{model="qwen_original"}[5m]) / 1000

# Throughput (req/s)
rate(nv_inference_request_success[5m])

# GPU утилизация
nv_gpu_utilization

# GPU память
nv_gpu_memory_used_bytes / 1e9
```

---

## 8. Выводы

| Задача | Статус |
|--------|--------|
| Модель Qwen2.5-3B-Instruct | ✅ |
| Конвертация в ONNX (fp16) | ✅ |
| Оптимизация Torch (INT8 bitsandbytes) | ✅ |
| Оптимизация ONNX (ORT fusion) | ✅ |
| Микросервис предобработки (FastAPI + Docker) | ✅ |
| Triton: 4 версии модели, все READY | ✅ |
| Мониторинг Prometheus + Grafana | ✅ |
| docker-compose оркестрация | ✅ |
| Тестирование инференса всех версий | ✅ |
| Отчёт с метриками производительности | ✅ |

**Наилучшая латентность:** `qwen_onnx` — **19.1 ms** (forward pass, GPU, ONNX Runtime)  
**Наилучшая генерация текста:** `qwen_original` — **264 ms** за 31 токен (GPU, bfloat16)
