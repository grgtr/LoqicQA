# LogicQA — Logical Anomaly Detection with VLM-Generated Questions

Reproduction of the paper:  
**"LogicQA: Logical Anomaly Detection with Vision Language Model Generated Questions"** ([arXiv:2503.20252](https://arxiv.org/abs/2503.20252))

LogicQA is a **training-free, annotation-free** few-shot logical anomaly detection framework. Given a few normal images, it automatically generates a checklist of binary Yes/No questions and uses a Vision Language Model (VLM) to answer them for test images, flagging any constraint violation as a logical anomaly.

---

## Framework Overview

```
[3 normal images]
       │
  Stage 1: Describe          ← VLM describes each normal image
       │
  Stage 2: Summarize         ← VLM distills shared normality patterns
       │
  Stage 3: Generate Qs       ← VLM generates & filters main questions
            + Sub-Qs             + generates 5 semantic variants each
       │
  Stage 4: Test              ← For each test image:
       │                         answer Sub-Qs → majority vote per Main-Q
       │                         → anomaly if ANY Main-Q votes "No"
       └──► anomaly score    ← log-prob based score for AUROC / F1-max
```

---

## Repository Structure

```
LogicQA/
├── config.yaml                     # Global configuration
├── requirements.txt
├── logicqa/
│   ├── config.py                   # Dataclass config + YAML loading
│   ├── vlm/
│   │   ├── base.py                 # Abstract VLMBase + VLMResponse
│   │   ├── internvl.py             # InternVL-2.5 local backend ★
│   │   ├── gpt4o.py                # GPT-4o via OpenAI API
│   │   ├── gemini.py               # Gemini-1.5-Flash
│   │   └── factory.py              # get_vlm() factory
│   ├── prompts/
│   │   └── __init__.py             # All 5 prompt templates
│   ├── preprocessing/
│   │   ├── bpm.py                  # Back Patch Masking
│   │   └── langsam_wrapper.py      # Lang-SAM segmentation
│   ├── pipeline/
│   │   ├── stage1_describe.py
│   │   ├── stage2_summarize.py
│   │   ├── stage3_questions.py
│   │   ├── stage4_test.py
│   │   └── logicqa.py              # LogicQAPipeline (orchestrator)
│   ├── data/
│   │   ├── mvtec_loco.py           # Dataset loader + download via dataset_tools
│   │   └── normality_definitions.py # Per-class normality constraints
│   └── evaluation/
│       └── metrics.py              # AUROC, F1-max, binary F1
├── scripts/
│   └── run_pipeline.py             # Full evaluation CLI
└── tests/
    ├── test_pipeline_smoke.py      # Mock-VLM end-to-end tests
    └── test_preprocessing.py       # BPM and normality definition tests
```

---

## Installation

```bash
# Clone and install dependencies
pip install -r requirements.txt

# Optional: for InternVL-2.5 local inference
# Requires 3× A100 GPU (38B model) or 1× A100 (8B model)
# model is auto-downloaded from HuggingFace on first run

# Optional: for Lang-SAM preprocessing
pip install lang-sam
```

---

## Dataset Setup

The MVTec LOCO AD dataset is automatically downloaded via `dataset_tools` on first run.  
You can also trigger it manually:

```python
import dataset_tools as dtools
dtools.download(dataset='MVTec LOCO AD', dst_dir='~/dataset-ninja/')
```

Or via the CLI (auto-triggered when `--data_dir` does not yet contain the dataset):

```bash
python scripts/run_pipeline.py --class_name breakfast_box --data_dir ~/dataset-ninja/
```

---

## Quick Start

### 1. Set environment variables (if using API-based VLMs)

```bash
export OPENAI_API_KEY=sk-...    # for GPT-4o
export GOOGLE_API_KEY=...       # for Gemini
```

### 2. Run on one class

```bash
# Using InternVL-2.5 (default, local)
python scripts/run_pipeline.py \
    --class_name breakfast_box \
    --vlm internvl \
    --data_dir ~/dataset-ninja/ \
    --n_shots 3 \
    --output_dir results/ \
    --save_questions

# Using GPT-4o
python scripts/run_pipeline.py \
    --class_name pushpins \
    --vlm gpt4o \
    --data_dir ~/dataset-ninja/ \
    --output_dir results/
```

### 3. Python API

```python
from logicqa import LogicQAPipeline, LogicQAConfig

cfg = LogicQAConfig.from_yaml("config.yaml")
pipeline = LogicQAPipeline(cfg)

# Setup (Stages 1-3) — run once per class
pipeline.setup(
    class_name="breakfast_box",
    normal_images=["img1.png", "img2.png", "img3.png"],
)

# Save generated questions for reuse
pipeline.save_questions("results/breakfast_box_questions.json")

# Predict (Stage 4)
result = pipeline.predict("test_image.png")
print(result.is_anomaly)       # True / False
print(result.anomaly_score)    # float in [0, 1]
print(result.explanation)      # natural language explanation
```

---

## Preprocessing

Per the paper (Appendix F):

| Class                 | BPM | Lang-SAM |
|-----------------------|:---:|:--------:|
| Breakfast Box         |     |          |
| Juice Bottle          |     |          |
| Pushpins              |     |    ✓     |
| Screw Bag             |  ✓  |          |
| Splicing Connectors   |  ✓  |    ✓     |

These are applied automatically when using `LogicQAPipeline`.

---

## Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## VLM Configuration

Edit `config.yaml` to switch backends or change hyperparameters:

```yaml
vlm:
  backend: "internvl"   # or "gpt4o" / "gemini"
  internvl:
    model_name: "OpenGVLab/InternVL2_5-8B"   # lighter 8B model
    temperature: 0.2
```

---

## Results (from paper, Table 1)

| Class                | AUROC | F1-max |
|----------------------|-------|--------|
| Breakfast Box        | 88.2% | 87.5%  |
| Juice Bottle         | 89.3% | 88.1%  |
| Pushpins             | 85.4% | 84.9%  |
| Screw Bag            | 87.1% | 86.8%  |
| Splicing Connectors  | 87.8% | 87.7%  |
| **Average**          | **87.6%** | **87.0%** |

(Achieved with GPT-4o. InternVL-2.5-38B performance is comparable.)

---
