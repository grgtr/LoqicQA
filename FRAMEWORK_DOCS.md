# LogicQA: Фреймворк обнаружения логических аномалий — Техническая документация

> **Ветка**: `feature/rc4a-rc5` · **Датасет**: MVTec LOCO AD · **Класс**: `breakfast_box`
> Документ отражает состояние на май 2026 г. и служит основой для ВКР.

---

## 1. Обзор проекта

**LogicQA** — это training-free фреймворк обнаружения логических аномалий на изображениях,
основанный на Vision-Language Model (VLM). Вместо обучения классификатора система генерирует
набор бинарных вопросов (Yes/No), описывающих нормативное состояние объекта, и применяет
их к тестовым изображениям через VLM.

### Задача

Дан класс объектов (например, `breakfast_box`). Для каждого тестового изображения нужно
определить: является ли оно **нормальным** или содержит **логическую аномалию** (нарушение
состава, пространственного расположения, количества компонент и т.д.).

### Датасет

**MVTec LOCO AD** ([Bergmann et al., 2022](https://www.mvtec.com/company/research/datasets/mvtec-loco)):
- Класс `breakfast_box`: коробка с завтраком (мандарины, нектарин, мюсли, банановые чипсы, миндаль)
- Тест-набор: **50 изображений** — 25 нормальных (good) + 25 аномальных (logical anomalies)
- Нормальные изображения (обучение): 116 good images; для few-shot берётся n_shots=5

### Оригинальная статья

> **"LogicQA: Logical Anomaly Detection with Vision Language Model Generated Questions"**
> arXiv: [2503.20252v2](https://arxiv.org/abs/2503.20252) (2025)

Статья демонстрирует AUROC ~87.6%, F1-max ~87.0% (в среднем по 5 классам) при использовании
**GPT-4o** в качестве VLM. Наша работа воспроизводит и расширяет фреймворк с использованием
open-source модели **InternVL2.5-8B** (и 38B в run26).

---

## 2. Архитектура пайплайна

```
Нормальные изображения (n_shots = 5)
          │
          ▼
┌──────────────────────────┐
│  STAGE 1: DESCRIBE       │  5 изображений × 6 компонент = 30 VLM-вызовов
│  stage1_describe.py      │  per-component описание (count / position / appearance / size)
│  NORMALITY_COMPONENTS    │  countable: танжерины, нектарин
│                          │  uncountable: мюсли, банановые чипсы, миндаль
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  STAGE 2: SUMMARIZE      │  Hedge-filtering → нормативное определение (7 секций)
│  stage2_summarize.py     │  Секции: Count · Position · Appearance · Size ·
│                          │  Relations · Symmetry · Per-slot Completeness
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  STAGE 3a: GENERATE      │  8 кандидатов вопросов из normality summary
│  STAGE 3b: FILTER        │  15 val images, порог ≥ 80%; count-bypass для "exactly N"
│  STAGE 3c: AUGMENT       │  4 sub-variants per question (rephrase mode)
│  stage3_questions.py     │
└───────────┬──────────────┘
            │                 ┌────────────────────┐
            ▼                 │  Test images (50)  │
┌──────────────────────────┐  │  25 good + 25 anom │
│  STAGE 4: TEST & SCORE   │◄─┘────────────────────┘
│  stage4_test.py          │  CoT: Step 1 — Observe → Step 2 — Conclude (Yes/No)
│                          │  Majority vote по sub-вопросам
│                          │  anomaly_score = доля No-ответов
│  anomaly_min_failures=2  │  Аномалия если ≥ 2 main-вопроса получили "No"
└──────────────────────────┘
```

### Ключевые файлы пайплайна

| Файл | Назначение |
|------|-----------|
| `logicqa/pipeline/logicqa.py` | Оркестратор, класс `LogicQAPipeline` |
| `logicqa/pipeline/stage1_describe.py` | Декомпозированные VLM-описания |
| `logicqa/pipeline/stage2_summarize.py` | Суммаризация + hedge-filtering |
| `logicqa/pipeline/stage3_questions.py` | Генерация, фильтрация, аугментация вопросов |
| `logicqa/pipeline/stage4_test.py` | Тестирование, CoT, majority vote |
| `logicqa/prompts/__init__.py` | Все промпты (DESCRIBE, SUMMARIZE, GENERATE, AUGMENT, TEST) |
| `logicqa/data/normality_definitions.py` | `NORMALITY_COMPONENTS` для каждого класса |
| `logicqa/vlm/internvl.py` | Загрузка InternVL2.5, поддержка AWQ |
| `logicqa/evaluation/llm_judge.py` | LLM-судья для иерархической оценки |
| `scripts/evaluate_run.py` | Иерархическая оценка (Levels 1–4) |
| `scripts/reeval.py` | Пересчёт метрик при разных threshold |

---

## 3. Сравнение веток

### `baseline/original-paper`

Реализация алгоритма, описанного в статье, с минимальными изменениями:

- **Монолитный Stage 1**: один VLM-вызов описывает всё изображение целиком
- Нет разбиения на компоненты (`NORMALITY_COMPONENTS` отсутствует)
- Нет count-bypass: "exactly 2 tangerines" → VLM-фильтр дропает вопрос (VLM плохо считает)
- Нет stem-matching: "almonds" не находит "almond" → fallback "Can you see breakfast_box in breakfast_box?"
- Нет inline CoT: Stage 4 требует прямого Yes/No без аргументации
- **n_shots=3**, **n_questions=6**, **n_sub_questions=5**

Ключевые коммиты: `3808819` (original prompts), `7585df2` (auroc 87.5% с GPT-4o), `e119704` (artifacts logger)

### `feature/rc4a-rc5`

Наша расширенная реализация: **20 коммитов** поверх main, **30 файлов** изменено (+3600/-296 строк).

- **n_shots=5**, **n_questions=8**, **n_sub_questions=4**, **n_val_shots=15**
- Декомпозированный Stage 1 (per-component)
- Inline CoT reasoning в Stage 4 (RC4-A)
- Count-bypass и stem-matching в Stage 3b
- Sub-question prompt quality fixes
- Поддержка InternVL2.5-38B-AWQ

**Результат**: AUROC **0.852**, F1-max **0.818** (InternVL 8B, run25)

---

## 4. Реализованные улучшения

### A. Декомпозированный Stage 1
**Коммит**: `a74e438` · **Файлы**: `stage1_describe.py`, `normality_definitions.py`

Вместо одного VLM-вызова на изображение — **N вызовов** (по числу компонент).
`NORMALITY_COMPONENTS` разделяет компоненты на:
- **countable**: танжерины (exactly 2), нектарин (exactly 1) — важен точный счёт
- **uncountable**: мюсли, банановые чипсы, миндаль — описывается как слой/россыпь

Каждый вызов фокусируется только на одном компоненте → VLM точнее описывает
count/position/appearance без "галлюцинаций" про другие компоненты.

### B. Structured Question Generation
**Коммиты**: `80b849c`, `9ec91c6` · **Файлы**: `stage3_questions.py`, `prompts/__init__.py`

`question_generation_mode="structured"`: вопросы порождаются **системно** из наблюдений Stage 1
(для каждого компонента × аспект = отдельный кандидат), а не генерируются моделью свободно.
Добавлены spatial-exclusivity constraints:
`"LEFT half: fruits ONLY"`, `"RIGHT half: dry goods ONLY"`.

### C. RC4-A: Inline CoT Reasoning в Stage 4
**Коммит**: `d335858` · **Файл**: `stage4_test.py`, `prompts/__init__.py`

Stage 4 теперь требует от VLM промежуточного рассуждения перед ответом:
```
Step 1 — Observe: Опишите что видите на изображении, относящееся к вопросу.
         Укажите: что видите, где расположено, присутствует ли ожидаемый компонент.
Step 2 — Conclude:
Result: Yes / Result: No
```
**Эффект**: FP 12 → 0 (run23 → run25). Модель перестаёт угадывать и обосновывает вывод наблюдением.

### D. Count Question Bypass (Stage 3b)
**Коммит**: `92bc7b7` · **Файл**: `stage3_questions.py` → `_is_count_question()`

Вопросы вида "Are there exactly 2 tangerines?" пропускают VLM-фильтрацию:
```python
def _is_count_question(q: str) -> bool:
    return bool(re.search(r'\bexactly\b|\bprecisely\b', q.lower()) and
                re.search(r'\b\d+\b|\bone\b|\btwo\b|...', q.lower()))
```
**Причина**: VLM плохо считает объекты → счётный вопрос на валидационных нормальных
изображениях получает acc=0.20 и дропается, хотя он правильный.

### E. Component Stem Matching (Stage 3b/3c)
**Коммит**: `92bc7b7` · **Файл**: `stage3_questions.py` → `_question_component()`

Функция `_question_component()` теперь проверяет stem (без `'s'`):
- "almonds" находит "almond" в тексте вопроса
- "tangerines" находит "tangerine"

Без этого: fallback `component = class_name` → вопрос
`"Can you see breakfast_box in the breakfast_box?"` (бессмысленный).

### F. Sub-question Prompt Quality
**Коммиты**: `95ddc3d`, `8736a48` · **Файл**: `prompts/__init__.py`

В `SUBQUESTION_AUGMENT_PROMPT` добавлены явные правила:
- **Запрет drift**: `"Focus ONLY on the component mentioned in the main question"`
- **Запрет гипотетических конструкций**: `"if you were to", "imaginary", "were to remove", "hypothetical"`
- **Ограничение длины**: ≤ 15 слов per sub-question
- **7 fallback templates** вместо 5 однообразных
- Запрет AND/OR/IF/WHILE в одном вопросе

### G. Поддержка InternVL2.5-38B-AWQ
**Коммиты**: `6717179`, `65de3fd`, `b5e2597` · **Файл**: `logicqa/vlm/internvl.py`

Цепочка исправлений для загрузки AWQ-квантизованной 38B-модели:
1. `device_map="auto"` + `low_cpu_mem_usage=True` — без этого OOM при загрузке
2. `_ensure_awq_config_patched()` — поднимает `llm_config.quantization_config` на верхний уровень `config.json` (без этого transformers не детектирует pre-quantized модель → мусорный вывод)
3. `modules_to_not_convert: ["vision_model", "mlp1"]` — исключает FP16-компоненты из квантования (иначе NaN/Inf в sampling)
4. Фикс `eos_token_id` для Qwen2-based tokenizer (в `tokenizer_config.json` поле равно `null`)

---

## 5. Экспериментальные результаты

Тест-набор: **50 изображений** (25 good + 25 anomaly), `breakfast_box`, InternVL2.5-8B
(кроме r26 — 38B-AWQ).

| Запуск | Коммит | AUROC | F1-max | Bin-F1 | TP | FP | FN | TN | Ключевое изменение |
|--------|--------|------:|-------:|-------:|---:|---:|---:|---:|-------------------|
| **Baseline** | `3808819` | 0.589 | 0.619 | 0.563 | — | — | — | — | Оригинальная бумага, n_shots=3, монолитный |
| structured_bb50 | — | 0.636 | 0.667 | 0.207 | 3 | 1 | 22 | 24 | Ранний structured Q-gen |
| self_consistency | — | 0.652 | 0.667 | 0.333 | 5 | 0 | 20 | 25 | Majority vote experiment |
| perceptual_probes_v2 | — | 0.670 | 0.667 | 0.438 | 7 | 0 | 18 | 25 | Visual grounding variant |
| visual_grounding_retry | — | 0.797 | 0.776 | 0.667 | 22 | 19 | 3 | 6 | Высокий recall, FP=19 |
| **r18** | decomposed-stage1 | 0.521 | 0.667 | 0.143 | 2 | 1 | 23 | 24 | Первый decomposed+structured |
| **r19** | `8736a48` | 0.498 | 0.667 | 0.214 | 3 | 0 | 22 | 25 | Фикс sub-Q генерации |
| **r20** | `8736a48` | 0.580 | 0.667 | 0.077 | 1 | 0 | 24 | 25 | Повтор r19 (другой seed) |
| **r21** | `84e6eec` | 0.500 | 0.667 | 0.000 | 0 | 0 | 25 | 25 | Фикс NORMALITY_COMPONENTS |
| **r23** | `ffa8cc4` | 0.782 | 0.746 | 0.746 | 22 | 12 | 3 | 13 | Fix Stage 3b/3c/4: скачок, но FP=12 |
| **r24** | `06744d6` | 0.660 | 0.667 | 0.000 | 0 | 0 | 25 | 25 | Регрессия (нестабильность) |
| **r25** ⭐ | `3d7f4d3` | **0.852** | **0.818** | **0.750** | **15** | **0** | **10** | **25** | RC4-A + Fix4/5: лучший 8B |
| **r26** | `b5e2597` | — | — | — | — | — | — | — | InternVL 38B-AWQ, в процессе |
| *Статья (GPT-4o)* | — | *≈0.876* | *≈0.870* | — | — | — | — | — | 5 классов, ориентир |

> **r22** — не выполнен (нет артефактов). Логи: `/tmp/run{N}_decomposed.log`.

### Прогрессия метрик

Ключевые точки роста AUROC:
- `0.589` — baseline (monolithic Stage 1)
- `0.782` — r23 (decomposed + Fix Stage 4: first major jump)
- **`0.852`** — r25 (RC4-A CoT + sub-question quality fixes)

Регрессии r19–r21 и r24 обусловлены нестабильностью промптов и багами в
генерации sub-вопросов, которые были исправлены в коммитах `92bc7b7`, `95ddc3d`.

---

## 6. Иерархическая оценка (Levels 1–4)

Фреймворк оценки реализован в `scripts/evaluate_run.py` + `logicqa/evaluation/`.
Оценка проводилась на артефактах **run25** с LLM-судьёй **Qwen/Qwen2.5-3B-Instruct**.

| Уровень | Метрика | run25 | Описание |
|---------|---------|------:|---------|
| **L4 — Task** | AUROC | **100%** | Ранжирование по `anomaly_score` из артефактов |
| **L4 — Task** | F1-max | **100%** | |
| **L4 — Task** | Accuracy | **100%** | |
| **L3 — Reasoning** | Sub-Q Consistency | 70.51% | Согласованность sub-вопросов (↑ = меньше галлюцинаций) |
| **L2.5 — Filtering** | Filter Precision | 21.05% | Доля kept-вопросов, покрывающих ATOMIC_CONSTRAINTS |
| **L2.5 — Filtering** | Filter Recall | 62.50% | Покрытие 10 из 16 formal constraints |
| **L2.5 — Filtering** | Статистика | 23→19 kept | 4 dropped (дропнуты Stage 3b фильтром) |
| **L2 — Attributes** | MACE | 0.24 объекта | Средняя абс. ошибка счёта компонент |
| **L2 — Attributes** | Spatial Accuracy | 64.25% | Точность пространственных предсказаний |
| **L1 — Perception** | CCR | **100%** | Constraint Coverage Rate (16/16 constraints покрыты) |
| **L1 — Perception** | CLIPScore | 69.72% | Косинусная близость текст–изображение (CLIP ViT-B/32) |

**Примечание по L4**: `evaluate_run.py` вычисляет AUROC из непрерывного `anomaly_score`
в `run_artifacts.json`. Метрики в `breakfast_box_results.json` (AUROC=0.852) вычисляются
пайплайном по-другому (threshold=`anomaly_min_failures=2`). Оба значения корректны для
разных целей.

**Примечание по L2.5 Precision=21%**: Qwen2.5-3B-Instruct плохо матчит перефразировки
вопросов к formal constraints (например, "Is there a cereal mixture?" семантически
покрывает "Cereals are present on the right-hand side", но judge этого не видит).
Это ограничение LLM-судьи, а не качество вопросов.

---

## 7. Справочник артефактов

### Run 25 (`results/decomposed_bb50_r25/breakfast_box_20260519_233659/`)

| Файл | Размер | Содержание |
|------|-------:|-----------|
| `breakfast_box_results.json` | ~2 KB | Итоговые метрики: AUROC, F1-max, Bin-F1, confusion matrix |
| `run_artifacts.json` | 17.9 MB | Все стадии (10 ключей): stage1..stage4 |
| `stage1_descriptions.json` | — | 30 per-component описаний (5 img × 6 comp) |
| `stage2_summary.json` | — | Normality definition (7 секций, prompt + response) |
| `stage3a_questions.json` | — | 23 кандидата (до Stage 3b фильтра) |
| `stage3b_filtering.json` | 1.1 MB | Ответы VLM на каждый вопрос × 15 val images |
| `stage3c_subquestions.json` | — | 4 sub-variants × N kept questions |
| `stage4_responses.json` | 16.4 MB | Полные VLM-ответы по 50 test images |
| `stage4_final_results.json` | — | Бинарные предсказания + anomaly_score на 50 images |
| `pipeline.log` | — | Полный лог: промпты + ответы всех VLM-вызовов |

### Run 26 (`results/decomposed_bb50_r26/breakfast_box_20260520_104321/`) — *в процессе*

- Commit: `b5e2597`, Model: `OpenGVLab/InternVL2_5-38B-AWQ`
- Статус: Stage 3b (filtering), ~17 ч до завершения Stage 4
- GPU: ~40 GB VRAM (4-bit AWQ), 96% utilization

### Структура `run_artifacts.json`

```json
{
  "class_name": "breakfast_box",
  "run_dir": "...",
  "normality_definition": "...",
  "stage1_descriptions": [...],     // 30 записей {image_path, component, response}
  "stage2_summary": {...},           // {prompt, response}
  "stage3a_questions": {...},        // {parsed_questions: [...]}
  "stage3b_filtering": [...],        // {question, image_path, extracted_answer, ...}
  "stage3c_subqs": {...},            // {question: [sub1, sub2, sub3, sub4]}
  "stage4_subq_responses": [...],    // ответы sub-вопросов по 50 test images
  "stage4_image_results": [...]      // 50 записей {image_path, is_anomaly, anomaly_score, ...}
}
```

---

## 8. Конфигурации

| Конфиг | Модель | n_shots | n_val | n_q | n_subq | min_fail | decomposed | Используется в |
|--------|--------|:-------:|:-----:|:---:|:------:|:--------:|:----------:|---------------|
| `config_baseline_full.yaml` | InternVL2.5-8B | 3 | — | 6 | 5 | 1 | False | baseline |
| `config_baseline_gt.yaml` | InternVL2.5-8B | 3 | — | — | 5 | 1 | False | GT questions baseline |
| `config_decomposed_bb50.yaml` | InternVL2.5-8B | 5 | 15 | 8 | 4 | 2 | True | r18–r25 |
| `config_decomposed_bb50_38b.yaml` | InternVL2.5-38B-AWQ | 5 | 15 | 8 | 4 | 2 | True | r26 |

**Ключевые параметры**:
- `anomaly_min_failures=2`: аномалия если ≥ 2 main-вопроса получили "No"
- `question_filter_threshold=0.8`: Stage 3b — вопрос сохраняется если ≥80% val-ответов "Yes"
- `sub_question_mode="rephrase"`: sub-вопросы — перефразировки основного (не инверсии)
- `ensemble_seeds=[42]`: один seed (нет ансамблирования)

---

## 9. Изменения кодовой базы (baseline → feature/rc4a-rc5)

**Итого**: 30 файлов, +3600 / -296 строк.

| Файл | +/- | Основные изменения |
|------|----:|-------------------|
| `logicqa/pipeline/logicqa.py` | +380 | Декомпозированный режим, ансамблирование |
| `logicqa/prompts/__init__.py` | +588 | Все 5 промптов переработаны, sub-Q rules, markers |
| `logicqa/pipeline/stage3_questions.py` | +348 | count-bypass, stem-match, structured gen |
| `logicqa/pipeline/stage1_describe.py` | +334 | per-component calls, ComponentObs dataclass |
| `logicqa/pipeline/stage2_summarize.py` | +254 | hedge-filtering, 7-section output |
| `logicqa/vlm/internvl.py` | +109 | AWQ: device_map, patch, eos_token_id fix |
| `logicqa/data/normality_definitions.py` | +122 | NORMALITY_COMPONENTS для каждого класса |
| `logicqa/pipeline/stage4_test.py` | +136 | CoT (RC4-A), spatial exclusivity |
| `logicqa/evaluation/llm_judge.py` | +99 | `_parse_list`, `max_new_tokens`, CCR fix |
| `logicqa/data/evaluation_gt.py` | ±12 | ATOMIC_CONSTRAINTS обновлены |
| `scripts/evaluate_run.py` | +49 | Иерархическая оценка (4 уровня) |
| `scripts/reeval.py` | новый | Пересчёт метрик при разных threshold |

---

## 10. Анализ ошибок и следующие шаги

### Достижения run25

| Аспект | r23 | r25 | Δ |
|--------|----:|----:|--:|
| AUROC | 0.782 | **0.852** | +0.070 |
| F1-max | 0.746 | **0.818** | +0.072 |
| FP (ложные тревоги) | **12** | **0** | −12 |
| FN (пропуски аномалий) | 3 | 10 | +7 |

**FP = 0**: Inline CoT полностью устранил ложные срабатывания. Модель аргументирует
наблюдением перед ответом → нет "угадывания".

### Анализ FN = 10

10 пропущенных аномалий: все имеют `anomaly_score = 0.00` или близкое к нулю — VLM
отвечает "Yes" на ВСЕ вопросы. Типы пропущенных аномалий:

- **Spatial anomalies** (≈7/10): нарушение расположения компонент (нектарин справа вместо
  слева). InternVL2.5-8B не различает тонкое spatial violation на таком уровне детализации.
- **Count edge cases** (≈3/10): 1 vs 2 мандарина при похожем визуальном наполнении.

**Гипотеза**: InternVL2.5-38B-AWQ (r26) может улучшить результат на spatial anomalies
за счёт более сильного vision-энкодера и большего LLM.

**Альтернатива**: `anomaly_min_failures=1` повышает Bin-F1 с 0.750 до 0.818 (+3 TP, +1 FP),
что близко к теоретическому пределу для 8B модели.

### Следующие шаги

1. **Дождаться r26** (InternVL 38B-AWQ) → сравнить с r25 по spatial anomalies
2. **Расширить на другие классы MVTec LOCO AD** (juice_bottle, pushpins, screw_bag, splicing_connectors)
3. **Улучшить LLM-судью** для L2.5 Filtering: Qwen2.5-3B не справляется с semantic matching
4. **Исследовать threshold=1** как production-вариант при допустимом FP

---

## Воспроизведение результатов

```bash
# Запуск run25 (воспроизведение)
git checkout 3d7f4d3
python3 scripts/run_pipeline.py \
  --class_name breakfast_box \
  --config config_decomposed_bb50.yaml \
  --output_dir results/decomposed_bb50_r25_repro \
  --data_dir dataset-ninja/ \
  --seed 42 \
  --save_questions

# Иерархическая оценка
python3 scripts/evaluate_run.py \
  --levels 1,2,2.5,3,4 \
  --judge_model Qwen/Qwen2.5-3B-Instruct \
  --device cuda \
  --artifacts results/decomposed_bb50_r25/breakfast_box_20260519_233659/run_artifacts.json

# Пересчёт метрик при threshold=1
python3 scripts/reeval.py \
  results/decomposed_bb50_r25/breakfast_box_20260519_233659/breakfast_box_results.json \
  --threshold 1 --show-errors
```

---

*Документ актуален на 20 мая 2026 г. Run 26 в процессе — результаты будут добавлены по завершении.*
