# План: Запуск run25-пайплайна на 4 оставшихся класса MVTec LOCO AD

## Контекст

- **Выполнено:** breakfast_box с run25 (8B) → AUROC=0.852, и run26 (38B-AWQ) → AUROC=0.900
- **Цель:** Запустить тот же пайплайн (run25, InternVL2.5-8B) на 4 оставшихся классах:
  - `juice_bottle`, `pushpins`, `screw_bag`, `splicing_connectors`
- **Фильтрация:** только `good` + `logical_anomalies` (без `structural_anomalies`)
- **Оптимизация:** вместо автоматической генерации вопросов (Stage 1–3) использовать
  заранее созданные файлы вопросов через `pipeline.load_questions(path)`

---

## Статистика датасета

| Класс | good | logical | structural | Итого (без structural) |
|-------|-----:|--------:|-----------:|-----------------------:|
| juice_bottle | 94 | 142 | 94 | 236 |
| pushpins | 138 | 91 | 81 | 229 |
| screw_bag | 122 | 137 | 82 | 259 |
| splicing_connectors | 119 | 108 | 85 | 227 |

---

## Типы логических аномалий по классам

### juice_bottle (142 изображения, 11 типов)
- `empty_bottle` — бутылка пустая или почти пустая
- `misplaced_fruit_icon` — иконка фрукта не в центре этикетки
- `misplaced_label_bottom` — нижняя этикетка не на месте
- `misplaced_label_top` — верхняя этикетка не на месте
- `missing_bottom_label` — нет нижней этикетки ("100% Juice")
- `missing_fruit_icon` — нет иконки фрукта на верхней этикетке
- `missing_top_label` — нет верхней (центральной) этикетки
- `swapped_labels` — этикетки перепутаны местами
- `wrong_fill_level_not_enough` — уровень наполнения слишком низкий
- `wrong_fill_level_too_much` — уровень наполнения слишком высокий (переполнена)
- `wrong_juice_type` — тип сока не соответствует иконке на этикетке

### pushpins (91 изображение, 4 типа)
- `1_additional_pushpin` — лишняя кнопка в отсеке (×247 масок)
- `2_additional_pushpins` — две лишних кнопки в отсеке
- `missing_pushpin` — кнопка отсутствует в отсеке (×143 масок)
- `missing_separator` — нет перегородки между отсеками

### screw_bag (137 изображений, 16 типов)
**Лишние элементы:** 1_additional_long_screw, 1_additional_nut, 1_additional_short_screw,
1_additional_washer, 2_additional_nuts, 2_additional_washers

**Недостающие элементы:** 1_missing_long_screw, 1_missing_nut, 1_missing_short_screw,
1_missing_washer, 2_missing_nuts, 2_missing_washers

**Неправильные элементы:** 1_very_short_screw, 2_very_short_screws, screw_too_long, screw_too_short

### splicing_connectors (108 изображений, 9 типов)
- `cable_color` — цвет кабеля не соответствует числу зажимов
- `cable_cut` — кабель повреждён/обрезан
- `extra_cable` — лишний кабель
- `missing_cable` — нет кабеля
- `missing_connector` — нет одного из коннекторов
- `wrong_cable_location` — кабель подключён не в ту позицию (нарушена зеркальная симметрия)
- `wrong_connector_type_3_2`, `_5_2`, `_5_3` — коннекторы разного типа (разное число зажимов)

---

## Существующая инфраструктура

### Ключевые файлы
- **Скрипт пайплайна:** `scripts/run_pipeline.py`
  - Аргумент `--questions_file PATH` → вызывает `pipeline.load_questions(path)`, пропуская Stage 1–3
  - Аргумент `--config PATH` для YAML конфига
  - Аргумент `--class_name` для выбора класса
- **Метод загрузки вопросов:** `logicqa/pipeline/logicqa.py:643` — `load_questions(path)`
- **Нормативные определения:** `logicqa/data/normality_definitions.py` — уже есть для всех 5 классов
- **Компоненты:** `NORMALITY_COMPONENTS` в том же файле
- **Референсный конфиг (8B):** `config_decomposed_bb50.yaml`
- **Пример скрипта запуска:** `run25_standalone.sh`

### Формат файла вопросов (JSON)
```json
{
  "class_name": "juice_bottle",
  "normality_definition": "...",
  "main_questions": [
    "Is there a label at the center of the bottle?",
    ...
  ],
  "sub_questions": {
    "Is there a label at the center of the bottle?": [
      "Can you see a label attached to the center of the bottle?",
      "Is a central label present on the bottle?",
      "Do you observe a label at the middle section of the bottle?",
      "Is there a visible label positioned at the center of the bottle?"
    ],
    ...
  }
}
```

### Пример существующего файла вопросов
`results/decomposed_bb50_r25/breakfast_box_20260519_233659/breakfast_box_questions.json`
или
`results/decomposed_bb50_r26/breakfast_box_20260520_104321/breakfast_box_questions.json`

---

## Что нужно сделать

### Шаг 1: Добавить фильтрацию structural в run_pipeline.py

В `scripts/run_pipeline.py` после `select_test_samples()` добавить:

```python
# Filter out unwanted labels (e.g., structural_anomalies)
exclude_labels = getattr(cfg.testing, 'exclude_labels', [])
if exclude_labels:
    before = len(test_samples)
    test_samples = [s for s in test_samples if s.label not in exclude_labels]
    print(f"[TestSelect] exclude_labels={exclude_labels} → {before}→{len(test_samples)} samples")
```

И в `logicqa/config.py` в `TestingConfig`:
```python
exclude_labels: List[str] = field(default_factory=list)
```
А в `LogicQAConfig.from_yaml()` добавить парсинг `exclude_labels`.

### Шаг 2: Создать файлы вопросов для каждого класса

Создать в `questions/` (новая папка):
- `questions/juice_bottle_questions.json`
- `questions/pushpins_questions.json`
- `questions/screw_bag_questions.json`
- `questions/splicing_connectors_questions.json`

**Процесс создания вопросов для каждого класса:**

1. Посмотреть 1–2 нормальных изображения класса
2. Посмотреть по 1 изображению каждого типа аномалии
3. Написать main_questions, покрывающие все типы аномалий
4. Написать 4 sub_questions (перефразировки) для каждого

**Черновики вопросов (для уточнения после визуального осмотра):**

#### juice_bottle (~12 вопросов)
1. Is there a label attached to the center of the bottle?  ← missing_top_label
2. Is there a label attached to the lower part of the bottle?  ← missing_bottom_label
3. Does the central label have a fruit icon on it?  ← missing_fruit_icon
4. Does the lower label have "100% Juice" text on it?
5. Is the central label positioned at the center/middle of the bottle?  ← misplaced_label_top
6. Is the lower label positioned at the bottom section of the bottle?  ← misplaced_label_bottom
7. Is the fruit icon centered on the central label?  ← misplaced_fruit_icon
8. Is the central label above the lower label? ← swapped_labels
9. Is the bottle filled between 70% and 99% of its capacity?  ← empty / too_much / not_enough
10. Is the juice color/type consistent with the fruit icon on the label?  ← wrong_juice_type
11. Is the bottle not completely full (no juice reaching the very top)?  ← wrong_fill_level_too_much
12. Is the bottle at least half full?  ← wrong_fill_level_not_enough / empty_bottle

#### pushpins (~6 вопросов)
1. Does every compartment contain at least one pushpin?  ← missing_pushpin
2. Does every compartment contain no more than one pushpin?  ← 1/2_additional_pushpin
3. Are there any empty compartments in the box?  ← missing_pushpin
4. Are there any compartments with two or more pushpins?  ← additional
5. Is there a separator/divider between the compartments?  ← missing_separator
6. Does each compartment contain exactly one pushpin?  ← combined check

#### screw_bag (~10 вопросов)
1. Are there exactly two washers in the bag?  ← washer anomalies
2. Are there exactly two nuts in the bag?  ← nut anomalies
3. Is there exactly one long screw in the bag?  ← long_screw anomalies
4. Is there exactly one short screw in the bag?  ← short_screw anomalies
5. Is the long screw clearly longer than the short screw?  ← screw_too_short/long
6. Are all screws of standard length (not unusually tiny or extra long)?  ← very_short/too_long
7. Is the long screw longer than 3 times the width of a washer?  ← screw length check
8. Is the short screw shorter than the long screw by a visible amount?
9. Are there no extra hardware items beyond 2 washers, 2 nuts, 1 long screw, 1 short screw?
10. Is there at least one screw of each length (one long and one short)?

#### splicing_connectors (~9 вопросов)
1. Are there exactly two splicing connectors visible?  ← missing_connector
2. Do both connectors have the same number of cable clamps?  ← wrong_connector_type
3. Is there exactly one cable connecting the two connectors?  ← extra_cable / missing_cable
4. Is the cable intact and not cut or broken?  ← cable_cut
5. Is the cable connected to the same terminal position on both connectors?  ← wrong_cable_location
6. Does the cable maintain mirror symmetry between the two connectors?  ← wrong_cable_location
7. Is the cable color consistent with the number of clamps on the connectors?  ← cable_color
8. Is the cable length longer than the width of one connector terminal block?
9. Is there no extra/additional cable in the image beyond the one connecting the connectors?  ← extra_cable

### Шаг 3: Создать конфиги YAML

Папка: `configs/` (новая)  
Шаблон (на основе `config_decomposed_bb50.yaml`):

```yaml
# Run XX: juice_bottle, 8B model, prebuilt questions, logical-only
vlm:
  backend: "internvl"
  internvl:
    model_name: "OpenGVLab/InternVL2_5-8B"
    temperature: 0.2
    top_p: 0.7
    repetition_penalty: 1.1
    do_sample: true
    max_new_tokens: 512
    max_tiles: 12
    device_map: "auto"

pipeline:
  decomposed_description: true
  question_generation_mode: "structured"
  n_shots: 5
  n_val_shots: 15
  n_questions: 8
  n_sub_questions: 4
  question_filter_threshold: 0.8
  ensemble_seeds: [42]
  use_llm_judge_hallucination: false
  anomaly_min_failures: 1    # оптимум из run26 анализа
  use_grounded_reasoning: false
  sub_question_mode: "rephrase"

dataset:
  name: "mvtec_loco"
  data_dir: "/home/chikibriki/LoqicQA/dataset-ninja/"
  download_if_missing: false

output:
  results_dir: "results/r27_juice_bottle"
  save_questions: true
  save_per_image: true

testing:
  mode: "all"
  exclude_labels: ["structural_anomaly"]   # НОВЫЙ параметр (см. Шаг 1)
```

4 конфига: `configs/r27_juice_bottle.yaml`, `configs/r28_pushpins.yaml`,
`configs/r29_screw_bag.yaml`, `configs/r30_splicing_connectors.yaml`

### Шаг 4: Создать скрипт последовательного запуска

```bash
#!/usr/bin/env bash
# run27-30_sequential.sh — Run 27-30: all 4 remaining classes, 8B model

MAIN_REPO="/home/chikibriki/LoqicQA"
VENV_PYTHON="${MAIN_REPO}/.venv/bin/python3"
SCRIPT="${MAIN_REPO}/scripts/run_pipeline.py"
GPU_THRESHOLD_MB=18000

# Классы → конфиги → файлы вопросов
declare -A CLASSES=(
  [juice_bottle]="configs/r27_juice_bottle.yaml"
  [pushpins]="configs/r28_pushpins.yaml"
  [screw_bag]="configs/r29_screw_bag.yaml"
  [splicing_connectors]="configs/r30_splicing_connectors.yaml"
)
declare -A QUESTIONS=(
  [juice_bottle]="questions/juice_bottle_questions.json"
  [pushpins]="questions/pushpins_questions.json"
  [screw_bag]="questions/screw_bag_questions.json"
  [splicing_connectors]="questions/splicing_connectors_questions.json"
)

for CLASS in juice_bottle pushpins screw_bag splicing_connectors; do
  echo "Starting ${CLASS}..."
  "${VENV_PYTHON}" "${SCRIPT}" \
    --class_name "${CLASS}" \
    --config "${MAIN_REPO}/${CLASSES[$CLASS]}" \
    --questions_file "${MAIN_REPO}/${QUESTIONS[$CLASS]}" \
    --data_dir "${MAIN_REPO}/dataset-ninja/" \
    --seed 42 \
    2>&1 | tee "/tmp/run_${CLASS}.log"
  echo "Finished ${CLASS}"
done
```

---

## Порядок выполнения

1. **Визуальный осмотр изображений** (интерактивно)
   - Для каждого класса: 2 нормальных + по 1 на каждый тип аномалии
   - Команда: `ls dataset-ninja/mvtec-loco-ad/test/ann/juice_bottle_logical*.json | python3 -c "..."`
   - Использовать Read tool на изображениях или отобразить их

2. **Создать question JSON файлы** (на основе визуального осмотра)

3. **Добавить exclude_labels** в `scripts/run_pipeline.py` и `logicqa/config.py`

4. **Создать конфиги** (`configs/*.yaml`)

5. **Запустить** `bash run27-30_sequential.sh`

---

## Важные ссылки

- Референсный формат вопросов: `results/decomposed_bb50_r26/breakfast_box_20260520_104321/breakfast_box_questions.json`
- Анализ run26: `results/decomposed_bb50_r26/ANALYSIS.md`
- Существующие нормативные определения: `logicqa/data/normality_definitions.py`
- Существующие компоненты: `NORMALITY_COMPONENTS` в том же файле
- Скрипт пересчёта метрик: `scripts/reeval.py --threshold 1`
- Аннотации датасета: `dataset-ninja/mvtec-loco-ad/test/ann/{class}_{type}_{NNN}.png.json`

---

## Примечания

- `anomaly_min_failures=1` — оптимум (из анализа run26: Bin-F1 0.864→0.889, FP=0)
- **juice_bottle** и **splicing_connectors** имеют варианты (тип фрукта / цвет кабеля) — нормативное определение параметризовано, проверить как пайплайн обрабатывает варианты
- **screw_bag** и **splicing_connectors** требуют BPM preprocessing (`BPM_CLASSES` в normality_definitions.py) — убедиться что `preprocessing.bpm.enabled: false` или разобраться нужно ли
- Ожидаемое время на класс при 8B: ~3 часа для 50 изображений → 229–259 изображений ≈ **14–16 часов на класс**
