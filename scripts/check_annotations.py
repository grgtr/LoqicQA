import json
from pathlib import Path
from typing import Dict, Set, Any

def find_unique_anomalies(dataset_root: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Проходит по всем JSON-файлам в подпапках ann (train, validation, test)
    и собирает уникальные значения classTitle для каждой категории сцены.
    Возвращает словарь вида {категория: {аномалия: {}}}.
    """
    # Множество тегов, которые не являются названиями категорий
    IGNORED_TAGS = {"logical_anomaly", "structural_anomaly"}

    # Результирующий словарь: категория -> множество аномалий
    anomalies_per_category: Dict[str, Set[str]] = {}

    # Проходим по всем подпапкам train, validation, test
    for split in ["train", "validation", "test"]:
        ann_dir = dataset_root / split / "ann"
        if not ann_dir.exists():
            continue

        # Ищем все JSON-файлы в ann_dir
        for json_path in ann_dir.glob("*.json"):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Определяем категорию сцены из тегов изображения
            category = None
            if "tags" in data:
                for tag in data["tags"]:
                    tag_name = tag.get("name")
                    if tag_name and tag_name not in IGNORED_TAGS:
                        category = tag_name
                        break  # берём первый подходящий тег

            if not category:
                # Если категория не найдена, пропускаем файл
                continue

            # Инициализируем множество для категории, если его нет
            if category not in anomalies_per_category:
                anomalies_per_category[category] = set()

            # Собираем classTitle из объектов
            for obj in data.get("objects", []):
                ct = obj.get("classTitle")
                if ct:
                    anomalies_per_category[category].add(ct)

    # Преобразуем множества в словари с пустыми значениями
    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for category, anomalies in anomalies_per_category.items():
        result[category] = {anomaly: {} for anomaly in sorted(anomalies)}

    return result

if __name__ == "__main__":
    # Путь к корню датасета (изменить при необходимости)
    dataset_path = Path("dataset-ninja/mvtec-loco-ad")
    attribute_gt = find_unique_anomalies(dataset_path)

    # Вывод результата в читаемом формате
    print(json.dumps(attribute_gt, indent=4, ensure_ascii=False))