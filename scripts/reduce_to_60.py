#!/usr/bin/env python3
import random
import sys

def reduce_to_60(filename):
    # Читаем все строки из файла
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Если строк уже 60 или меньше, ничего не делаем
    if len(lines) <= 60:
        print(f"Файл содержит {len(lines)} строк (<= 60), изменений не требуется.")
        return

    # Выбираем случайные 60 индексов (без повторений)
    indices = random.sample(range(len(lines)), 60)
    # Сортируем индексы, чтобы сохранить исходный порядок
    indices.sort()

    # Выбираем соответствующие строки
    selected_lines = [lines[i] for i in indices]

    # Записываем обратно в файл (перезаписываем)
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(selected_lines)

    print(f"Было {len(lines)} строк, оставлено 60 случайных (порядок сохранён).")

if __name__ == "__main__":
    # Если имя файла передано аргументом, используем его, иначе по умолчанию 'output.txt'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "output.txt"

    try:
        reduce_to_60(filename)
    except FileNotFoundError:
        print(f"Ошибка: файл '{filename}' не найден.")
        sys.exit(1)