'''1.3. Отримання ембеддингів

Вам необхідно реалізувати скрипт 02_embed.py для перетворення текстових даних у векторні представлення (ембеддинги) з використанням попередньо навченої моделі з HuggingFace.

Скрипт повинен виконувати такі кроки:

1. Завантажити датасет із файлу data/arxiv_subset.parquet з використанням бібліотеки pandas.

2. Підготувати тексти для кодування:

для кожного запису об’єднати поля title і abstract в один рядок у форматі:title + " [SEP] " + abstract
важливо: токен [SEP] обов’язковий, оскільки модель навчена працювати саме з таким форматом вхідних даних.

3. Згенерувати ембеддинги текстів за допомогою моделі allenai/specter2_base з бібліотеки sentence-transformers.

4. Закодувати всі тексти в ембеддинги з урахуванням таких вимог:

використовувати батчеву обробку (наприклад, batch_size=64);
увімкнути відображення прогресу;
нормалізувати ембеддинги (normalize_embeddings=True).
5. Вивести в консоль:

загальну кількість оброблених текстів;
розмірність ембеддингів (очікується 768);
норму першого ембеддингу (повинна бути близька до 1.0).
6. Зберегти отримані ембеддинги у файл embeddings/embeddings.npy у форматі NumPy.

7. Перед збереженням переконатися, що директорія embeddings існує; за потреби створити її.'''

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def main():
    path_to_data = "data/arxiv_subset.parquet"
    df = pd.read_parquet(path_to_data)

    print(f"Загальна кількість текстів: {len(df)}")
    
    # Дивлюсь формат даних для подальшої обробки
    print(df.head(5))

    # Підготовка текстів для кодування - розділення title та abstract за допомогою [SEP]
    texts = df.apply(lambda row: f"{row['title']} [SEP] {row['abstract']}", axis=1).tolist()

    # Згенерувати ембеддинги текстів за допомогою моделі allenai/specter2_base
    model = SentenceTransformer('allenai/specter2_base')
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    print(f"Загальна кількість оброблених текстів: {len(embeddings)}")
    print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
    print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0])}")

    # Зберегти отримані ембеддинги у файл
    os.makedirs("embeddings", exist_ok=True)
    np.save("embeddings/embeddings.npy", embeddings)

if __name__ == "__main__":
    main()