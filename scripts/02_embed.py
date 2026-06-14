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