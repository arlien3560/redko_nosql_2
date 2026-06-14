import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200   # Pinecone рекомендує батчі до 200 векторів

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

def get_index(index_name: str):
    print("Перевірка наявності індексу...")
    # Перевірка наявності індексу
    print(f"Список існуючих індексів: {pc.list_indexes()}")

    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Індекс '{index_name}' не існує. Створюю...")
        pc.create_index(name=index_name, dimension=VECTOR_DIM, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    else:
        print(f"Індекс '{index_name}' вже існує.")
    
    # Підключення до індексу
    index = pc.Index(index_name)
    return index

def main():
    index = get_index(INDEX_NAME)

    # Завантаження даних
    print("Завантаження даних...")
    df = pd.read_parquet(INPUT_PARQUET)
    embeddings = np.load(INPUT_EMBEDDINGS)
    print(f"Завантажено {len(df)} записів.")
    print(f"Завантажено {len(embeddings)} ембеддингів.")

    # Перевірка відповідності кількості записів та ембеддингів
    if len(df) != len(embeddings):
        raise ValueError("Кількість записів у датасеті не відповідає кількості ембеддингів.")
    
    # Підготовка даних для завантаження
    print("Підготовка даних для завантаження...")
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Завантаження батчів"):
        end = min(start + BATCH_SIZE, len(df))
        batch_df = df.iloc[start:end]
        batch_embeddings = embeddings[start:end]

        # Формування об'єктів для завантаження
        vectors_to_upsert = []
        for i, row in batch_df.iterrows():
            vector_id = f"paper_{i}"
            metadata = {
                "arxiv_id": row["id"],
                "title": row["title"],
                "abstract": row["abstract"][:500],  # обмеження до 500 символів
                "authors": row["authors"][:200],    # обмеження до 200 символів
                "year": row["year"],
                "category": row["category"]
            }
            vectors_to_upsert.append((vector_id, batch_embeddings[i - start].tolist(), metadata))

        # Завантаження батчу в Pinecone
        index.upsert(vectors=vectors_to_upsert)
        print(f"Завантажено батч {start//BATCH_SIZE + 1} з {len(df)//BATCH_SIZE + 1}")

    # Після завершення завантаження вивести загальну кількість векторів в індексі
    index_stats = index.describe_index_stats()
    total_vectors = index_stats["total_vector_count"]
    print(f"Завершено завантаження. Загальна кількість векторів в індексі '{INDEX_NAME}': {total_vectors}")

if __name__ == "__main__":
    main()