import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")  # для отримання повного abstract

# Реалізація функції кодування запиту в ембеддинг
def encode_query(query: str):
    return model.encode([query], normalize_embeddings=True)[0].tolist()

def main():
    # Підключення до індексу Pinecone
    print(f"Підключення до індексу '{INDEX_NAME}' у Pinecone...")
    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        raise ValueError(f"Індекс '{INDEX_NAME}' не існує. Будь ласка, створіть його перед запуском скрипту.")
    
    # Виконати чистий семантичний пошук
    query = "teaching machines to recognize objects in pictures"
    query_embedding = encode_query(query)
    results = index.query(vector=query_embedding, top_k=TOP_K, include_metadata=True)

    # Вивести результати
    for i, result in enumerate(results['matches']):
        print(result)
        print(f"Результат {i+1}:")
        print(f"  Назва: {result['metadata']['title']}")
        print(f"  Категорія: {result['metadata']['category']}")
        print(f"  Рік: {result['metadata']['year']}")
        print(f"  Абстракт: {result['metadata']['abstract'][:100]}...")
        print()

    # Виконати пошук з фільтрацією
    # Приклад A: статті по reinforcement learning за останні 5 років і категорія cs.LG
    query_a = f"{query} reinforcement learning"
    query_embedding_a = encode_query(query_a)
    filter_a = {
        "year": {"$gte": 2019},
        "category": {"$eq": "cs.LG"}
    }
    results_a = index.query(vector=query_embedding_a, top_k=TOP_K, include_metadata=True, filter=filter_a)
    print("Результати пошуку з фільтрацією (Приклад A):")
    for i, result in enumerate(results_a['matches']):
        print(f"Результат {i+1}:")
        print(f"  Назва: {result['metadata']['title']}")
        print(f"  Категорія: {result['metadata']['category']}")
        print(f"  Рік: {result['metadata']['year']}")
        print(f"  Абстракт: {result['metadata']['abstract'][:100]}...")
        print()

    # Приклад B: більш старі статті (до 2015 року), будь-яка категорія
    filter_b = {
        "year": {"$lte": 2015}
    }
    results_b = index.query(vector=query_embedding_a, top_k=TOP_K, include_metadata=True, filter=filter_b)
    print("Результати пошуку з фільтрацією (Приклад B):")
    for i, result in enumerate(results_b['matches']):
        print(f"Результат {i+1}:")
        print(f"  Назва: {result['metadata']['title']}")
        print(f"  Категорія: {result['metadata']['category']}")
        print(f"  Рік: {result['metadata']['year']}")
        print(f"  Абстракт: {result['metadata']['abstract'][:100]}...")
        print()

    # Порівняти різні метрики схожості на локальних ембеддингах
    embeddings = np.load("embeddings/embeddings.npy")
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Норми ембеддингів: min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}")
    
    query_embedding_local = model.encode([query], normalize_embeddings=True)[0]
    
    similarities_local = np.dot(embeddings, query_embedding_local)
    top_k_indices_cosine = np.argsort(similarities_local)[-TOP_K:][::-1]
    print("Топ-5 статей за cosine similarity:")
    for idx in top_k_indices_cosine:
        print(f"  Cosine: {similarities_local[idx]:.4f}")
        print(f"  Назва: {df.iloc[idx]['title']}")
        print(f"  Категорія: {df.iloc[idx]['category']}")
        print(f"  Рік: {df.iloc[idx]['year']}")
        print(f"  Абстракт: {df.iloc[idx]['abstract'][:100]}...")
        print()
    
    # Dot product
    dot_products = np.dot(embeddings, query_embedding_local)
    top_k_indices_dot = np.argsort(dot_products)[-TOP_K:][::-1]
    print("Топ-5 статей за dot product:")
    for idx in top_k_indices_dot:
        print(f"  Dot Product: {dot_products[idx]:.4f}")
        print(f"  Назва: {df.iloc[idx]['title']}")
        print(f"  Категорія: {df.iloc[idx]['category']}")
        print(f"  Рік: {df.iloc[idx]['year']}")
        print(f"  Абстракт: {df.iloc[idx]['abstract'][:100]}...")
        print()

    # L2-distance
    l2_distances = np.linalg.norm(embeddings - query_embedding_local, axis=1)
    top_k_indices_l2 = np.argsort(l2_distances)[:TOP_K]
    print("Топ-5 статей за L2-distance:")
    for idx in top_k_indices_l2:
        print(f"  L2 Distance: {l2_distances[idx]:.4f}")
        print(f"  Назва: {df.iloc[idx]['title']}")
        print(f"  Категорія: {df.iloc[idx]['category']}")
        print(f"  Рік: {df.iloc[idx]['year']}")
        print(f"  Абстракт: {df.iloc[idx]['abstract'][:100]}...")
        print()

if __name__ == "__main__":
    main()
