import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")


# Функція для завантаження чанків в Pinecone
def upload_chunks(chunks, index_name):
    index = pc.Index(index_name)
    batch_size = 10
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()
        vectors = [
            {
                "id": f"{chunk['arxiv_id']}_{chunk['chunk_num']}",
                "values": emb,
                "metadata": {
                    "arxiv_id": chunk["arxiv_id"],
                    "title": chunk["title"],
                    "text": chunk["text"],
                    "chunk_num": chunk["chunk_num"],
                    "year": chunk["year"],
                    "category": chunk["category"],
                },
            }
            for chunk, emb in zip(batch, embeddings)
        ]
        index.upsert(vectors=vectors)


# Функція для пошуку по чанках
def search_chunks(query, index_name):
    index = pc.Index(index_name)
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()[0]
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    return results


def main():
    # 1. Вибір 30 статей із найдовшими анотаціями
    df["abstract_length"] = df["abstract"].apply(lambda x: len(re.findall(r'\w+', x)))
    top_articles = df.nlargest(30, "abstract_length")

    # 2. Розбиття текстів на чанки
    fixed_chunks = []
    semantic_chunks = []

    for _, row in tqdm(top_articles.iterrows(), total=top_articles.shape[0]):
        abstract = row["abstract"]
        words = re.findall(r'\w+', abstract)

        chunk_size = 100
        overlap = 20

        # Fixed-size chunking
        fixed_num = 1  # локальний лічильник для цієї статті
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                fixed_chunks.append({
                    "arxiv_id": row["id"],
                    "title": row["title"],
                    "text": " ".join(chunk_words),
                    "chunk_num": fixed_num,
                    "year": row["year"],
                    "category": row["category"]
                })
                fixed_num += 1

        # Semantic chunking
        sentences = re.split(r'(?<=[.!?]) +', abstract)
        current_chunk = []
        current_length = 0
        semantic_num = 1  # локальний лічильник для цієї статті

        for sentence in sentences:
            sentence_length = len(re.findall(r'\w+', sentence))
            if current_length + sentence_length > chunk_size:
                if current_chunk:
                    semantic_chunks.append({
                        "arxiv_id": row["id"],
                        "title": row["title"],
                        "text": " ".join(current_chunk),
                        "chunk_num": semantic_num,
                        "year": row["year"],
                        "category": row["category"]
                    })
                    semantic_num += 1
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            semantic_chunks.append({
                "arxiv_id": row["id"],
                "title": row["title"],
                "text": " ".join(current_chunk),
                "chunk_num": semantic_num,
                "year": row["year"],
                "category": row["category"]
            })
            semantic_num += 1

    # 3. Створення індексів в Pinecone
    existing = [idx.name for idx in pc.list_indexes()]
    if "arxiv-chunks-fixed" not in existing:
        pc.create_index(
            "arxiv-chunks-fixed",
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    if "arxiv-chunks-semantic" not in existing:
        pc.create_index(
            "arxiv-chunks-semantic",
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    # 4 + 5. Створення ембеддингів і завантаження в Pinecone батчами
    upload_chunks(fixed_chunks, "arxiv-chunks-fixed")
    upload_chunks(semantic_chunks, "arxiv-chunks-semantic")

    # 6. Пошук по чанках
    test_queries = [
        "What are the latest advancements in machine learning?",
        "How does quantum computing impact cryptography?",
        "What are the applications of deep learning in healthcare?"
    ]

    for query in test_queries:
        print(f"\nSearch results for query: '{query}' in fixed-size chunks:")
        fixed_results = search_chunks(query, "arxiv-chunks-fixed")
        for res in fixed_results["matches"]:
            print(f"Title: {res['metadata']['title']}, Chunk Text: {res['metadata']['text'][:100]}...")

        print(f"\nSearch results for query: '{query}' in semantic chunks:")
        semantic_results = search_chunks(query, "arxiv-chunks-semantic")
        for res in semantic_results["matches"]:
            print(f"Title: {res['metadata']['title']}, Chunk Text: {res['metadata']['text'][:100]}...")


if __name__ == "__main__":
    main()