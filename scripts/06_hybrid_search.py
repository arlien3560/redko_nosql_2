import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10        # беремо ширше, щоб RRF міг переранжувати
RRF_K = 60        # згладжувальна константа RRF (канонічне значення)

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

arxiv_to_idx = {str(row["id"]): i for i, row in df.iterrows()}

# 1. BM25-індекс за title + abstract
corpus = (df["title"] + " " + df["abstract"]).tolist()
tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)


# 3. RRF: об’єднання двох ранжованих списків індексів датафрейму
def reciprocal_rank_fusion(bm25_ranking, vector_ranking, k=RRF_K, top_k=TOP_K):
    rrf_scores = {}
    for rank, doc_idx in enumerate(bm25_ranking):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (k + rank + 1)
    for rank, doc_idx in enumerate(vector_ranking):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (k + rank + 1)
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_k]


# 4. Функції пошуку. Обидві повертають індекси датафрейму (єдиний простір ID)
def search_bm25(query):
    tokenized_query = query.split()
    doc_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(doc_scores)[::-1][:TOP_K]
    return list(top_indices), doc_scores[top_indices]


def search_vector(query):
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()[0]
    vector_results = index.query(vector=query_embedding, top_k=TOP_K, include_metadata=True)

    indices = []
    scores = []
    for res in vector_results["matches"]:
        arxiv_id = str(res["metadata"]["arxiv_id"])
        idx = arxiv_to_idx.get(arxiv_id)
        if idx is not None:
            indices.append(idx)
            scores.append(res["score"])
    return indices, scores


def search_hybrid(query):
    bm25_indices, _ = search_bm25(query)
    vector_indices, _ = search_vector(query)
    return reciprocal_rank_fusion(bm25_indices, vector_indices)


# Допоміжний вивід рядка результату
def fmt(idx):
    return df.iloc[idx]["title"][:70]


# 5 + 6. Демонстрація і порівняння
queries = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text",
]

for query in queries:
    print(f"Query: {query}")

    bm25_indices, bm25_scores = search_bm25(query)
    print("\nBM25 Top-5:")
    for idx, score in zip(bm25_indices[:5], bm25_scores[:5]):
        print(f"  [{idx}] {fmt(idx)}  | BM25: {score:.4f}")

    vector_indices, vector_scores = search_vector(query)
    print("\nVector Search Top-5:")
    for idx, score in zip(vector_indices[:5], vector_scores[:5]):
        print(f"  [{idx}] {fmt(idx)}  | cos: {score:.4f}")

    hybrid_results = search_hybrid(query)
    print("\nHybrid Search (RRF) Top-5:")
    for doc_idx, rrf_score in hybrid_results[:5]:
        print(f"  [{doc_idx}] {fmt(doc_idx)}  | RRF: {rrf_score:.4f}")