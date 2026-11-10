import chromadb
from rank_bm25 import BM25Okapi
from src.config import CHROMA_PATH

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="uno_rules")

# In-memory BM25 index — rebuilt on every app startup (cheap, local, no API calls)
bm25_index = None
bm25_corpus_chunks = []


def already_ingested() -> bool:
    return collection.count() > 0


def add_chunks(embedded_chunks: list[dict]):
    collection.add(
        ids=[c["chunk_id"] for c in embedded_chunks],
        embeddings=[c["embedding"] for c in embedded_chunks],
        documents=[c["text"] for c in embedded_chunks],
        metadatas=[{"source": c["source"]} for c in embedded_chunks],
    )


def build_bm25_index():
    global bm25_index, bm25_corpus_chunks
    all_data = collection.get()
    bm25_corpus_chunks = [
        {
            "chunk_id": all_data["ids"][i],
            "text": all_data["documents"][i],
            "source": all_data["metadatas"][i]["source"],
        }
        for i in range(len(all_data["ids"]))
    ]
    tokenized = [c["text"].lower().split() for c in bm25_corpus_chunks]
    bm25_index = BM25Okapi(tokenized)


def keyword_search(query: str, top_k: int = 5) -> list[dict]:
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [bm25_corpus_chunks[i] for i in ranked]


def vector_search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    matches = []
    for i in range(len(results["ids"][0])):
        matches.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
        })
    return matches


def hybrid_search(query: str, query_embedding: list[float], top_k: int = 5, rrf_k: int = 60) -> list[dict]:
    vec_results = vector_search(query_embedding, top_k=top_k * 2)
    kw_results = keyword_search(query, top_k=top_k * 2)

    scores = {}
    chunk_lookup = {}

    for rank, chunk in enumerate(vec_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (rrf_k + rank + 1)
        chunk_lookup[cid] = chunk

    for rank, chunk in enumerate(kw_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (rrf_k + rank + 1)
        chunk_lookup[cid] = chunk

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]
    return [{**chunk_lookup[cid], "score": scores[cid]} for cid in ranked_ids]