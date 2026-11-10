from src.embeddings import embed_text
from src.vectorstore import already_ingested, build_bm25_index, hybrid_search
from src.generator import generate_answer

if not already_ingested():
    print("Run test_vectorstore.py first to ingest data.")
    exit()

build_bm25_index()

question = "What happens when you play a Skip card according to uno extreme game?"
query_vector = embed_text(question, task_type="RETRIEVAL_QUERY")
chunks = hybrid_search(question, query_vector, top_k=3)

answer = generate_answer(question, chunks)
print("Answer:\n")
print(answer)