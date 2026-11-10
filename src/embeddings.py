from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, EMBEDDING_MODEL

client = genai.Client(api_key=GEMINI_API_KEY)

def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return result.embeddings[0].values

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Adds an 'embedding' field to each chunk dict."""
    embedded = []
    for chunk in chunks:
        vector = embed_text(chunk["text"], task_type="RETRIEVAL_DOCUMENT")
        embedded.append({**chunk, "embedding": vector})
    return embedded