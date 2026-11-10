import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL = "gemini-embedding-001"
GROQ_MODEL = "llama-3.3-70b-versatile"
CHROMA_PATH = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

from src.config import GROQ_API_KEY, GEMINI_API_KEY

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")