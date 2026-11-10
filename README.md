# UNO Rules Assistant

A retrieval-augmented generation (RAG) application that answers questions about
UNO Classic, UNO Flip, and UNO Extreme using their PDF rulebooks as the source
of truth.

The project uses Google Gemini embeddings, ChromaDB for vector search, and Groq
for answer generation. A Streamlit interface and a command-line interface are
available.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A Google Gemini API key
- A Groq API key

## Setup

Install the dependencies:

```powershell
uv sync
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

Optional LangSmith tracing variables are also supported:

```env
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=uno-rules-assistant
```

The PDF rulebooks should be placed in [`data/`](data/). The included rulebooks
are loaded automatically.

## Usage

### Streamlit app

```powershell
uv run streamlit run app.py
```

Open the displayed local URL and ask questions such as:

```text
What happens when I play a Skip card in UNO Flip?
```

### Command line

Ingest the rulebooks and build the local ChromaDB index:

```powershell
uv run main.py ingest
```

Ask a question from the command line:

```powershell
uv run main.py query "What happens when you play a Skip card?"
```

Ingestion is skipped when the vector store already contains documents.
Generated vector-store data is stored in `chroma_db_lc/` and is ignored by
Git. Delete that directory if you need to rebuild the index from scratch.

## Project structure

```text
app.py              Streamlit user interface
main.py             Command-line ingestion and query interface
data/               PDF rulebooks
src/
  config.py         Environment variables and model settings
  pipeline.py       Loading, chunking, retrieval, and RAG chain
  embeddings.py     Embedding helpers
  generator.py      Answer generation helpers
  loader.py         Document-loading helpers
  vectorstore.py    Vector-store and hybrid-search helpers
```

## Notes

- Answers are constrained to the retrieved rulebook context.
- Source documents are shown with each answer in the Streamlit app.
- Keep API keys in `.env`; do not commit them.