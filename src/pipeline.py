import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from src.config import GEMINI_API_KEY, GROQ_API_KEY, GROQ_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

CHROMA_PATH_LC = "chroma_db_lc"
DATA_DIR = "data"


def load_and_split():
    """Loads every PDF in data/ and splits into chunks, preserving metadata."""
    all_docs = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(DATA_DIR, filename))
            all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(all_docs)


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
    )


def get_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        collection_name="uno_rules_lc",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH_LC,
    )


def ingest():
    """Ingests PDFs into Chroma (skips if already done) and returns both the
    vectorstore and the raw chunks (needed for BM25, which isn't persisted)."""
    vectorstore = get_vectorstore()
    chunks = load_and_split()

    if vectorstore._collection.count() > 0:
        print("Already ingested — skipping embedding step.")
    else:
        vectorstore.add_documents(chunks)
        print(f"Ingested {len(chunks)} chunks.")

    return vectorstore, chunks


def build_chain(vectorstore, all_chunks):
    """Builds a RAG chain using Self-Query retrieval — the LLM extracts both
    a semantic query AND a metadata filter (e.g. which UNO variant) from the
    user's question, then searches only within that filtered scope."""

    llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2)

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description=(
                "The UNO rulebook file this chunk came from. One of: "
                "'uno_classic_rules.pdf' (standard/original UNO), "
                "'uno_flip_rules.pdf' (UNO Flip variant with a light/dark side), "
                "'uno_extreme_rules.pdf' (UNO Extreme variant with a card-launcher)."
            ),
            type="string",
        ),
    ]

    self_query_retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents="Rules and instructions for playing various UNO card game variants.",
        metadata_field_info=metadata_field_info,
        search_kwargs={"k": 3},
    )

    prompt = ChatPromptTemplate.from_template("""You are a helpful assistant answering questions about UNO card game rules.
Use ONLY the context below to answer. If different UNO variants have different rules, mention the differences explicitly.
If the context doesn't contain the answer, say so clearly.

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )

    chain = (
        {"context": self_query_retriever | format_docs, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, self_query_retriever