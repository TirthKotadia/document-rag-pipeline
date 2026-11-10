import sys
from src.pipeline import ingest, build_chain


def run_ingest():
    ingest()


def run_query(question: str):
    vectorstore, chunks = ingest()
    chain, retriever = build_chain(vectorstore, chunks)

    answer = chain.invoke(question)

    print(f"\nQuestion: {question}\n")
    print("Answer:")
    print(answer)

    print("\n--- Sources used ---")
    for doc in retriever.invoke(question):
        print(f"- {doc.metadata.get('source', 'unknown')}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  uv run main.py ingest")
        print('  uv run main.py query "your question here"')
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        run_ingest()
    elif command == "query":
        if len(sys.argv) < 3:
            print('Please provide a question: uv run main.py query "what does skip do"')
            sys.exit(1)
        run_query(sys.argv[2])
    else:
        print(f"Unknown command: {command}")