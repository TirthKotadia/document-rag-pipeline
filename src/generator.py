from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)

def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return f"""You are a helpful assistant answering questions about UNO card game rules.
Use ONLY the context below to answer. If the context doesn't contain the answer, say so clearly.
If different UNO variants (Classic, Flip, Extreme) have different rules for the same question, mention the differences explicitly.

Context:
{context}

Question: {question}

Answer:"""

def generate_answer(question: str, chunks: list[dict]) -> str:
    prompt = build_prompt(question, chunks)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content