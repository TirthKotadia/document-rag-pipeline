import streamlit as st
from src.pipeline import ingest, build_chain

st.title("UNO Rules Assistant")
st.caption("Ask questions about UNO Classic, Flip, and Extreme rules")

@st.cache_resource
def load_chain():
    vectorstore, chunks = ingest()
    chain, retriever = build_chain(vectorstore, chunks)
    return chain, retriever

chain, retriever = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question := st.chat_input("Ask about UNO rules..."):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.spinner("Thinking..."):
        answer = chain.invoke(question)
        sources = retriever.invoke(question)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

    with st.expander("Sources used"):
        for doc in sources:
            st.write(f"- {doc.metadata.get('source', 'unknown')}")