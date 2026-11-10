import os
import fitz  # pymupdf

def load_documents(data_dir: str = "data") -> list[dict]:
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            documents.append({
                "source": filename,
                "text": text
            })
    return documents