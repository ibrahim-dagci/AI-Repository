import chromadb
import ollama
import fitz
import os

chroma_client = chromadb.PersistentClient(path=f"{os.path.dirname(__file__)}/chroma")
collection = chroma_client.get_or_create_collection(name="documents")

def embed_text(text):
    response = ollama.embeddings(model="nomic-embed-text",prompt=text)
    return response["embedding"]

def add_document(id, text):
    embedding = embed_text(text)
    collection.add(ids=[id], embeddings=[embedding], metadatas=[{"text": text}])

def load_pdf_and_store(file_path = f"{os.path.dirname(__file__)}/belge.pdf"):
    existing_docs = collection.get(include=["metadatas"])

    if existing_docs["metadatas"]:
        print("\033[93mPDF zaten işlenmiş, tekrar eklenmeyecek.\033[0m")
        return
    
    doc = fitz.open(file_path)

    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()

        paragraphs = text.split("\n\n")

        for i, paragraph in enumerate(paragraphs):
            clean_paragraph = paragraph.strip()
            if clean_paragraph:
                doc_id = f"page{page_num}para{i}"
                add_document(doc_id, clean_paragraph)
    print(f"chromaDB created -> \033[92m{os.path.dirname(__file__)}/belge.pdf\033[0m")

