from rag.readPdf import load_pdf_and_store
import chromadb
import ollama
import os


chroma_client = chromadb.PersistentClient(path=f"{os.path.dirname(__file__)}/chroma")
collection = chroma_client.get_collection(name="documents")

def embed_text(text):
    response = ollama.embeddings(model="nomic-embed-text",prompt=text)
    return response["embedding"]

def retrieve_relevant_context(user_input, top_k=3):
    query_embedding = embed_text(user_input)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    relevant_context = " ".join([doc["text"] for doc in results["metadatas"][0]])

    return relevant_context

def chatbot():

    load_pdf_and_store()

    print("send for exit: exit or quit or çıkış")
    while True:
        user_input = input("Sen: ")
        if user_input.lower() in ["exit", "quit", "çıkış"]:
            break

        context = retrieve_relevant_context(user_input)
        #print(context)
        response = ollama.chat(
            model="deepseek-r1",
            messages = [
                {"role": "system", "content": f"When responding to the user's question, use the following context:\n{context}"},
                {"role": "user", "content": user_input}
            ]
        )

        print("DeepSeek:", response['message']['content'])
