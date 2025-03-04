import ollama
import chromadb

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")

collection.add(
    documents=[
        "computer",
        "train",
        "car",
        "apple",
        "bike"
    ],
    ids=["id1", "id2","id3","id4","id5"]
)


model_name = "deepseek-r1"


while True:
    user_input = input("Sen: ")
    if user_input.lower() in ["exit", "quit", "çıkış"]:
        break

    results = collection.query(query_texts=[user_input], n_results=1)  # kaç sonuç döndürüleceği
    print(results['documents'])
    retrievedText = " ".join(results['documents'][0])
    print(retrievedText)
    response = ollama.chat(
        model="deepseek-r1",
        messages = [
            {"role": "system", "content": f"When responding to the user's question, use the following context:\n{retrievedText}"},
            {"role": "user", "content": user_input}
        ]

    )

    print("DeepSeek:", response['message']['content'])
