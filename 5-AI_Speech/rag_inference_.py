from llama_cpp import Llama
import sentence_transformers
import chromadb

# Ścieżka do pobranego modelu GGUF
MODEL_PATH = "models/mistral-7b-v0.1.Q4_K_S.gguf" # Zmień na ścieżkę do Twojego modelu

# Inicjalizacja modelu Llama
llm = Llama(model_path=MODEL_PATH, n_ctx=2048) # Dostosuj n_ctx w zależności od modelu

# Inicjalizacja modelu Sentence Transformers do generowania osadzeń zapytań
embedding_model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")

# Połączenie z ChromaDB
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_collection("shop_manual")

def retrieve_context(query, top_n=3):
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_n
    )
    return results['documents'][0] if results and results['documents'] else []

def generate_response(query, context):
    prompt = f"""Poniżej znajduje się pytanie użytkownika oraz fragmenty kontekstu z bazy wiedzy. Użyj tych fragmentów kontekstu, aby odpowiedzieć na pytanie. Jeśli nie znasz odpowiedzi na podstawie kontekstu, poinformuj o tym.

    Kontekst:
    {' '.join(context)}

    Pytanie: {query}

    Odpowiedź: """
    output = llm(prompt, max_tokens=512, stop=["Q:", "\n\n"], echo=False)
    return output['choices'][0]['text'].strip()

if __name__ == "__main__":
    user_query = "Jakie eydtować pracownika?"
    context = retrieve_context(user_query)
    if context:
        response = generate_response(user_query, context)
        print(f"Pytanie: {user_query}")
        print(f"Odpowiedź: {response}")
    else:
        print("Nie znaleziono relevantnych informacji w bazie wiedzy.")