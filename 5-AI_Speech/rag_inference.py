from llama_cpp import Llama
import sentence_transformers
import chromadb
from piper import PiperVoice  # Poprawny import klasy PiperVoice
import sounddevice as sd
import numpy as np
import json

# Ścieżka do pobranego modelu GGUF
MODEL_PATH = "models/mistral-7b-v0.1.Q3_K_S.gguf" # Zmień na ścieżkę do Twojego modelu

# Ścieżki do modelu i konfiguracji Piper
VOICE_ONNX_PATH = "voices/pl_PL-darkman-medium.onnx" # Zmień na ścieżkę do Twojego pliku .onnx
VOICE_CONFIG_PATH = "voices/pl_PL-darkman-medium.onnx.json" # Zmień na ścieżkę do Twojego pliku .json

# Inicjalizacja modelu Llama
llm = Llama(model_path=MODEL_PATH, n_ctx=2048) # Dostosuj n_ctx w zależności od modelu

# Inicjalizacja modelu Sentence Transformers do generowania osadzeń zapytań
embedding_model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")

# Połączenie z ChromaDB
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_collection("shop_manual")

# Inicjalizacja Piper TTS
synthesizer = PiperVoice.load(VOICE_ONNX_PATH, VOICE_CONFIG_PATH) # Użycie metody statycznej load()

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

def synthesize_and_play(text):
    # Zakładam, że metoda synthesize zwraca numpy array lub bytes
    audio = synthesizer.synthesize_ids_to_raw(synthesizer.phonemes_to_ids(synthesizer.phonemize(text)[0])) # Przetwarzanie tekstu na audio
    audio_np = np.frombuffer(audio, dtype=np.int16)
    sd.play(audio_np, samplerate=synthesizer.config.sample_rate)
    sd.wait()

if __name__ == "__main__":
    user_query = "JAk założyć Inwentaryzację?"
    context = retrieve_context(user_query)
    if context:
        response = generate_response(user_query, context)
        print(f"Pytanie: {user_query}")
        print(f"Odpowiedź: {response}")
        synthesize_and_play(response)
    else:
        print("Nie znaleziono relevantnych informacji w bazie wiedzy.")