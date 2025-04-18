from PyPDF2 import PdfReader
import sentence_transformers

def load_and_chunk_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    # Prosty podział na zdania (może wymagać ulepszenia)
    sentences = text.split(". ")
    return [s.strip() + "." for s in sentences if s.strip()]

def generate_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    model = sentence_transformers.SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings

def add_to_chromadb(texts, embeddings):
    import chromadb

    # Połączenie z ChromaDB działającym w Dockerze (domyślny adres i port)
    client = chromadb.HttpClient(host="localhost", port=8000)

    # Tworzenie lub pobieranie kolekcji
    collection = client.get_or_create_collection("shop_manual")

    # Dodawanie tekstów i osadzeń
    collection.add(
        embeddings=embeddings.tolist(),
        documents=texts,
        ids=[f"doc_{i}" for i in range(len(texts))] # Generowanie unikalnych ID
    )

if __name__ == "__main__":
    pdf_file = "example_files/Instrukcja_Shop_HiperCloud_Gold.pdf"
    chunks = load_and_chunk_pdf(pdf_file)
    embeddings = generate_embeddings(chunks)
    add_to_chromadb(chunks, embeddings)

    print(f"Załadowano {len(chunks)} fragmentów i dodano je do ChromaDB.")