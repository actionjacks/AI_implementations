import chromadb
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains import LLMChain
from PyPDF2 import PdfReader
from langchain_ollama import OllamaEmbeddings

print("1. Inicjalizacja skryptu...")

# Ścieżka do PDF
pdf_path = "example.pdf"
print(f"2. Ścieżka do pliku PDF: {pdf_path}")

# Połączenie z serwerem Chroma
print("3. Łączenie z serwerem ChromaDB...")
client = chromadb.HttpClient(host='localhost', port=8000)
print("   Połączono z ChromaDB")

# Inicjalizacja modelu embeddingu
print("4. Inicjalizacja modelu embeddingu...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
print("   Model embeddingu zainicjalizowany")

# Wczytaj PDF
print("5. Wczytywanie pliku PDF...")
reader = PdfReader(pdf_path)
num_pages = len(reader.pages)
print(f"   Liczba stron w PDF: {num_pages}")

# Tworzymy kolekcję dokumentów z PDF
print("6. Przygotowywanie dokumentów do ChromaDB...")
collection_name = "pdf_collection"
print(f"   Nazwa kolekcji: {collection_name}")

vector_store = client.get_or_create_collection(collection_name)
print(f"   Kolekcja '{collection_name}' gotowa")

ids = []
documents = []
metadatas = []

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        document_id = f"pdf_{i + 1}"
        metadata = {"source": pdf_path, "page": i + 1}
        
        pdf_document = Document(
            page_content=text.strip(),
            metadata=metadata
        )

        ids.append(document_id)
        documents.append(pdf_document)
        metadatas.append(metadata)

print(f"   Przygotowano {len(documents)} dokumentów do zapisu")
print("   Przykładowy dokument:")
print("   ID:", ids[0] if ids else "brak")
print("   Treść:", documents[0].page_content[:100] + "..." if documents else "brak")
print("   Metadane:", metadatas[0] if metadatas else "brak")

# Dodajemy dokumenty do Chroma
print("7. Zapisywanie dokumentów do ChromaDB...")
vector_store.add(
    ids=ids,
    documents=[doc.page_content for doc in documents],
    metadatas=metadatas
)
print("   Dokumenty zapisane do ChromaDB")

# Inicjalizacja modelu LLM
print("8. Inicjalizacja modelu językowego...")
model = OllamaLLM(model="llama3")
print("   Model językowy gotowy")

# Szablon promptu
print("9. Przygotowanie szablonu promptu...")
template = """
Jesteś ekspertem z Polski od analizy treści dokumentów aplikacji.
Poniżej znajdują się istotne fragmenty dokumentu: {dokumenty}
Odpowiedz na następujące pytanie: {pytanie}. 
Pamietaj żeby odpowiadać w języku polskim.
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
print("   Łańcuch (chain) gotowy do użycia")

# Główna pętla pytań
print("\n10. Gotowy do przyjmowania pytań...")
print("------------------------------------")

while True:
    print("\n\n-------------------------------")
    pytanie = input("Zadaj pytanie dotyczące pliku PDF (q aby wyjść): ")
    print("\n\n")
    
    if pytanie.lower() == "q":
        print("11. Zamykanie aplikacji...")
        break

    print(f"12. Przetwarzanie pytania: '{pytanie}'")
    
    # Pobierz istotne fragmenty z ChromaDB
    print("13. Wyszukiwanie w ChromaDB...")
    results = vector_store.query(
        query_texts=[pytanie],
        n_results=5
    )
    
    print("\n14. Wyniki wyszukiwania w ChromaDB:")
    print("   Struktura wyników:", type(results))
    print("   Klucze w wynikach:", results.keys())
    
    if 'documents' in results:
        print("\n15. Znalezione dokumenty:")
        for i, doc_list in enumerate(results['documents']):
            for j, doc in enumerate(doc_list):
                print(f"   Dokument {i+1}-{j+1}: {doc[:200]}...")
        
        teksty = "\n---\n".join([doc for sublist in results['documents'] for doc in sublist])
        print("\n16. Połączone teksty dokumentów:")
        print(teksty[:500] + "..." if len(teksty) > 500 else teksty)
    else:
        teksty = "Nie znaleziono odpowiednich fragmentów w dokumencie."
        print("   Brak dokumentów w wynikach")

    # Przekazujemy teksty i pytanie do modelu
    print("\n17. Przesyłanie zapytania do modelu językowego...")
    result = chain.invoke({"dokumenty": teksty, "pytanie": pytanie})
    
    print("\n18. Pełna odpowiedź modelu:")
    print("Odpowiedź:\n", result)
    print("\n------------------------------------")