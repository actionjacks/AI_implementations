import chromadb
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains import LLMChain
from PyPDF2 import PdfReader
from langchain_ollama import OllamaEmbeddings

# Ścieżka do PDF
pdf_path = "example.pdf"

# Połączenie z serwerem Chroma
client = chromadb.HttpClient(host='localhost', port=8000)

# Inicjalizacja modelu embeddingu
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Wczytaj PDF
reader = PdfReader(pdf_path)

# Tworzymy kolekcję dokumentów z PDF, przypisując unikalne ID na podstawie numeru strony
collection_name = "pdf_collection"
vector_store = client.get_or_create_collection(collection_name)  # Pobierz lub stwórz kolekcję
ids = []  # Lista identyfikatorów
documents = []  # Lista dokumentów

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        # Tworzymy unikalne ID jako ciąg znaków na podstawie numeru strony
        document_id = f"pdf_{i + 1}"  # ID dokumentu jako ciąg znaków
        pdf_document = Document(
            page_content=text.strip(),
            metadata={"source": pdf_path, "page": i + 1}
        )

        # Dodajemy ID oraz dokument do odpowiednich list
        ids.append(document_id)
        documents.append(pdf_document)

# Dodajemy dokumenty do Chroma w jednej operacji
# Przekazujemy tylko teksty (page_content) do bazy
vector_store.add(
    ids=ids,
    documents=[doc.page_content for doc in documents]
)

model = OllamaLLM(model="llama3")

# Przykładowy szablon promptu
template = """
Jesteś ekspertem z Polski od analizy treści dokumentów aplikacji.
Poniżej znajdują się istotne fragmenty dokumentu: {dokumenty}
Odpowiedz na następujące pytanie: {pytanie}. 
Pamietaj żeby odpowiadać w języku polskim.
"""
prompt = ChatPromptTemplate.from_template(template)
# Łączenie promptu z modelem
chain = prompt | model

# Funkcja do uzyskiwania odpowiedzi na pytanie
while True:
    print("\n\n-------------------------------")
    pytanie = input("Zadaj pytanie dotyczące pliku PDF (q aby wyjść): ")
    print("\n\n")
    if pytanie.lower() == "q":
        break

    # Pobierz istotne fragmenty z wektorowej bazy
    results = vector_store.query(query_texts=[pytanie], n_results=5)
    # teksty = "\n---\n".join([result[0] for result in results['documents']])  # Pobieramy tylko teksty dokumentów
    teksty = "\n---\n".join([result for result in results])  # Pobieramy tylko teksty dokumentów
    
    # Przekazujemy teksty i pytanie do modelu
    result = chain.invoke({"dokumenty": teksty, "pytanie": pytanie})
    print("Odpowiedź:\n", result)
