from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from PyPDF2 import PdfReader
import os

# Ścieżka do PDF
pdf_path = "example.pdf"
# Inicjalizacja modelu embeddingu
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Ścieżka do lokalnej bazy wektorowej
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

# Lista dokumentów i ID
documents = []
ids = []

# Wczytaj PDF
if add_documents:
    reader = PdfReader(pdf_path)
    pdf_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pdf_text += text.strip() + "\n"

    # Tworzymy jeden dokument z całego PDF
    pdf_document = Document(
        page_content=pdf_text,
        metadata={"source": pdf_path},
        id="pdf_1"
    )
    documents.append(pdf_document)
    ids.append("pdf_1")

# Tworzenie lub ładowanie wektorowej bazy
vector_store = Chroma(
    collection_name="pdf_collection",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Dodaj dokumenty jeśli to pierwsze uruchomienie
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Retriever do wyszukiwania
retriever_pdf = vector_store.as_retriever(
    search_kwargs={"k": 5})
