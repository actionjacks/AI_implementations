from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_pdf import retriever_pdf

model = OllamaLLM(model="llama3")

# Polski prompt
template = """
Jesteś ekspertem z Polski od analizy treści dokumentów aplikacji.
Poniżej znajdują się istotne fragmenty dokumentu: {dokumenty}
Odpowiedz na następujące pytanie: {pytanie}. 
Pamietaj żeby odpowiadać w języku polskim.
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    pytanie = input("Zadaj pytanie dotyczące pliku PDF (q aby wyjść): ")
    print("\n\n")
    if pytanie.lower() == "q":
        break

    # Pobierz istotne fragmenty z wektorowej bazy
    dokumenty = retriever_pdf.invoke(pytanie)
    teksty = "\n---\n".join([doc.page_content for doc in dokumenty])

    # Przekazujemy teksty i pytanie do modelu
    result = chain.invoke({"dokumenty": teksty, "pytanie": pytanie})
    print("Odpowiedź:\n", result)
