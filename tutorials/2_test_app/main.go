package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"rag-app/backend"
	"rag-app/embedding"
	"rag-app/parser"
	"rag-app/qdrant"
)

func main() {
	ctx := context.Background()

	ollamaHost := "http://localhost:11434"
	embeddingModel := "mxbai-embed-large"
	llmModel := "llama3:8b"

	file := "docs/example.pdf" // lub .txt / .csv

	// Parsowanie pliku
	text, err := parser.ParseFile(file)
	if err != nil {
		log.Fatal("❌ Błąd parsowania pliku:", err)
	}
	fmt.Println("Zawartość pliku:", text)

	// Generowanie embeddingu
	vec, err := embedding.GetEmbedding(ctx, ollamaHost, embeddingModel, text)
	if err != nil {
		log.Fatal("❌ Błąd generowania embeddingu: ", err)
	}
	fmt.Println("Wektor ma rozmiar:", len(vec))

	// Zapisywanie embeddingu do Qdrant
	err = qdrant.SaveEmbedding(ctx, vec, text, file)
	if err != nil {
		log.Fatal("❌ Błąd zapisu do Qdrant:", err)
	}

	// Nowe pytanie
	question := "O czym jest dokument dotyczący katalogu towarowego i bufora?"
	fullContext := text + "\n\n" + question // Łączenie tekstu z pytaniem
	answer, err := backend.AskWithContext(ctx, ollamaHost, llmModel, question, fullContext)
	if err != nil {
		log.Fatal("❌ Błąd generowania odpowiedzi:", err)
	}

	// Jeśli odpowiedź z modelu jest zbyt ogólna, przechodzimy do Qdrant
	if isAnswerTooGeneral(answer) {
		fmt.Println("🔍 Odpowiedź z modelu jest zbyt ogólna. Szukam w Qdrant...")

		// Wykonywanie wyszukiwania w Qdrant
		qdrantAnswer, err := qdrant.SearchInQdrant(ctx, question) // Funkcja, która szuka w Qdrant
		if err != nil {
			log.Fatal("❌ Błąd podczas wyszukiwania w Qdrant:", err)
		}

		// Wyświetlanie odpowiedzi z Qdrant
		fmt.Println("\n🔍 Odpowiedź z Qdrant:", qdrantAnswer)
	} else {
		// Wyświetlanie odpowiedzi z modelu
		fmt.Println("\n🗣️ Pytanie:", question)
		fmt.Println("💬 Odpowiedź z modelu:", answer)
	}
}

// Funkcja sprawdzająca, czy odpowiedź jest zbyt ogólna
func isAnswerTooGeneral(answer string) bool {
	// Może to być sprawdzenie, czy odpowiedź jest zbyt krótka lub zawiera ogólniki
	return strings.TrimSpace(answer) == "Ten" || len(strings.TrimSpace(answer)) < 5
}
