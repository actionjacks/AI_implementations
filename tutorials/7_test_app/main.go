package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/qdrant/go-client/qdrant"
	"github.com/unidoc/unipdf/v3/extractor"
	"github.com/unidoc/unipdf/v3/model"
)

const (
	collectionName = "pdf_collection"
	vectorSize     = 1024 // mxbai-embed-large zwraca 1024-wymiarowe embeddingi
	ollamaURLStr   = "http://localhost:11434"
	qdrantHost     = "localhost"
	qdrantPort     = 6334
	embeddingModel = "mxbai-embed-large" // Nazwa modelu Ollama do embeddingów
	chatModel      = "llama3"            // Nazwa modelu Ollama do czatu
	pdfFilePath    = "example.pdf"       // Ścieżka do pliku PDF
)

func main() {
	// 0. Sprawdzenie czy plik PDF istnieje
	if _, err := os.Stat(pdfFilePath); os.IsNotExist(err) {
		log.Fatalf("Plik PDF nie istnieje: %v", err)
	}

	// 1. Ekstrakcja tekstu z PDF
	pdfText, err := extractTextFromPDF(pdfFilePath)
	if err != nil {
		log.Fatalf("Błąd ekstrakcji PDF: %v", err)
	}
	chunks := splitTextIntoChunks(pdfText, 500)

	// 2. Generowanie embeddingów
	ollamaURL, err := url.Parse(ollamaURLStr)
	if err != nil {
		log.Fatalf("Błąd parsowania URL Ollama: %v", err)
	}
	ollamaClient := api.NewClient(ollamaURL, http.DefaultClient)

	embeddings, err := generateEmbeddings(ollamaClient, embeddingModel, chunks)
	if err != nil {
		log.Fatalf("Błąd generowania embeddingów: %v", err)
	}

	// 3. Zapis do Qdrant
	qdrantClient, err := qdrant.NewClient(&qdrant.Config{
		Host: qdrantHost,
		Port: qdrantPort,
	})
	if err != nil {
		log.Fatalf("Nie można połączyć się z Qdrant: %v", err)
	}
	defer qdrantClient.Close()

	ctx := context.Background()

	// Utwórz kolekcję (ignoruj błąd, jeśli już istnieje)
	err = qdrantClient.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: collectionName,
		VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
			Size:     vectorSize,
			Distance: qdrant.Distance_Cosine,
		}),
	})
	if err != nil && !strings.Contains(err.Error(), "already exists") {
		log.Fatalf("Błąd tworzenia kolekcji: %v", err)
	} else if err != nil {
		log.Printf("Kolekcja już istnieje: %v", err)
	}

	// Przygotuj punkty do zapisu
	points := make([]*qdrant.PointStruct, len(chunks))
	for i, chunk := range chunks {
		pointID := qdrant.NewIDNum(uint64(i + 1))
		vectors := qdrant.NewVectors(embeddings[i]...)
		payload := map[string]any{"text": chunk}
		points[i] = &qdrant.PointStruct{
			Id:      pointID,
			Vectors: vectors,
			Payload: qdrant.NewValueMap(payload),
		}
	}

	// Zapis punktów do Qdrant
	operationInfo, err := qdrantClient.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: collectionName,
		Points:         points,
	})
	if err != nil {
		log.Fatalf("Błąd upsert: %v", err)
	}
	log.Printf("Upsert operation info: %+v", operationInfo)

	// 4. Wyszukiwanie i generowanie odpowiedzi
	question := "Jakie są główne wnioski z dokumentu?" // Przykładowe pytanie
	answer, err := getAnswerFromDocument(ollamaClient, qdrantClient, chatModel, embeddingModel, collectionName, question)
	if err != nil {
		log.Fatalf("Błąd podczas odpowiadania na pytanie: %v", err)
	}

	fmt.Printf("Pytanie: %s\nOdpowiedź: %s\n", question, answer)
}

// Funkcja do ekstrakcji tekstu z pliku PDF
func extractTextFromPDF(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("nie można otworzyć pliku: %v", err)
	}
	defer file.Close()

	pdfReader, err := model.NewPdfReader(file)
	if err != nil {
		return "", fmt.Errorf("błąd czytania PDF: %v", err)
	}

	var text strings.Builder
	numPages, err := pdfReader.GetNumPages()
	if err != nil {
		return "", fmt.Errorf("błąd pobierania liczby stron: %v", err)
	}

	for i := 1; i <= numPages; i++ {
		page, err := pdfReader.GetPage(i)
		if err != nil {
			return "", fmt.Errorf("błąd pobierania strony %d: %v", i, err)
		}

		ex, err := extractor.New(page)
		if err != nil {
			return "", fmt.Errorf("błąd inicjalizacji ekstraktora: %v", err)
		}

		pageText, err := ex.ExtractText()
		if err != nil {
			return "", fmt.Errorf("błąd ekstrakcji tekstu ze strony %d: %v", i, err)
		}
		text.WriteString(pageText + "\n")
	}
	return text.String(), nil
}

// Funkcja do dzielenia tekstu na fragmenty
func splitTextIntoChunks(text string, chunkSize int) []string {
	var chunks []string
	for i := 0; i < len(text); i += chunkSize {
		end := i + chunkSize
		if end > len(text) {
			end = len(text)
		}
		chunks = append(chunks, text[i:end])
	}
	return chunks
}

// Funkcja do generowania embeddingów dla wielu fragmentów tekstu
func generateEmbeddings(client *api.Client, model string, texts []string) ([][]float32, error) {
	embeddings := make([][]float32, 0, len(texts)) // Alokacja z góry, żeby uniknąć realokacji
	for _, text := range texts {
		embedding, err := generateEmbedding(client, model, text)
		if err != nil {
			return nil, fmt.Errorf("błąd generowania embeddingu dla tekstu '%s': %v", text, err)
		}
		embeddings = append(embeddings, embedding)
	}
	return embeddings, nil
}

// Funkcja do generowania embeddingu dla pojedynczego tekstu
func generateEmbedding(client *api.Client, model, text string) ([]float32, error) {
	resp, err := client.Embeddings(context.Background(), &api.EmbeddingRequest{
		Model:  model,
		Prompt: text,
	})
	if err != nil {
		return nil, fmt.Errorf("błąd generowania embeddingu (%s): %v", model, err)
	}

	embedding32 := make([]float32, len(resp.Embedding))
	for i, v := range resp.Embedding {
		embedding32[i] = float32(v)
	}
	return embedding32, nil
}

// Funkcja do pobierania kontekstu z Qdrant
func getContextFromQdrant(qdrantClient *qdrant.Client, collectionName string, questionEmbedding []float32) (string, error) {
	limit := uint64(5) // Konwersja int do uint64
	searchResult, err := qdrantClient.Query(context.Background(), &qdrant.QueryPoints{
		CollectionName: collectionName,
		Query:          qdrant.NewQuery(questionEmbedding...),
		WithPayload:    qdrant.NewWithPayload(true),
		Limit:          &limit, // Użycie adresu zmiennej typu *uint64
	})
	if err != nil {
		return "", fmt.Errorf("błąd wyszukiwania w Qdrant: %v", err)
	}

	var contextBuilder strings.Builder
	for _, result := range searchResult {
		if payload := result.GetPayload(); payload != nil {
			if textValue, ok := payload["text"]; ok {
				if text := textValue.GetStringValue(); text != "" {
					contextBuilder.WriteString(text + "\n\n")
				} else {
					log.Printf("Oczekiwano string w payloadzie 'text', znaleziono inny typ.")
				}
			}
		}
	}
	return contextBuilder.String(), nil
}

// Funkcja do zadawania pytania Ollamie z kontekstem
func askOllamaWithContext(ollamaClient *api.Client, chatModel, question, contextStr string) (string, error) {
	prompt := fmt.Sprintf(`
	Odpowiedz na pytanie na podstawie poniższych fragmentów dokumentu. 
	Jeśli nie znasz odpowiedzi, powiedz "Nie wiem".
	Pytanie: %s Kontekst: %s Odpowiedź:`, question, contextStr)

	var response string
	err := ollamaClient.Generate(context.Background(), &api.GenerateRequest{
		Model:  chatModel,
		Prompt: prompt,
		Options: map[string]interface{}{
			"temperature": 0.3,
		},
	}, func(genResp api.GenerateResponse) error {
		response += genResp.Response
		return nil
	})
	if err != nil {
		return "", fmt.Errorf("błąd generowania odpowiedzi przez Ollama: %v", err)
	}
	return response, nil
}

// Funkcja do zadawania pytania i uzyskiwania odpowiedzi z kontekstem z Qdrant i Ollamy
func getAnswerFromDocument(ollamaClient *api.Client, qdrantClient *qdrant.Client, chatModel, embeddingModel, collectionName, question string) (string, error) {
	// 1. Generowanie embeddingu dla pytania
	questionEmbedding, err := generateEmbedding(ollamaClient, embeddingModel, question)
	if err != nil {
		return "", fmt.Errorf("błąd generowania embeddingu pytania: %v", err)
	}

	// 2. Pobranie kontekstu z Qdrant
	context, err := getContextFromQdrant(qdrantClient, collectionName, questionEmbedding)
	if err != nil {
		return "", err
	}

	// 3. Zadanie pytania Ollamie z kontekstem
	answer, err := askOllamaWithContext(ollamaClient, chatModel, question, context)
	if err != nil {
		return "", err
	}
	return answer, nil
}
