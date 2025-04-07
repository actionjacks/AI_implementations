package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/gookit/color" // Dodano bibliotekę kolorów
	"github.com/ledongthuc/pdf"
	"github.com/ollama/ollama/api"
	"github.com/qdrant/go-client/qdrant"
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

// Funkcja do wyświetlania animacji spinnera
func showSpinner(done chan bool) {
	spinner := []string{"|", "/", "-", "\\"}
	i := 0
	for {
		select {
		case <-done:
			return
		default:
			fmt.Printf("\rOczekiwanie na odpowiedź... %s", spinner[i])
			i = (i + 1) % len(spinner)
			time.Sleep(100 * time.Millisecond) // Zmniejszono opóźnienie dla lepszej widoczności
		}
	}
}

func main() {
	// Definiowanie flagi wiersza poleceń
	keepHistory := flag.Bool("keep-history", false, "Czy model ma pamiętać historię rozmowy")
	flag.Parse() // Parsowanie flag

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

	// Pętla interaktywna
	reader := bufio.NewReader(os.Stdin)
	conversationHistory := "" // Inicjalizacja pustej historii konwersacji

	for {
		color.Printf("<blue>Zadaj pytanie</>: ") // Pytanie użytkownika na niebiesko
		question, _ := reader.ReadString('\n')
		question = strings.TrimSpace(question)

		if strings.ToLower(question) == "koniec" {
			color.Println("<yellow>Zakończono.</>") // Informacja o zakończeniu na żółto
			break
		}

		if question == "" {
			color.Println("<red>Pytanie nie może być puste. Spróbuj ponownie.</>") // Komunikat o błędzie na czerwono
			continue
		}

		// Dodaj pytanie do historii, jeśli flaga jest ustawiona
		if *keepHistory {
			conversationHistory += fmt.Sprintf("Pytanie: %s\n", question)
		}

		// Uruchom spinner
		done := make(chan bool)
		go showSpinner(done)

		answer, err := getAnswerFromDocument(ollamaClient, qdrantClient, chatModel, embeddingModel, collectionName, question, conversationHistory)
		// Zatrzymaj spinner
		close(done)
		fmt.Println() // Przejdź do nowej linii po zatrzymaniu spinnera

		if err != nil {
			log.Printf("Błąd podczas odpowiadania na pytanie: %v", err)
			color.Println("<red>Przepraszam, wystąpił problem z uzyskaniem odpowiedzi.</>") // Komunikat o błędzie na czerwono
			continue
		}

		// Dodaj odpowiedź do historii, jeśli flaga jest ustawiona
		if *keepHistory {
			conversationHistory += fmt.Sprintf("Odpowiedź: %s\n", answer)
		}
		color.Printf("<blue>Pytanie:</> %s\n<green>Odpowiedź:</> %s\n", question, answer) // Pytanie na niebiesko, odpowiedź na zielono
	}
}

// Funkcja do ekstrakcji tekstu z pliku PDF
func extractTextFromPDF(path string) (string, error) {
	f, r, err := pdf.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	var buf strings.Builder
	b, err := r.GetPlainText()
	if err != nil {
		return "", err
	}
	_, err = io.Copy(&buf, b)
	return buf.String(), err
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
	embeddings := make([][]float32, 0, len(texts))
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
	limit := uint64(5)
	searchResult, err := qdrantClient.Query(context.Background(), &qdrant.QueryPoints{
		CollectionName: collectionName,
		Query:          qdrant.NewQuery(questionEmbedding...),
		WithPayload:    qdrant.NewWithPayload(true),
		Limit:          &limit,
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
    %s
    Odpowiedz na pytanie na podstawie poniższych fragmentów dokumentu. 
    Jeśli nie znasz odpowiedzi, powiedz "Nie wiem".
    Pytanie: %s Kontekst: %s Odpowiedź:`, contextStr, question, contextStr)

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
func getAnswerFromDocument(ollamaClient *api.Client, qdrantClient *qdrant.Client, chatModel, embeddingModel, collectionName, question, conversationHistory string) (string, error) {
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

	// 3. Zadanie pytania Ollamie z kontekstem i historią
	answer, err := askOllamaWithContext(ollamaClient, chatModel, question, context)
	if err != nil {
		return "", err
	}
	return answer, nil
}
