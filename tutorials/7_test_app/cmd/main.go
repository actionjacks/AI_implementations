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

	"github.com/ledongthuc/pdf"
	"github.com/ollama/ollama/api"
	"github.com/pterm/pterm"
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

// Funkcja do wyświetlania animacji spinnera PTerm
func showSpinner(done chan bool) {
	spinner, _ := pterm.DefaultSpinner.
		WithRemoveWhenDone(true).
		// WithSequence(" ", "█", " ", "█", " ").
		Start("Oczekiwanie na odpowiedź...")

	for {
		select {
		case <-done:
			spinner.Stop()
			return
		default:
			time.Sleep(100 * time.Millisecond)
		}
	}
}

// go run main.go -use-context=false
func main() {
	// Definiowanie flagi wiersza poleceń
	keepHistory := flag.Bool("keep-history", false, "Czy model ma pamiętać historię rozmowy")
	useContext := flag.Bool("use-context", true, "Czy model ma używać kontekstu z dokumentu") // Nowa flaga
	flag.Parse()

	// 0. Sprawdzenie czy plik PDF istnieje
	if _, err := os.Stat(pdfFilePath); os.IsNotExist(err) {
		pterm.Fatal.Printf("Plik PDF nie istnieje: %v\n", err)
	}

	// 1. Ekstrakcja tekstu z PDF
	pdfText, err := extractTextFromPDF(pdfFilePath)
	if err != nil {
		pterm.Fatal.Printf("Błąd ekstrakcji PDF: %v\n", err)
	}
	chunks := splitTextIntoChunks(pdfText, 500)

	// 2. Generowanie embeddingów
	ollamaURL, err := url.Parse(ollamaURLStr)
	if err != nil {
		pterm.Fatal.Printf("Błąd parsowania URL Ollama: %v\n", err)
	}
	ollamaClient := api.NewClient(ollamaURL, http.DefaultClient)

	embeddings, err := generateEmbeddings(ollamaClient, embeddingModel, chunks)
	if err != nil {
		pterm.Fatal.Printf("Błąd generowania embeddingów: %v\n", err)
	}

	// 3. Zapis do Qdrant
	qdrantClient, err := qdrant.NewClient(&qdrant.Config{
		Host: qdrantHost,
		Port: qdrantPort,
	})
	if err != nil {
		pterm.Fatal.Printf("Nie można połączyć się z Qdrant: %v\n", err)
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
		pterm.Fatal.Printf("Błąd tworzenia kolekcji: %v\n", err)
	} else if err != nil {
		pterm.Info.Printf("Kolekcja już istnieje: %v\n", err)
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
		pterm.Fatal.Printf("Błąd upsert: %v\n", err)
	}
	pterm.Info.Printf("Upsert operation info: %+v\n", operationInfo)

	// Pętla interaktywna
	reader := bufio.NewReader(os.Stdin)
	conversationHistory := ""

	pterm.Success.Printf("Zaczynaj rozmowę z modelem _ %s\n", chatModel)
	for {
		question, _ := reader.ReadString('\n')
		question = strings.TrimSpace(question)

		if strings.ToLower(question) == "koniec" {
			pterm.Info.Printf("Zamykam aplikację.")
			break
		}

		if question == "" {
			pterm.Error.Printf("Pytanie nie może być puste. Spróbuj ponownie.")
			continue
		}

		// Dodaj pytanie do historii
		if *keepHistory {
			conversationHistory += fmt.Sprintf("Pytanie: %s\n", question)
		}

		// Uruchom spinner
		done := make(chan bool)
		go showSpinner(done)

		var answer string
		var err error

		if *useContext { // Użyj kontekstu z Qdrant, jeśli flaga jest ustawiona
			answer, err = getAnswerFromDocument(ollamaClient, qdrantClient, chatModel, embeddingModel, collectionName, question, conversationHistory)
		} else {
			// Jeśli flaga useContext jest false, po prostu zapytaj Ollamę bez kontekstu
			answer, err = askOllama(ollamaClient, chatModel, question, conversationHistory)
		}

		close(done)
		pterm.Println()

		if err != nil {
			pterm.Error.Printf("Przepraszam, wystąpił problem z uzyskaniem odpowiedzi.: %v\n", err)
			continue
		}

		// Dodaj odpowiedź do historii
		if *keepHistory {
			conversationHistory += fmt.Sprintf("Odpowiedź: %s\n", answer)
		}

		// Wyświetl odpowiedź i czas
		pterm.Success.Printf("Pytanie: %s\n", question)
		pterm.Description.Printf("Odpowiedź: %s\n", answer)
		pterm.Info.Printf("Czas odpowiedzi: %s\n", time.Since(time.Now()).String())
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
	// Komentarz: Ta funkcja ekstrahuje tekst z pliku PDF.
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
	// Komentarz: Ta funkcja dzieli tekst na mniejsze fragmenty.
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
	// Komentarz: Ta funkcja generuje embeddingi dla listy tekstów.
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
	// Komentarz: Ta funkcja generuje embedding dla pojedynczego tekstu.
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
	// Komentarz: Ta funkcja pobiera kontekst z Qdrant na podstawie zapytania.
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
	// Komentarz: Ta funkcja wysyła zapytanie do Ollamy z kontekstem.
}

// Funkcja do zadawania pytania Ollamie bez kontekstu
func askOllama(ollamaClient *api.Client, chatModel, question string, conversationHistory string) (string, error) {
	prompt := fmt.Sprintf(`%s
    Odpowiedz na pytanie: %s Odpowiedź:`, conversationHistory, question)

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
	// Komentarz: Ta funkcja wysyła zapytanie do Ollamy bez kontekstu.
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
	// Komentarz: Ta funkcja koordynuje proces uzyskiwania odpowiedzi na pytanie.
}
