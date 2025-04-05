package qdrant

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"rag-app/embedding"

	"github.com/google/uuid"
)

type Point struct {
	ID      string                 `json:"id"`
	Vector  []float64              `json:"vector"`
	Payload map[string]interface{} `json:"payload"`
}

type UpsertRequest struct {
	Ids    []string `json:"ids"`    // Lista identyfikatorów
	Points []Point  `json:"points"` // Lista punktów
}

type QdrantResponse struct {
	Status string  `json:"status"`
	Time   float64 `json:"time"`
}

// Funkcja do zapisania embeddingu w Qdrant
func SaveEmbedding(ctx context.Context, vector []float64, content, source string) error {
	// Sprawdzenie długości wektora
	fmt.Println("Długość wektora:", len(vector))

	// Tworzenie unikalnego ID
	id := uuid.New().String()

	// Payload z danymi
	payload := map[string]interface{}{
		"content":  content,
		"filename": source,
	}

	// Przygotowanie punktu
	point := Point{
		ID:      id,      // ID punktu
		Vector:  vector,  // Wektor
		Payload: payload, // Payload
	}

	// Przygotowanie zapytania do Qdrant
	req := UpsertRequest{
		Ids:    []string{id},   // Lista z jednym ID
		Points: []Point{point}, // Lista punktów
	}

	// Marshaling danych
	data, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("błąd marshalingu danych: %w", err)
	}

	// Wysyłanie zapytania do Qdrant
	url := "http://localhost:6333/collections/documents/points?wait=true"
	resp, err := http.Post(url, "application/json", bytes.NewBuffer(data))
	if err != nil {
		return fmt.Errorf("błąd wysyłania zapytania: %w", err)
	}
	defer resp.Body.Close()

	// Logowanie statusu odpowiedzi
	fmt.Println("Status odpowiedzi z Qdrant:", resp.StatusCode)

	// Logowanie odpowiedzi z Qdrant
	var respBody QdrantResponse
	err = json.NewDecoder(resp.Body).Decode(&respBody)
	if err != nil {
		return fmt.Errorf("błąd dekodowania odpowiedzi: %w", err)
	}

	// Logowanie całej odpowiedzi
	fmt.Println("Odpowiedź z Qdrant:", respBody)

	// Jeśli status odpowiedzi nie jest 200 OK, zwróć błąd
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("Qdrant returned status: %d - %v", resp.StatusCode, respBody)
	}

	// Jeśli wszystko poszło dobrze, logujemy sukces
	fmt.Println("✅ Zapisano embedding do Qdrant")
	return nil
}

type SearchRequest struct {
	Limit       int       `json:"limit"`
	Vector      []float64 `json:"vector"`
	WithPayload bool      `json:"with_payload"`
}

type SearchResponse struct {
	Result []struct {
		ID      string                 `json:"id"`
		Score   float64                `json:"score"`
		Payload map[string]interface{} `json:"payload"`
	} `json:"result"`
}

// Funkcja do wyszukiwania najbliższych sąsiadów w Qdrant
func SearchInQdrant(ctx context.Context, query string) (string, error) {
	// Generowanie embeddingu zapytania
	embeddingModel := "mxbai-embed-large"
	ollamaHost := "http://localhost:11434"
	queryVec, err := embedding.GetEmbedding(ctx, ollamaHost, embeddingModel, query)
	if err != nil {
		return "", fmt.Errorf("❌ Błąd generowania embeddingu zapytania: %w", err)
	}

	// Przygotowanie zapytania do Qdrant
	searchRequest := SearchRequest{
		Limit:       1, // Zwróć tylko najbliższy punkt
		Vector:      queryVec,
		WithPayload: true,
	}

	// Marshaling zapytania
	data, err := json.Marshal(searchRequest)
	if err != nil {
		return "", fmt.Errorf("❌ Błąd marshalingu zapytania: %w", err)
	}

	// Wysyłanie zapytania do Qdrant
	url := "http://localhost:6333/collections/documents/points/search"
	resp, err := http.Post(url, "application/json", bytes.NewBuffer(data))
	if err != nil {
		return "", fmt.Errorf("❌ Błąd wysyłania zapytania do Qdrant: %w", err)
	}
	defer resp.Body.Close()

	// Logowanie statusu odpowiedzi
	fmt.Println("Status odpowiedzi z Qdrant:", resp.StatusCode)

	// Dekodowanie odpowiedzi
	var respBody SearchResponse
	err = json.NewDecoder(resp.Body).Decode(&respBody)
	if err != nil {
		return "", fmt.Errorf("❌ Błąd dekodowania odpowiedzi: %w", err)
	}

	// Jeśli nie znaleziono żadnych wyników, zwróć odpowiedni komunikat
	if len(respBody.Result) == 0 {
		return "Brak wyników", nil
	}

	// Zwrócenie wyników jako odpowiedzi
	closestResult := respBody.Result[0] // Zwracamy najbliższy wynik
	return fmt.Sprintf("ID: %s, Score: %f, Payload: %v", closestResult.ID, closestResult.Score, closestResult.Payload), nil
}
