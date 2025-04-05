package backend

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
)

// Struktura zapytania do API Ollama
type OllamaRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

// Funkcja do generowania odpowiedzi z modelu Llama3 przez Ollama API
func AskWithContext(ctx context.Context, host, model, query, contextText string) (string, error) {
	// Tworzymy zapytanie do API Ollama
	requestData := OllamaRequest{
		Model:  model,
		Prompt: contextText + "\n\n" + query,
	}

	// Kodowanie zapytania do formatu JSON
	jsonData, err := json.Marshal(requestData)
	if err != nil {
		return "", fmt.Errorf("błąd kodowania zapytania: %w", err)
	}

	// Tworzymy zapytanie HTTP
	url := fmt.Sprintf("%s/api/generate", host)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
	if err != nil {
		return "", fmt.Errorf("błąd tworzenia zapytania HTTP: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Logowanie zapytania przed wysłaniem (do debugowania)
	log.Printf("Wysyłane zapytanie do %s: %s\n", url, string(jsonData))

	// Wysyłanie zapytania do serwera
	client := &http.Client{
		Timeout: 30 * time.Second, // Zwiększenie czasu oczekiwania
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("błąd wysyłania zapytania HTTP: %w", err)
	}
	defer resp.Body.Close()

	// Logowanie statusu odpowiedzi
	log.Printf("Odpowiedź z serwera Ollama - Status: %d", resp.StatusCode)

	// Odczytanie odpowiedzi z serwera
	var respData map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&respData)
	if err != nil {
		return "", fmt.Errorf("błąd dekodowania odpowiedzi: %w", err)
	}

	// Logowanie odpowiedzi z serwera (do debugowania)
	log.Printf("Odpowiedź z serwera Ollama: %+v", respData)

	// Sprawdzamy, czy odpowiedź zawiera pole "response"
	if response, ok := respData["response"].(string); ok {
		return response, nil
	}

	return "", fmt.Errorf("nie znaleziono pola 'response' w odpowiedzi: %+v", respData)
}
