package rag

import (
	"context"
	"fmt"
	"net/url"

	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/memory"

	"github.com/hantmac/langchaingo-ollama-rag/rag/logger"

	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/qdrant"
)

var (
	collectionName = "langchaingo-ollama-rag"
	qdrantUrl      = "http://localhost:6333"
	ollamaServer   = "http://localhost:11434"
)

// getOllamaEmbedder Pobiera embedder Ollama
func getOllamaEmbedder() *embeddings.EmbedderImpl {
	logger.Debug("Inicjalizowanie embeddera Ollama...")
	// Tworzenie nowego modelu Ollama, model "nomic-embed-text:latest"
	ollamaEmbedderModel, err := ollama.New(
		ollama.WithModel("nomic-embed-text:latest"),
		ollama.WithServerURL(ollamaServer))
	if err != nil {
		logger.Fatal("Nie udało się utworzyć modelu Ollama: %v", err)
	}
	// Tworzenie nowego embeddera przy użyciu stworzonego modelu Ollama
	ollamaEmbedder, err := embeddings.NewEmbedder(ollamaEmbedderModel)
	if err != nil {
		logger.Fatal("Nie udało się utworzyć embeddera Ollama: %v", err)
	}
	logger.Debug("Embedder Ollama został pomyślnie utworzony.")
	return ollamaEmbedder
}

func getOllamaQwen() *ollama.LLM {
	logger.Debug("Inicjalizowanie modelu Ollama Qwen...")
	// Tworzenie nowego modelu Ollama, model "qwena:1.8b"
	llm, err := ollama.New(
		ollama.WithModel("qwen2.5:7b"),
		ollama.WithServerURL(ollamaServer))
	if err != nil {
		logger.Fatal("Nie udało się utworzyć modelu Ollama: %v", err)
	}
	logger.Debug("Model Ollama Qwen został pomyślnie utworzony.")
	return llm
}

// getStore Pobiera obiekt magazynu
func getStore() *qdrant.Store {
	logger.Debug("Inicjalizowanie magazynu Qdrant...")
	// Parsowanie URL
	qdUrl, err := url.Parse(qdrantUrl)
	if err != nil {
		logger.Fatal("Nie udało się sparsować URL: %v", err)
	}
	// Tworzenie nowego magazynu Qdrant
	store, err := qdrant.New(
		qdrant.WithURL(*qdUrl),                    // Ustawienie URL
		qdrant.WithAPIKey(""),                     // Ustawienie klucza API
		qdrant.WithCollectionName(collectionName), // Ustawienie nazwy kolekcji
		qdrant.WithEmbedder(getOllamaEmbedder()),  // Ustawienie embeddera
	)
	if err != nil {
		logger.Fatal("Nie udało się utworzyć magazynu Qdrant: %v", err)
	}
	logger.Debug("Magazyn Qdrant został pomyślnie utworzony.")
	return &store
}

// storeDocs Przechowuje dokumenty w bazie wektorowej
func storeDocs(docs []schema.Document, store *qdrant.Store) error {
	logger.Debug("Przechowywanie dokumentów w bazie wektorowej...")
	// Jeśli tablica dokumentów ma długość większą niż 0
	if len(docs) > 0 {
		// Dodawanie dokumentów do magazynu
		logger.Info("Dodawanie dokumentów do magazynu. Liczba dokumentów: %d", len(docs))
		_, err := store.AddDocuments(context.Background(), docs)
		if err != nil {
			logger.Error("Błąd podczas dodawania dokumentów do magazynu: %v", err)
			return err
		}
		logger.Debug("Dokumenty zostały pomyślnie dodane do magazynu.")
	} else {
		logger.Warning("Brak dokumentów do przechowania.")
	}
	return nil
}

// useRetriaver Funkcja używająca retrievera
func useRetriaver(store *qdrant.Store, prompt string, topk int) ([]schema.Document, error) {
	logger.Debug("Używanie retrievera do wyszukiwania dokumentów...")
	// Ustawienie opcji wektora
	optionsVector := []vectorstores.Option{
		vectorstores.WithScoreThreshold(0), // Ustawienie progu punktów
	}

	// Tworzenie retrievera
	retriever := vectorstores.ToRetriever(store, topk, optionsVector...)
	// Wykonanie wyszukiwania
	logger.Info("Wyszukiwanie dokumentów z progiem: %.2f i topk: %d", 0.80, topk)
	docRetrieved, err := retriever.GetRelevantDocuments(context.Background(), prompt)

	if err != nil {
		logger.Error("Nie udało się wyszukać dokumentów: %v", err)
		return nil, fmt.Errorf("Nie udało się wyszukać dokumentów: %v", err)
	}

	// Zwrócenie znalezionych dokumentów
	logger.Debug("Dokumenty zostały pomyślnie wyszukane.")
	return docRetrieved, nil
}

// GetAnswer Pobiera odpowiedź
func GetAnswer(ctx context.Context, llm llms.Model, docRetrieved []schema.Document, prompt string) (string, error) {
	logger.Debug("Pobieranie odpowiedzi na podstawie dokumentów...")
	// Tworzenie nowej historii wiadomości czatu
	history := memory.NewChatMessageHistory()
	// Dodanie pobranych dokumentów do historii
	for _, doc := range docRetrieved {
		logger.Debug("Dodawanie dokumentu do historii: %s", doc.PageContent)
		history.AddAIMessage(ctx, doc.PageContent)
	}
	// Tworzenie nowego bufora rozmowy na podstawie historii
	conversation := memory.NewConversationBuffer(memory.WithChatHistory(history))

	executor := agents.NewExecutor(
		agents.NewConversationalAgent(llm, nil),
		nil,
		agents.WithMemory(conversation),
	)
	// Ustawienie opcji dla wywołania łańcucha
	options := []chains.ChainCallOption{
		chains.WithTemperature(0.8),
	}

	// Uruchomienie łańcucha
	logger.Info("Uruchamianie łańcucha z parametrem temperatury: %.2f", 0.8)
	res, err := chains.Run(ctx, executor, prompt, options...)
	if err != nil {
		logger.Error("Błąd podczas uruchamiania łańcucha: %v", err)
		return "", err
	}

	logger.Debug("Odpowiedź została pomyślnie pobrana.")
	return res, nil
}
