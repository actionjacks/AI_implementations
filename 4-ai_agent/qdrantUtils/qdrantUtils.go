package qdrantutils

import (
	ollamautils "ai_agent/ollamaUtils"
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/qdrant/go-client/qdrant"
)

const (
	collectionName = "pdf_collection" // Qdrant Collection Name
	vectorSize     = 1024             // Embedding vector size
	qdrantHost     = "localhost"      // Adres hosta Qdrant
	qdrantPort     = 6334             // Port Qdrant
)

// GetContextFromQdrant queries the Qdrant vector database using the provided embedding,
// retrieves the top matching results, and extracts their textual payloads to build a context string.
// This context is used to enhance the relevance of LLM-based answers.
//
// Parameters:
//   - qdrantClient: an instance of the Qdrant client used for querying the database
//   - collectionName: the name of the Qdrant collection to search
//   - questionEmbedding: the embedding vector representing the user's question
//
// Returns:
//   - A string representing the combined context extracted from the matched documents
//   - An error if the query fails or payloads cannot be parsed
func GetContextFromQdrant(
	qdrantClient *qdrant.Client,
	collectionName string,
	questionEmbedding []float32) (string, error) {
	limit := uint64(5)                                                                 // Set a search result limit.
	searchResult, err := qdrantClient.Query(context.Background(), &qdrant.QueryPoints{ // Executing a query to Qdrant.
		CollectionName: collectionName,
		Query:          qdrant.NewQuery(questionEmbedding...), // Passing the question embedding vector.
		WithPayload:    qdrant.NewWithPayload(true),           // Downloading payload (text)
		Limit:          &limit,                                // Set a result limit.
	})
	if err != nil {
		return "", fmt.Errorf("qdrant search error: %v", err)
	}

	var contextBuilder strings.Builder // Utworzenie bufora do przechowywania kontekstu
	for _, result := range searchResult {
		if payload := result.GetPayload(); payload != nil {
			if textValue, ok := payload["text"]; ok {
				if text := textValue.GetStringValue(); text != "" {
					contextBuilder.WriteString(text + "\n\n")
				} else {
					log.Printf("Expected string in payload 'text', found other type.")
				}
			}
		}
	}
	return contextBuilder.String(), nil
}

// GetAnswerFromDocument generates an embedding for the given question, retrieves relevant context
// from Qdrant, and sends both the question and context to the Ollama LLM for an answer.
// Optionally, conversation history can be incorporated into the context.
//
// Parameters:
//   - ollamaClient: an instance of the Ollama API client
//   - qdrantClient: a Qdrant vector database client
//   - chatModel: the name of the LLM model to use for answering
//   - embeddingModel: the model used to generate vector embeddings
//   - collectionName: the name of the Qdrant collection
//   - question: the user question to answer
//   - conversationHistory: previous chat history (optional)
//
// Returns:
//   - A string containing the generated answer from the LLM
//   - An error if embedding generation, context retrieval, or the LLM call fails
func GetAnswerFromDocument(
	ollamaClient *api.Client,
	qdrantClient *qdrant.Client,
	chatModel,
	embeddingModel,
	collectionName,
	question string,
	conversationHistory string) (string, error) {
	questionEmbedding, err := ollamautils.GenerateEmbedding(ollamaClient, embeddingModel, question)
	if err != nil {
		return "", fmt.Errorf("question embedding generation error: %v", err)
	}

	// Pobranie kontekstu z Qdrant
	context, err := GetContextFromQdrant(qdrantClient, collectionName, questionEmbedding)
	if err != nil {
		return "", err
	}

	// Zadanie pytania Ollamie z kontekstem i historią
	answer, err := ollamautils.AskOllamaWithContext(ollamaClient, chatModel, question, context)
	if err != nil {
		return "", err
	}
	return answer, nil
}
