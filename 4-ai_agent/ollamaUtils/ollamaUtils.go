package ollamautils

import (
	"context"
	"fmt"

	"github.com/ollama/ollama/api"
)

// GenerateEmbedding sends a single text input to the Ollama API and returns its embedding vector.
// The embedding is converted from float64 (as returned by the API) to float32.
//
// Parameters:
//   - client: the Ollama API client
//   - model: the name of the embedding model to use
//   - text: the input text to embed
//
// Returns:
//   - A slice of float32 representing the embedding vector
//   - An error if the embedding generation fails
func GenerateEmbedding(client *api.Client, model, text string) ([]float32, error) {
	resp, err := client.Embeddings(context.Background(), &api.EmbeddingRequest{
		Model:  model,
		Prompt: text,
	})
	if err != nil {
		return nil, fmt.Errorf("embedding generation error (%s): %v", model, err)
	}

	embedding32 := make([]float32, len(resp.Embedding))
	for i, v := range resp.Embedding {
		embedding32[i] = float32(v)
	}
	return embedding32, nil
}

// GenerateEmbeddings processes a list of texts and returns their corresponding embedding vectors
// by calling GenerateEmbedding for each one individually.
//
// Parameters:
//   - client: the Ollama API client
//   - model: the name of the embedding model to use
//   - texts: a slice of input texts to embed
//
// Returns:
//   - A 2D slice of float32 where each inner slice is an embedding vector for a text
//   - An error if embedding generation fails for any text
func GenerateEmbeddings(client *api.Client, model string, texts []string) ([][]float32, error) {
	embeddings := make([][]float32, 0, len(texts))
	for _, text := range texts {
		embedding, err := GenerateEmbedding(client, model, text)
		if err != nil {
			return nil, fmt.Errorf("error generating embedding for text '%s': %v", text, err)
		}
		embeddings = append(embeddings, embedding)
	}
	return embeddings, nil
}

// AskOllamaWithContext sends a question to the Ollama API along with contextual information,
// typically extracted from documents, to improve the quality of the generated answer.
//
// Parameters:
//   - ollamaClient: the Ollama API client
//   - chatModel: the name of the language model to use
//   - question: the user question to ask
//   - contextStr: supporting context (e.g., from document embeddings)
//
// Returns:
//   - A string with the model's generated answer
//   - An error if the generation request fails
func AskOllamaWithContext(ollamaClient *api.Client, chatModel, question, contextStr string) (string, error) {
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
			"temperature": 0.3, // Generation temperature setting.
		},
	}, func(genResp api.GenerateResponse) error {
		response += genResp.Response
		return nil
	})
	if err != nil {
		return "", fmt.Errorf("ollama response generation error: %v", err)
	}
	return response, nil
}

// AskOllama sends a standalone question (optionally including chat history) to the Ollama API
// and returns the generated response from the model.
//
// Parameters:
//   - ollamaClient: the Ollama API client
//   - chatModel: the name of the language model to use
//   - question: the user question to ask
//   - conversationHistory: optional chat history to maintain context across turns
//
// Returns:
//   - A string with the model's generated answer
//   - An error if the generation request fails
func AskOllama(ollamaClient *api.Client, chatModel, question string, conversationHistory string) (string, error) {
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
		return "", fmt.Errorf("ollama response generation error: %v", err)
	}
	return response, nil
}
