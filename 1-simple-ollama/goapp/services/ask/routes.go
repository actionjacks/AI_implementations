package ask

import (
	"api_ollama/configs"
	"api_ollama/utils"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gorilla/mux"
)

type OllamaRequest struct {
	Prompt string `json:"prompt"`
	Model  string `json:"model"`
	Stream bool   `json:"stream"`
}

type ErrorResponse struct {
	Error   string `json:"error"`
	Details string `json:"details,omitempty"`
}

func RegisterRoutes(router *mux.Router) {
	router.HandleFunc("/ask", handleAsk).Methods(http.MethodGet)
}

func handleAsk(w http.ResponseWriter, r *http.Request) {
	prompt, err := validatePrompt(r)
	if err != nil {
		sendError(w, http.StatusBadRequest, err.Error())
		return
	}

	response, err := callOllamaAPI(prompt)
	if err != nil {
		sendError(w, http.StatusInternalServerError, err.Error())
		return
	}

	utils.WriteJSON(w, http.StatusOK, response)
}

func validatePrompt(r *http.Request) (string, error) {
	prompt := strings.Trim(r.URL.Query().Get("prompt"), "\"")
	if strings.TrimSpace(prompt) == "" {
		return "", fmt.Errorf("prompt is required")
	}
	return prompt, nil
}

func callOllamaAPI(prompt string) (map[string]interface{}, error) {
	requestData := OllamaRequest{
		Prompt: prompt,
		Model:  configs.Envs.OllamaModel,
		Stream: false,
	}

	requestBody, err := json.Marshal(requestData)
	if err != nil {
		return nil, fmt.Errorf("failed to create request body")
	}

	apiURL := fmt.Sprintf("%s/api/generate", configs.Envs.OllamaAPIURL)

	response, err := http.Post(apiURL, "application/json", bytes.NewBuffer(requestBody))
	if err != nil {
		return nil, fmt.Errorf("failed to call Ollama API")
	}
	defer response.Body.Close()

	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read API response")
	}

	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API returned status %d: %s", response.StatusCode, string(responseBody))
	}

	var ollamaResponse map[string]interface{}
	if err := json.Unmarshal(responseBody, &ollamaResponse); err != nil {
		return nil, fmt.Errorf("failed to parse API response")
	}

	return ollamaResponse, nil
}

func sendError(w http.ResponseWriter, statusCode int, message string) {
	utils.WriteJSON(w, statusCode, ErrorResponse{
		Error: message,
	})
}
