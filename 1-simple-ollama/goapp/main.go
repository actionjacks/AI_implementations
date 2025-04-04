package main

import (
	"api_ollama/api"
	"api_ollama/configs"
	"fmt"
	"log"
)

// Define the request body structure
type OllamaRequest struct {
	Prompt string `json:"prompt"`
	Model  string `json:"model"`
	Stream bool   `json:"stream"`
}

func main() {
	server := api.NewAPIServer(fmt.Sprintf(":%s", configs.Envs.Port))
	if err := server.Start(); err != nil {
		log.Fatal(err)
	}
}
