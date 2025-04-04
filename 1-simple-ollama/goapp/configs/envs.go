package configs

import (
	"log"
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	Port         string
	OllamaModel  string
	OllamaAPIURL string
}

var Envs = initConfig()

func initConfig() Config {
	godotenv.Load()

	if err := godotenv.Load(); err != nil {
		log.Println("Warning: No .env file found or could not be loaded")
	}

	return Config{
		Port:         getEnv("PORT", "8080"),
		OllamaModel:  getEnv("OLLAMA_MODEL", "llama3:8b"),
		OllamaAPIURL: getEnv("OLLAMA_API_URL", "http://ollamadeepseek:11436"),
	}
}

func getEnv(key string, defaultValue string) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	return defaultValue
}
