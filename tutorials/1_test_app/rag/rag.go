package rag

import (
	"context"
	"fmt"

	"github.com/spf13/cobra"

	"github.com/hantmac/langchaingo-ollama-rag/rag/logger"
)

// rootCmd reprezentuje główną komendę, która jest wywoływana bez żadnych podkomend
var rootCmd = &cobra.Command{
	Use:   "langchaingo-ollama",
	Short: "System pytań i odpowiedzi oparty na langchaingo",
	Long:  `System pytań i odpowiedzi oparty na langchaingo`,
}

func Execute() {
	cobra.CheckErr(rootCmd.Execute())
}

func init() {
	rootCmd.Flags().BoolP("toggle", "t", false, "Pomocnicza wiadomość dla przełącznika")
	// ========
	rootCmd.AddCommand(GetAnwserCmd)
	GetAnwserCmd.Flags().IntP("topk", "t", 5, "Liczba zwróconych dokumentów, domyślnie 5")
}

// FileToChunksCmd Funkcja przetwarzająca plik na fragmenty
func FileToChunksCmd() {
	filepath := "test.txt"
	chunkSize := 50
	chunkOverlap := 10

	docs, err := TextToChunks(filepath, chunkSize, chunkOverlap)
	if err != nil {
		logger.Error("Nie udało się przekonwertować pliku na fragmenty, błąd: %v", err)
	}
	logger.Info("Plik został pomyślnie przekonwertowany na fragmenty, liczba fragmentów: ", len(docs))
	for _, v := range docs {
		fmt.Printf("🗂 Zawartość fragmentu==> %v\n", v.PageContent)
	}
}

// EmbeddingCmd Funkcja przetwarzająca plik na fragmenty i zapisująca wektory
func EmbeddingCmd() {
	filepath := "test.txt"
	chunkSize := 5
	chunkOverlap := 2
	docs, err := TextToChunks(filepath, chunkSize, chunkOverlap)
	if err != nil {
		logger.Error("Nie udało się przekonwertować pliku na fragmenty, błąd: %v", err)
	}
	err = storeDocs(docs, getStore())
	if err != nil {
		logger.Error("Nie udało się przekonwertować fragmentów na wektory, błąd: %v", err)
	} else {
		logger.Info("Fragmenty zostały pomyślnie przekonwertowane na wektory")
	}
}

// GetAnwserCmd Komenda do uzyskiwania odpowiedzi
var GetAnwserCmd = &cobra.Command{
	Use:   "getanswer",
	Short: "Pobierz odpowiedź",
	Run: func(cmd *cobra.Command, args []string) {
		topk, _ := cmd.Flags().GetInt("topk")
		FileToChunksCmd()
		EmbeddingCmd()

		prompt, err := GetUserInput("Wprowadź swoje pytanie")
		if err != nil {
			logger.Error("Nie udało się pobrać danych wejściowych od użytkownika, błąd: %v", err)
		}
		rst, err := useRetriaver(getStore(), prompt, topk)
		if err != nil {
			logger.Error("Nie udało się pobrać dokumentów, błąd: %v", err)
		}
		answer, err := GetAnswer(context.Background(), getOllamaQwen(), rst, prompt)
		if err != nil {
			logger.Error("Nie udało się uzyskać odpowiedzi, błąd: %v", err)
		} else {
			fmt.Printf("🗂 Odpowiedź==> %s\n\n", answer)
		}
	},
}
