package rag

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/textsplitter"
)

// TextToChunks Funkcja konwertuje plik tekstowy na fragmenty dokumentu
func TextToChunks(dirFile string, chunkSize, chunkOverlap int) ([]schema.Document, error) {
	file, err := os.Open(dirFile)
	if err != nil {
		return nil, err
	}
	// Tworzymy nowy loader dokumentów tekstowych
	docLoaded := documentloaders.NewText(file)
	// Tworzymy nowy rekurencyjny dzielnik tekstu
	split := textsplitter.NewRecursiveCharacter()
	// Ustawiamy rozmiar fragmentu
	split.ChunkSize = chunkSize
	// Ustawiamy nakładający się rozmiar fragmentu
	split.ChunkOverlap = chunkOverlap
	// Ładujemy i dzielimy dokument
	docs, err := docLoaded.LoadAndSplit(context.Background(), split)
	if err != nil {
		return nil, err
	}
	return docs, nil
}

// GetUserInput Funkcja pobiera dane wejściowe od użytkownika
func GetUserInput(promptString string) (string, error) {
	fmt.Print(promptString, ": ")
	var Input string
	reader := bufio.NewReader(os.Stdin)

	Input, _ = reader.ReadString('\n')

	Input = strings.TrimSuffix(Input, "\n")
	Input = strings.TrimSuffix(Input, "\r")

	return Input, nil
}
