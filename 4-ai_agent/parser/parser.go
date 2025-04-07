package parser

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/ledongthuc/pdf"
	"github.com/pterm/pterm"
)

// ParseFile determines the file type based on its extension and parses it accordingly.
// It supports PDF, CSV, and TXT files. If the file type is not supported,
// it returns an error.
func ParseFile(filePath string) (string, error) {
	if strings.HasSuffix(filePath, ".pdf") {
		return parsePDF(filePath)
	} else if strings.HasSuffix(filePath, ".csv") {
		return parseCSV(filePath)
	} else if strings.HasSuffix(filePath, ".txt") {
		return parseTXT(filePath)
	}
	pterm.Error.Printf("Unsupported file type: %s\n", filePath)
	return "", fmt.Errorf("unsupported file type: %s", filePath)
}

func parseTXT(path string) (string, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		pterm.Error.Println("Failed to read TXT file:", err)
		return "", err
	}
	return string(content), nil
}

func parsePDF(path string) (string, error) {
	f, r, err := pdf.Open(path)
	if err != nil {
		pterm.Error.Println("Failed to open PDF:", err)
		return "", err
	}
	defer f.Close()

	var buf strings.Builder
	b, err := r.GetPlainText()
	if err != nil {
		pterm.Error.Println("Failed to extract text from PDF:", err)
		return "", err
	}

	_, err = io.Copy(&buf, b)
	if err != nil {
		pterm.Error.Println("Failed to copy PDF text to buffer:", err)
		return "", err
	}
	return buf.String(), nil
}

func parseCSV(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		pterm.Error.Println("Failed to open CSV file:", err)
		return "", err
	}
	defer file.Close()

	r := csv.NewReader(file)
	var result strings.Builder
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			pterm.Error.Println("Failed to read CSV record:", err)
			return "", err
		}
		result.WriteString(strings.Join(record, " ") + "\n")
	}
	return result.String(), nil
}

// Splits the given text into chunks of the given size.
// Returns a slice of strings, where each element is a chunk of the original text.
func SplitTextIntoChunks(text string, chunkSize int) []string {
	var chunks []string
	for i := 0; i < len(text); i += chunkSize {
		end := i + chunkSize
		if end > len(text) {
			end = len(text)
		}
		chunks = append(chunks, text[i:end])
	}
	return chunks
}
