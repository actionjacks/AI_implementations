package parser

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/ledongthuc/pdf"
)

func ParseFile(filePath string) (string, error) {
	if strings.HasSuffix(filePath, ".pdf") {
		return parsePDF(filePath)
	} else if strings.HasSuffix(filePath, ".csv") {
		return parseCSV(filePath)
	} else if strings.HasSuffix(filePath, ".txt") {
		return parseTXT(filePath)
	}
	return "", fmt.Errorf("unsupported file type")
}

func parseTXT(path string) (string, error) {
	content, err := os.ReadFile(path)
	return string(content), err
}

func parsePDF(path string) (string, error) {
	f, r, err := pdf.Open(path)
	defer f.Close()
	if err != nil {
		return "", err
	}
	var buf strings.Builder
	b, err := r.GetPlainText()
	if err != nil {
		return "", err
	}
	_, err = io.Copy(&buf, b)
	return buf.String(), err
}

func parseCSV(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
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
			return "", err
		}
		result.WriteString(strings.Join(record, " ") + "\n")
	}
	return result.String(), nil
}
