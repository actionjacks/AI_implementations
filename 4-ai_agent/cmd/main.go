package main

import (
	ollamautils "ai_agent/ollamaUtils"
	"ai_agent/parser"
	qdrantutils "ai_agent/qdrantUtils"
	"bufio"
	"context"
	"flag"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/pterm/pterm"
	"github.com/qdrant/go-client/qdrant"
)

const (
	collectionName = "pdf_collection"
	vectorSize     = 1024 // mxbai-embed-large returns 1024-dimensional embeddings.
	ollamaURLStr   = "http://localhost:11434"
	qdrantHost     = "localhost"
	qdrantPort     = 6334
	embeddingModel = "mxbai-embed-large" // Model name Ollama for embeddings.
	chatModel      = "llama3"            // Ollama chat model name.
	pdfFilePath    = "../example.pdf"    // Path to file.
	textChunkSize  = 500
)

func main() {
	keepHistory := flag.Bool("keep-history", false, "Should the model remember the conversation history?")
	useContext := flag.Bool("use-context", true, "Should the model use the context from the document?")
	flag.Parse()

	if _, err := os.Stat(pdfFilePath); os.IsNotExist(err) {
		pterm.Info.Printf("File does not exist: %v\n", err)
	}

	text, err := parser.ParseFile(pdfFilePath)
	if err != nil {
		pterm.Fatal.Printf("Extraction error: %v\n", err)
	}

	chunks := parser.SplitTextIntoChunks(text, textChunkSize)

	ollamaURL, err := url.Parse(ollamaURLStr)
	if err != nil {
		pterm.Fatal.Printf("Ollama URL parsing error: %v\n", err)
	}

	ollamaClient := api.NewClient(ollamaURL, http.DefaultClient)
	embeddings, err := ollamautils.GenerateEmbeddings(ollamaClient, embeddingModel, chunks)
	if err != nil {
		pterm.Fatal.Printf("Embedding generation error: %v\n", err)
	}

	qdrantClient, err := qdrant.NewClient(&qdrant.Config{ // Sign up for Qdrant
		Host: qdrantHost,
		Port: qdrantPort,
	})
	if err != nil {
		pterm.Fatal.Printf("Cannot connect to Qdrant: %v\n", err)
	}
	defer qdrantClient.Close()

	ctx := context.Background()

	// Create collection (ignore error if already exists)
	err = qdrantClient.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: collectionName,
		VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
			Size:     vectorSize,
			Distance: qdrant.Distance_Cosine,
		}),
	})
	if err != nil && !strings.Contains(err.Error(), "already exists") {
		pterm.Fatal.Printf("Error creating collection: %v\n", err)
	} else if err != nil {
		pterm.Info.Printf("The collection already exists: %v\n", err)
	}

	points := make([]*qdrant.PointStruct, len(chunks)) // Prepare points for recording.
	for i, chunk := range chunks {
		pointID := qdrant.NewIDNum(uint64(i + 1))
		vectors := qdrant.NewVectors(embeddings[i]...)
		payload := map[string]any{"text": chunk}
		points[i] = &qdrant.PointStruct{
			Id:      pointID,
			Vectors: vectors,
			Payload: qdrant.NewValueMap(payload),
		}
	}

	operationInfo, err := qdrantClient.Upsert(ctx, &qdrant.UpsertPoints{ // Saving points to Qdrant.
		CollectionName: collectionName,
		Points:         points,
	})
	if err != nil {
		pterm.Fatal.Printf("Upsert error: %v\n", err)
	}
	pterm.Info.Printf("Upsert operation info: %+v\n", operationInfo)

	// Pętla interaktywna
	reader := bufio.NewReader(os.Stdin)
	conversationHistory := ""

	pterm.Success.Printf("Start a conversation with a model _ %s\n", chatModel)
	pterm.Success.Printf("/q exit \n")
	for {
		question, _ := reader.ReadString('\n')
		question = strings.TrimSpace(question)

		if strings.ToLower(question) == "/q" {
			break
		}

		if question == "" {
			pterm.Error.Printf("Question cannot be empty. Please try again.")
			continue
		}

		if *keepHistory { // Add a question to the story.
			conversationHistory += fmt.Sprintf("Question: %s\n", question)
		}

		done := make(chan bool)
		go showSpinner(done)

		var answer string
		var err error

		if *useContext { // Use context from Qdrant if flag is set.
			answer, err = qdrantutils.GetAnswerFromDocument(
				ollamaClient,
				qdrantClient,
				chatModel,
				embeddingModel,
				collectionName,
				question,
				conversationHistory)
		} else {
			answer, err = ollamautils.AskOllama(
				ollamaClient,
				chatModel,
				question,
				conversationHistory)
		}

		close(done)
		pterm.Println()

		if err != nil {
			pterm.Error.Printf("There was a problem getting a response: %v\n", err)
			continue
		}

		if *keepHistory { // Add a reply to the story.
			conversationHistory += fmt.Sprintf("Answer: %s\n", answer)
		}

		pterm.Success.Printf("Question: %s\n", question)
		pterm.Description.Printf("Answer: %s\n", answer)
		pterm.Info.Printf("Response time: %s\n", time.Since(time.Now()).String())

	}
}

func showSpinner(done chan bool) {
	spinner, _ := pterm.DefaultSpinner.
		WithRemoveWhenDone(true).
		Start("Waiting for response...")

	for {
		select {
		case <-done:
			spinner.Stop()
			return
		default:
			time.Sleep(100 * time.Millisecond)
		}
	}
}
