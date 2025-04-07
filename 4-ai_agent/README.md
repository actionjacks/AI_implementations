# FILE (pdf,csv,txt) Q&A Bot with Ollama & Qdrant

This project is a command-line tool that allows you to interact with a PDF document using natural language. It uses Ollama for generating text embeddings and chat responses, and Qdrant as a vector database to store and retrieve contextual information from the PDF.

✨ **Features**
- Extracts and splits text from a file.
- Generates embeddings using Ollama (mxbai-embed-large model).
- Stores embeddings in Qdrant for semantic search.
- Uses chat model (llama3 by default) to answer questions with or without document context.
- Supports conversation history.
- Terminal-based interactive chat with a loading spinner.

📦 **Requirements**
- Go 1.20+
- Running instance of Ollama
- Running instance of Qdrant
- PDF file (default: example.pdf in project root)

🔧 **Installation**
```bash
go mod tidy
Make sure ollama and qdrant are running locally:
```

Ollama: http://localhost:11434
Qdrant: localhost:6334

🚀 **Usage**

```bash
go run main.go
```
Optional flags:
- -keep-history – retain chat history in the conversation
- -use-context=false – disable using document context (defaults to true)

Example:

```bash
go run main.go -keep-history -use-context=false
```

🧠 **How It Works**

Text Extraction: Extracts all plain text from a specified file.
Chunking: Splits text into 500-character chunks.
Embeddings: Sends each chunk to Ollama’s mxbai-embed-large model to generate 1024-dimensional embeddings.
Storage: Embeddings are stored in a Qdrant collection named pdf_collection.
Interactive Q&A:
Input a question in the terminal.
If -use-context is enabled, the tool retrieves relevant chunks from Qdrant using vector search.
It sends your question along with relevant context to Ollama's llama3 model.
Returns an answer based on the PDF content.

📁 **Project Structure**

- cmd/main.go         // Main logic and CLI interaction
- example.pdf         // Default PDF file to analyze

🧪 **Example Use Case**

Imagine you have a long technical manual in PDF and want to ask questions like:

- `"What are the requirements for system installation?"
This app finds the most relevant sections and gives an informed answer via LLM.`

📝 **License**

 - MIT