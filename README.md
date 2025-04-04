# 2-rag

Based on [rag-web-ui](https://github.com/rag-web-ui/rag-web-ui), with modifications to use the **Ollama** language model as the default backend implementation.

## Getting Started

After launching the containers, you need to **manually download the required Ollama models** by running the following commands:

```bash
docker exec -it 2-rag-ollama-1 sh
ollama pull deepseek-r1:7b
ollama pull omic-embed-text
