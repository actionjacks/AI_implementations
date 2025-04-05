ollama run llama3:8b
ollama pull mxbai-embed-large


tworzenie kolecji do zapisywanie plikow w qadrant
```
curl -X PUT "http://localhost:6333/collections/documents" -H "Content-Type: application/json" -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  }
}'
``` ## wykasowac?

sprawdzenie 
```
curl -X GET "http://localhost:6333/collections/documents"
```

usuwanie kolekcji
```
curl -X DELETE "http://localhost:6333/collections/documents"
```
