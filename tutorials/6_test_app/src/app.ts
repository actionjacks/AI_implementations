import express from 'express';
import { errorHandler } from './middlewares/errorHandler';
import modelRoutes from './routes/modelRoutes';
import { RAGApplication, RAGApplicationBuilder } from '@llm-tools/embedjs';
import { OllamaEmbeddings, Ollama } from '@llm-tools/embedjs-ollama';
import { HNSWDb } from '@llm-tools/embedjs-hnswlib';

// let ragApplication: RAGApplication;

const app = express();

app.use(express.json());

// Funkcja do inicjalizacji modelu
async function initializeRAG() {
    try {
        const ragApplication = await new RAGApplicationBuilder()
            .setModel(new Ollama({ modelName: "llama3:latest", baseUrl: 'http://localhost:11434' }))
            .setEmbeddingModel(new OllamaEmbeddings({ model: 'mxbai-embed-large:latest', baseUrl: 'http://localhost:11434' }))
            .setVectorDatabase(new HNSWDb())
            .build();

        // Zapisanie modelu do app.locals, aby był dostępny globalnie
        // ragApplication = app;
        app.locals = {
            ragApplication
        };
        console.log('RAG Application initialized successfully');
    } catch (err) {
        console.error('Failed to initialize RAG application:', err);
    }
}

// Inicjalizowanie modelu przed startem serwera
initializeRAG();

// Trasy
app.use('/api/models', modelRoutes);

// Middleware do obsługi błędów
app.use(errorHandler);

export default app;
