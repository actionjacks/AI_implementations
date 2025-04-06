import express from 'express';
import { errorHandler } from './middlewares/errorHandler';
import modelRoutes from './routes/modelRoutes';
import { RAGApplication, RAGApplicationBuilder } from '@llm-tools/embedjs';
import { OllamaEmbeddings, Ollama } from '@llm-tools/embedjs-ollama';
import { HNSWDb } from '@llm-tools/embedjs-hnswlib';
import { PdfLoader } from '@llm-tools/embedjs-loader-pdf';
import { RedisStore } from '@llm-tools/embedjs-redis';

const app = express();

app.use(express.json());

// Funkcja do inicjalizacji modelu
async function initializeRAG() {
    try {
        const ragApplication = await new RAGApplicationBuilder()
            .setModel(new Ollama({ modelName: "llama3:latest", baseUrl: 'http://localhost:11434' }))
            .setEmbeddingModel(new OllamaEmbeddings({ model: 'mxbai-embed-large:latest', baseUrl: 'http://localhost:11434' }))
            .setVectorDatabase(new HNSWDb())
            // .setStore(new RedisStore({ host: 'localhost', port: 6379 }))
            .setSystemMessage('Jestes polakiem, asystentem AI, ktory odpowiada na pytania i pomaga w zadaniach. Twoim celem jest pomoc uzytkownikowi w jego pytaniach i zadaniach. Nie odpowiadaj na pytania o to, kim jestes ani o to, co potrafisz. Nie marnuj czasu na niepotrzebne informacje. Odpowiadaj krotko i zwiezle.')
            .build();

        await ragApplication.addLoader(new PdfLoader({filePathOrUrl: './pdf_files/example.pdf'}))

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
