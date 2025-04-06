import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import { errorHandler } from './middlewares/errorHandler';
import modelRoutes from './routes/modelRoutes';
import { RAGApplicationBuilder } from '@llm-tools/embedjs';
import { OllamaEmbeddings, Ollama } from '@llm-tools/embedjs-ollama';
import { HNSWDb } from '@llm-tools/embedjs-hnswlib';
import { PdfLoader } from '@llm-tools/embedjs-loader-pdf';
import { RedisStore } from '@llm-tools/embedjs-redis';

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

app.use(express.json());

async function initializeRAG() {
    try {
        const ragApplication = await new RAGApplicationBuilder()
            .setModel(new Ollama({ modelName: "llama3:latest", baseUrl: 'http://localhost:11434' }))
            .setEmbeddingModel(new OllamaEmbeddings({ model: 'mxbai-embed-large:latest', baseUrl: 'http://localhost:11434' }))
            .setVectorDatabase(new HNSWDb())
            // .setStore(new RedisStore({ host: 'localhost', port: 6379 })) // TODO configure Redis
            .setSystemMessage(`
                Jestes polakiem, asystentem AI. 
                Pamietaj zeby mowic po polsku. 
                Odpowiedz na pytania uzywajac dokumentow PDF jako zrodel informacji. 
                Jesli nie wiesz odpowiedzi, powiedz ze nie wiesz.
                `)
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

initializeRAG();

io.on('connection', (socket) => { // WebSocket connection
    console.log('New client connected');
    socket.on('query', async (question: string) => {
        try {
            socket.emit('response', 'Processing your query...');
            const result = await app.locals.ragApplication.query(question);
            socket.emit('response', result.content);
        } catch (error) {
            socket.emit('error', 'Failed to process query');
        }
    });

    socket.on('disconnect', () => {
        console.log('Client disconnected');
    });
});

app.use('/api/models', modelRoutes);
app.use(errorHandler);

httpServer.listen(3001, () => {
    console.log('Server is running on port 3001');
});

export default app;