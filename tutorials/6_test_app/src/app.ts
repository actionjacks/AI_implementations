import express from 'express';
import { errorHandler } from './middlewares/errorHandler';
import modelRoutes from './routes/modelRoutes';

const app = express();

app.use(express.json());

app.use('/api/models', modelRoutes);
app.use(errorHandler);

export default app;