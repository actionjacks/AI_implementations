import { Router } from 'express';
import { Request, Response, NextFunction } from 'express';
import { errorHandler } from '../middlewares/errorHandler';

const router = Router();

// http://localhost:3000/api/models?q=fo
router.get('/', async(req: Request, res: Response, next: NextFunction) => {
    const question = req.query.q
    if (!question) {
        errorHandler(
            {
                name: 'Bad Request.',
                message: 'Query parameter "q" is required.',
                status: 400
            }
            , req, res, next);
            return;
    }
    const result = await req.app.locals.ragApplication.query(question);
    try {
        res.status(200).json({
            message: 'Success',
            data: result,
            error: null,
        });
    } catch (err) {
        next(err);
    }
});

export default router;
