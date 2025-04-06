import { Router } from 'express';
import { Request, Response, NextFunction } from 'express';

const router = Router();

router.get('/', async(req: Request, res: Response, next: NextFunction) => {
    console.log(req.app.locals.ragApplication, '<---');

    const result = await req.app.locals.ragApplication.query('co to jest bufor?');

    console.log(result, '<--- result');

    try {
        res.status(200).json({
            message: 'Model route is working',
            data: result.content,
            error: null,
        });
    } catch (err) {
        next(err);
    }
});

router.get('/upload', async(req: Request, res: Response, next: NextFunction) => {
    console.log('=====================');

    res.status(200).json({
        message: 'upload',
        data: 'upload',
        error: null,
    })
});

export default router;
