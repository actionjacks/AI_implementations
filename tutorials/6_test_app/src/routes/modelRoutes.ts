import { Router } from 'express';
import { Request, Response, NextFunction } from 'express';

const router = Router();

router.get('/', (req: Request, res: Response, next: NextFunction)=> {
    res.status(200).json({
        message: 'Model route is working',
        data: null,
        error: null,
    });
});

export default router;