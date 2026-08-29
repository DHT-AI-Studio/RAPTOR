from fastapi import Request, APIRouter

router = APIRouter()


@router.post("/finetune/{item_id}")
async def finetune(item_id: int, request: Request):
    return {
        "path": request.url.path,
        "method": request.method,
        "full_url": str(request.url)
    }


@router.post("/inference/{item_id}")
async def inference(item_id: int, request: Request):
    return {
        "path": request.url.path,
        "method": request.method,
        "full_url": str(request.url)
    }
