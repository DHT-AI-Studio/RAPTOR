from fastapi import Request, APIRouter

router = APIRouter()

@router.post("/finetune/{item_id}")
async def finetune(item_id: int, request: Request):
    return {
        "path": request.url.path,      # "/items/123"
        "method": request.method,      # "GET"
        "full_url": str(request.url)   # "http://127.0.0.1:8000/items/123"
    }

@router.post("/inference/{item_id}")
async def inference(item_id: int, request: Request):
    return {
        "path": request.url.path,      # "/items/123"
        "method": request.method,      # "GET"
        "full_url": str(request.url)   # "http://127.0.0.1:8000/items/123"
    }