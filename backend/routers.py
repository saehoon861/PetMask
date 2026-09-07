from fastapi import APIRouter
from backend.services.model import IMG_SIZE, model_service

router = APIRouter()

@router.get("/api/health")
async def health():
    return {"status": "ok"}


@router.get("/api/model")
async def get_model_info():
    return {
        "name": "UNet-ResNet18",
        "input_size": IMG_SIZE,
        "threshold": model_service.threshold,
        "device": str(model_service.device),
    }