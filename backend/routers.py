from backend.services.model import IMG_SIZE, model_service
from backend.schemas import ImageUploadResponse
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


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
    

@router.post(
    "/api/segment",
    response_model=ImageUploadResponse,
)
async def segment(file: UploadFile = File(...)):
    # 1. 파일 이름 확인
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "FILE_NOT_PROVIDED",
                "message": "이미지 파일이 제공되지 않았습니다.",
            },
        )

    # 2. 확장자 확인
    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail={
                "code": "INVALID_FILE_TYPE",
                "message": "지원하지 않는 이미지 형식입니다.",
            },
        )

    # 3. 업로드 파일 읽기
    contents = await file.read()

    # 4. 파일 크기 확인
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": "이미지 파일 크기가 제한을 초과했습니다.",
            },
        )

    # 5. bytes -> NumPy 배열
    image_array = np.frombuffer(
        contents,
        dtype=np.uint8,
    )

    # 6. NumPy 배열 -> OpenCV 이미지
    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    # 7. 이미지 decode 실패 확인
    if image is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "IMAGE_DECODE_FAILED",
                "message": "이미지를 정상적으로 읽을 수 없습니다.",
            },
        )

   # 8. 모델 추론
    try:
        probability, mask = model_service.predict(image)

    except Exception:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INFERENCE_FAILED",
                "message": "모델 추론 중 오류가 발생했습니다.",
            },
        )

    # 9. 응답
    return ImageUploadResponse(
        filename=file.filename,
        width=image.shape[1],
        height=image.shape[0],
        message="이미지 세그멘테이션이 완료되었습니다.",
    )