from backend.services.model import IMG_SIZE, model_service
from backend.schemas import SegmentationResponse
from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


RESULT_DIR = Path("temp/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

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
    response_model=SegmentationResponse,
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
        result = model_service.predict(image)

        mask = result["mask"]
        overlay = result["overlay"]

            
    except Exception:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INFERENCE_FAILED",
                "message": "모델 추론 중 오류가 발생했습니다.",
            },
        )
        
    result_id = str(uuid4())
    result_path = RESULT_DIR / f"{result_id}.png"
    success = cv2.imwrite(
        str(result_path),
        overlay,
    )
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "MASK_SAVE_FAILED",
                "message": "결과 이미지를 저장하는 중 오류가 발생했습니다.",
            },
        )
        
    # 9. 응답
    return SegmentationResponse(
        result_id=result_id,
        filename=file.filename,
        width=image.shape[1],
        height=image.shape[0],
        mask_url=f"/api/results/{result_id}/mask",
        message="이미지 세그멘테이션이 완료되었습니다.",
    )
   
    
@router.get("/api/results/{result_id}/image")
async def get_image(result_id: str):
    result_path = RESULT_DIR / f"{result_id}.png"

    if not result_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "RESULT_NOT_FOUND",
                "message": "요청한 마스크 결과를 찾을 수 없습니다.",
            },
        )

    return FileResponse(
        path=result_path,
        media_type="image/png",
    )
    
    
