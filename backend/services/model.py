from pathlib import Path
import os

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from src.models.model import ResNetUNet


IMG_SIZE = 512

DEFAULT_THRESHOLD = float(
    os.getenv("MODEL_THRESHOLD", "0.85")
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHECKPOINT_PATH = Path(
    os.getenv(
        "MODEL_CHECKPOINT",
        PROJECT_ROOT / "checkpoints" / "checkpoint.pth"
    )
)


class PetMaskModelService:

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.checkpoint_path = Path(checkpoint_path)
        self.threshold = threshold

        self.transform = A.Compose([
            A.LongestMaxSize(max_size=IMG_SIZE),

            A.PadIfNeeded(
                min_height=IMG_SIZE,
                min_width=IMG_SIZE,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
            ),

            A.Normalize(),

            ToTensorV2(),
        ])

        self.model = self._load_model()

    def _load_model(self):

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        model = ResNetUNet(n_class=1)


#torch.load()는 파일을 읽는 것, model.load_state_dict()는 읽어온 가중치를 모델에 넣는 것
        state_dict = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()

        return model

    def preprocess(
        self,
        image: np.ndarray
    ) -> tuple[torch.Tensor, dict]:

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "image must be numpy.ndarray"
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "image must have shape (H, W, 3)"
            )

        original_h, original_w = image.shape[:2]

        scale = IMG_SIZE / max(
            original_h,
            original_w,
        )

        resized_h = round(original_h * scale)
        resized_w = round(original_w * scale)

        transformed = self.transform(
            image=image
        )

        tensor = transformed["image"].float()
        tensor = tensor.unsqueeze(0)

        meta = {
            "original_h": original_h,
            "original_w": original_w,
            "resized_h": resized_h,
            "resized_w": resized_w,
        }

        return tensor.to(self.device), meta
    

    def postprocess(
        self,
        probability: np.ndarray,
        mask: np.ndarray,
        meta: dict,
    ) -> tuple[np.ndarray, np.ndarray]:

        original_h = meta["original_h"]
        original_w = meta["original_w"]

        resized_h = meta["resized_h"]
        resized_w = meta["resized_w"]

        # PadIfNeeded는 기본적으로 중앙 기준 padding
        pad_top = (IMG_SIZE - resized_h) // 2
        pad_left = (IMG_SIZE - resized_w) // 2

        # padding 제거
        probability = probability[
            pad_top:pad_top + resized_h,
            pad_left:pad_left + resized_w
        ]

        mask = mask[
            pad_top:pad_top + resized_h,
            pad_left:pad_left + resized_w
        ]

        # 원본 이미지 크기로 복원
        probability = cv2.resize(
            probability,
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = cv2.resize(
            mask,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST,
        )
        
        return probability, mask

    def create_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        overlay = image.copy()

        mask_bool = mask > 0

        overlay[mask_bool] = (0, 255, 0)

        result = cv2.addWeighted(
            image,
            1 - alpha,
            overlay,
            alpha,
            0,
        )

        return result

    @torch.inference_mode()
    def predict(
        self,
        image: np.ndarray
    ):

        input_tensor, meta = self.preprocess(image)
        outputs = self.model(input_tensor)
        logits = outputs[:, 0]
        probability = torch.sigmoid(logits)
        binary_mask = (
            probability >= self.threshold
        )

        probability = (
            probability
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        mask = (
            binary_mask
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.uint8)
            * 255
        )

        probability, mask = self.postprocess(
            probability,
            mask,
            meta,
        )
        
        overlay = self.create_overlay(image, mask)
        

        return {
            "probability": probability,
            "mask": mask,
            "threshold": self.threshold,
            "overlay": overlay
        }


model_service = PetMaskModelService()