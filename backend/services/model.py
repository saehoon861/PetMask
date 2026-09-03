from pathlib import Path
import os

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from src.models.model import ResNetUNet


IMG_SIZE = 256

DEFAULT_THRESHOLD = float(
    os.getenv("MODEL_THRESHOLD", "0.87")
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

        model = ResNetUNet(num_class=1)

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
    ) -> torch.Tensor:

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "image must be numpy.ndarray"
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "image must have shape (H, W, 3)"
            )

        transformed = self.transform(
            image=image
        )

        tensor = transformed["image"].float()

        # [C, H, W]
        #      ↓
        # [1, C, H, W]
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    @torch.inference_mode()
    def predict(
        self,
        image: np.ndarray
    ):

        input_tensor = self.preprocess(image)

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

        return {
            "probability": probability,
            "mask": mask,
            "threshold": self.threshold,
        }


model_service = PetMaskModelService()

if __name__ == "__main__":
    print("device:", model_service.device)
    print("checkpoint:", model_service.checkpoint_path)
    print("threshold:", model_service.threshold)
    print("model loaded successfully")