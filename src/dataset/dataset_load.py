import torch
from torch import nn
import os
from os import path
import torchvision
import torchvision.transforms as T
from typing import Sequence
from torchvision.transforms import functional as F
import numbers
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torchmetrics as TM

# Convert a pytorch tensor into a PIL image
t2img = T.ToPILImage()
# Convert a PIL image into a pytorch tensor
img2t = T.ToTensor()

# Set the working (writable) directory.
working_dir = "/home/sehoon/workspace/PetMask/src/dataset"

pets_path_train = os.path.join(working_dir, 'OxfordPets', 'train')
pets_path_test = os.path.join(working_dir, 'OxfordPets', 'test')
pets_train_orig = torchvision.datasets.OxfordIIITPet(root=pets_path_train, split="trainval", target_types="segmentation", download=True)
pets_test_orig = torchvision.datasets.OxfordIIITPet(root=pets_path_test, split="test", target_types="segmentation", download=True)

print(f"Number of training samples: {len(pets_train_orig)}")
print(f"Number of test samples: {len(pets_test_orig)}")
(train_pets_input, train_pets_target) = pets_train_orig[0]

# plt.imshow(train_pets_input)
# plt.axis("off")
# plt.show()

# plt.imshow(train_pets_target)
# plt.axis("off")
# plt.show()

# enum: 선택 가능한 값들을 이름으로 묶어놓은 것 
from enum import IntEnum
class TrimapClass(IntEnum):
    PET = 0
    BACKGROUND = 1
    BORDER = 2
    
#BCEWithLogitsLoss는 시그모이드 활성화 함수와 이진 교차 엔트로피 손실을 결합한 손실 함수 이 함수는 정답이 float형을 필요로 함
# tripmap를 통해서 mask를 0 0.5 1로 바꿔주는 함수 0 = 클래스1이 절대 아님 1 = 클래스1이 확실히 맞음 0.5 = 클래스1인지 클래스0인지 확실하지 않음
def trimap2f(trimap):
    # Handle both Tensors (from Albumentations) and PIL/NumPy arrays
    if isinstance(trimap, torch.Tensor):
        # If it's a tensor, assume it has integer values {1, 2, 3}
        t = trimap.long()
    else:
        # If it's PIL/NumPy, convert to tensor and scale back to integer values
        t = (img2t(trimap) * 255.0).long()

    # Ensure t is 3D (C, H, W) for consistency, even if C=1
    if t.dim() == 2:
        t = t.unsqueeze(0)
    
    # Create a new float tensor for the target soft labels
    target = torch.full_like(t, 0.0, dtype=torch.float32, device=t.device)
    
    # Apply the mapping based on the comments' intention
    # Pet (value 1) becomes 1.0
    target[t == 1] = 1.0
    # Background (value 2) becomes 0.0
    target[t == 2] = 0.0
    # Border (value 3) becomes 0.5 (uncertain)
    target[t == 3] = 0.5
    
    return target
    
# plt.imshow(t2img(trimap2f(train_pets_target)))
# plt.axis("off")
# plt.show()

#mask값 변환 확인
import numpy as np
mask = np.array(train_pets_target)
print("기존 마스크 :", np.unique(mask))

mask_converted = trimap2f(train_pets_target)
print("변환된 마스크 :", np.unique(mask_converted)) 


def __getitem__(self, idx):
    image, mask = super().__getitem__(idx)

    problem_indices = {1093, 1690, 1858, 2292, 2837, 2856, 3473}

    if idx in problem_indices:
        from pathlib import Path

        print("\n" + "=" * 70)
        print(f"[index={idx}] sample information")

        # torchvision OxfordIIITPet 내부에 저장된 실제 파일 경로
        image_path = self._images[idx]
        mask_path = self._segs[idx]

        image_stem = Path(image_path).stem
        mask_stem = Path(mask_path).stem

        print("image path:", image_path)
        print("mask path :", mask_path)
        print("image stem:", image_stem)
        print("mask stem :", mask_stem)
        print("same stem :", image_stem == mask_stem)

        # 디스크에 저장된 마스크 파일을 직접 열어서 확인
        raw_mask_from_file = np.array(Image.open(mask_path))

        print(f"\n[index={idx}] mask loaded directly from file")
        print("dtype:", raw_mask_from_file.dtype)
        print("shape:", raw_mask_from_file.shape)
        print("unique:", np.unique(raw_mask_from_file))
        print(
            "unique counts:",
            np.unique(raw_mask_from_file, return_counts=True)
        )
        print("min:", raw_mask_from_file.min())
        print("max:", raw_mask_from_file.max())
        print("pet pixel count:", np.sum(raw_mask_from_file == 1))
        print("background pixel count:", np.sum(raw_mask_from_file == 2))
        print("border pixel count:", np.sum(raw_mask_from_file == 3))

    # super().__getitem__()이 반환한 마스크 확인
    if idx in problem_indices:
        mask_before = np.array(mask)

        print(f"\n[index={idx}] before transform")
        print("type:", type(mask))
        print("dtype:", mask_before.dtype)
        print("shape:", mask_before.shape)
        print("unique:", np.unique(mask_before))
        print(
            "unique counts:",
            np.unique(mask_before, return_counts=True)
        )
        print("min:", mask_before.min())
        print("max:", mask_before.max())
        print("pet pixel count:", np.sum(mask_before == 1))
        print("background pixel count:", np.sum(mask_before == 2))
        print("border pixel count:", np.sum(mask_before == 3))

        # 디스크 원본과 super().__getitem__ 결과가 같은지 확인
        print(
            "raw file equals returned mask:",
            np.array_equal(raw_mask_from_file, mask_before)
        )

    image_np = np.array(image)
    mask_np = np.array(mask)

    if self.transform:
        augmented = self.transform(
            image=image_np,
            mask=mask_np
        )
        image = augmented["image"]
        mask = augmented["mask"]

    # transform 후, trimap2f 직전 확인
    if idx in problem_indices:
        print(f"\n[index={idx}] after transform, before trimap2f")
        print("type:", type(mask))
        print("dtype:", mask.dtype)

        if isinstance(mask, torch.Tensor):
            print("shape:", mask.shape)
            print("unique:", torch.unique(mask))
            print(
                "unique counts:",
                torch.unique(mask, return_counts=True)
            )
            print("min:", mask.min().item())
            print("max:", mask.max().item())
            print("pet pixel count:", (mask == 1).sum().item())
            print("background pixel count:", (mask == 2).sum().item())
            print("border pixel count:", (mask == 3).sum().item())

        else:
            print("shape:", mask.shape)
            print("unique:", np.unique(mask))
            print(
                "unique counts:",
                np.unique(mask, return_counts=True)
            )
            print("min:", mask.min())
            print("max:", mask.max())
            print("pet pixel count:", np.sum(mask == 1))
            print("background pixel count:", np.sum(mask == 2))
            print("border pixel count:", np.sum(mask == 3))

    mask = trimap2f(mask)

    # trimap2f 후 최종 마스크 확인
    if idx in problem_indices:
        print(f"\n[index={idx}] after trimap2f")
        print("type:", type(mask))
        print("dtype:", mask.dtype)
        print("shape:", mask.shape)
        print("unique:", torch.unique(mask))
        print(
            "unique counts:",
            torch.unique(mask, return_counts=True)
        )
        print("min:", mask.min().item())
        print("max:", mask.max().item())
        print("pet pixel count:", (mask == 1.0).sum().item())
        print("background pixel count:", (mask == 0.0).sum().item())
        print("border pixel count:", (mask == 0.5).sum().item())
        print("=" * 70)

    return image, mask
    
    
def tensor_trimap(t):
    x = t * 255
    x = x.to(torch.long)
    x = x - 1
    return x

#keyword argument들을: dictionary로 그대로 반환
def args_to_dict(**kwargs):
    return kwargs
    
# transform_dict = args_to_dict(
#     pre_transform=T.ToTensor(),
#     pre_target_transform=T.ToTensor(),
#     common_transform=T.Compose([
#         # Random Horizontal Flip as data augmentation.
#         T.RandomHorizontalFlip(p=0.5)
#     ]),
#     post_transform=T.Compose([
#         T.Resize((128, 128), interpolation=T.InterpolationMode.BILINEAR),
#         # Color Jitter as data augmentation.
#         T.ColorJitter(contrast=0.3),
#     ]),
#     post_target_transform=T.Compose([
#         T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST),
#         T.Lambda(tensor_trimap),
#     ]))


    
# pets_train = OxfordIIITPetsAugmented(
#     root=pets_path_train,
#     split="trainval",
#     target_types="segmentation",
#     download=False,
#     **transform_dict,
# )
# pets_test = OxfordIIITPetsAugmented(
#     root=pets_path_test,
#     split="test",
#     target_types="segmentation",
#     download=False,
#     **transform_dict,
# )

# pets_train_loader = torch.utils.data.DataLoader(
#     pets_train,
#     batch_size=64,
#     shuffle=True,
# )
# pets_test_loader = torch.utils.data.DataLoader(
#     pets_test,
#     batch_size=21,
#     shuffle=True,
# )

# (train_pets_inputs, train_pets_targets) = next(iter(pets_train_loader))
# (test_pets_inputs, test_pets_targets) = next(iter(pets_test_loader))

# print(f"Batch of training inputs shape: {train_pets_inputs.shape}")
# print(f"Batch of training targets shape: {train_pets_targets.shape}")


# pets_input_grid = torchvision.utils.make_grid(train_pets_inputs, nrow=8)
# plt.imshow(t2img(pets_input_grid))
# plt.axis("off")
# plt.show()

# pets_targets_grid = torchvision.utils.make_grid(train_pets_targets / 2.0, nrow=8)
# plt.imshow(t2img(pets_targets_grid))
# plt.axis("off")
# plt.show()

if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(
            min_height=256,
            min_width=256,
            border_mode=0,
            fill=0,
            fill_mask=0,
        ),
        ToTensorV2(),
    ])

    dataset = OxfordIIITPetsAugmented(
        root=pets_path_test,
        split="test",
        transform=transform,
        download=False,
    )

    problem_indices = [1093, 1690, 1858, 2292, 2837, 2856, 3473]

    for idx in problem_indices:
        print("\n" + "=" * 80)
        print(f"Loading sample {idx}")
        image, mask = dataset[idx]
        print("returned image shape:", image.shape)
        print("returned mask shape :", mask.shape)
        print("=" * 80)