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


class OxfordIIITPetsAugmented(torchvision.datasets.OxfordIIITPet):
    def __init__(self, root: str, split: str, transform=None, **kwargs):
        super().__init__(root=root, split=split, target_types="segmentation", **kwargs)
        self.transform = transform

    def __getitem__(self, idx):
        # super() returns a PIL image and a PIL mask
        image, mask = super().__getitem__(idx)

        # Convert PIL to NumPy for Albumentations
        image_np = np.array(image)
        mask_np = np.array(mask)

        if self.transform:
            augmented = self.transform(image=image_np, mask=mask_np)
            image = augmented['image']
            mask = augmented['mask']
        
        # After augmentations, mask is a NumPy array. We apply trimap2f.
        # trimap2f is designed to take a PIL image or NumPy array and convert to a soft-label tensor.
        mask = trimap2f(mask)
        
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

