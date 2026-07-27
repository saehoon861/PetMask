import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import random
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.classification import (
    Dice, 
    BinaryAccuracy, 
    BinaryPrecision, 
    BinaryRecall, 
    BinarySpecificity, 
    BinaryAveragePrecision
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dice_score_from_tensors(preds, targets, smooth=1.):
    """
    Computes the Dice score for a batch of predictions and targets.
    preds and targets are expected to be binary (0 or 1).
    """
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def find_optimal_threshold(model, dataloader, device):
    print("\n" + "="*40)
    print("Finding optimal threshold on test set...")
    print("="*40 + "\n")

    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            
            # Model outputs logits, apply sigmoid to get probabilities
            outputs = torch.sigmoid(model(inputs)[:, 0]) 
            
            # Ground truth labels should be binary for metric calculation
            # target is float (0.0, 0.5, 1.0), convert to 0 or 1 for pet mask
            targets_binary = (labels.squeeze(1) > 0.5).long() 

            all_preds.append(outputs.cpu())
            all_targets.append(targets_binary.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Search thresholds from 0.05 to 0.95 with step 0.01
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_dice = 0.0
    best_threshold = 0.0

    for thr in thresholds:
        pred_masks = (all_preds > thr).long()
        current_dice = dice_score_from_tensors(pred_masks, all_targets)

        if current_dice > best_dice:
            best_dice = current_dice
            best_threshold = thr
    
    print(f"Optimal threshold found: {best_threshold:.2f}")
    print(f"With Dice Score at optimal threshold: {best_dice:.4f}")
    return best_threshold

def run_evaluation(checkpoint_path, data_dir, batch_size, img_size):
    # 평가 시에도 동일한 데이터 분할을 보장하기 위해 시드 설정
    set_seed(42)

    from dataset.dataset_load import OxfordIIITPetsAugmented
    from models.model import ResNetUNet

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_class = 1 # Binary segmentation (pet vs. not-pet)

    # 1. 모델 로드
    model = ResNetUNet(num_class).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    model.eval()

    # 2. 테스트셋 로드 (train.py의 validation transform과 동일하게 설정)
    val_transform = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.Normalize(),
        ToTensorV2(),
    ])
    
    pets_path_test = os.path.join(data_dir, 'OxfordPets', 'test')
    # Ensure data is downloaded if not present
    torchvision.datasets.OxfordIIITPet(root=pets_path_test, split="test", target_types="segmentation", download=True)
    
    test_dataset = OxfordIIITPetsAugmented(root=pets_path_test, split="test", transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nEvaluating on {len(test_dataset)} test images...")

    # 3. TorchMetrics 계산기 초기화
    # AP는 확률(logits)을 직접 사용하므로 별도 관리
    ap_calculator = BinaryAveragePrecision().to(device)
    
    # 나머지 메트릭은 0.5 임계값을 기준으로 계산
    metrics_at_0_5 = {
        "Accuracy": BinaryAccuracy().to(device),
        "Dice": Dice().to(device),
        "Precision": BinaryPrecision().to(device),
        "Recall": BinaryRecall().to(device),
        "Specificity": BinarySpecificity().to(device),
    }

    # 4. 평가 루프 실행
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device) # Shape: (N, 1, H, W), values are 0.0, 0.5, 1.0

            # Convert mask to binary format (pet=1, other=0)
            binary_masks = (masks.squeeze(1) > 0.5).int() # Shape: (N, H, W), torchmetrics는 int 타입을 선호
            
            outputs = model(images) # Shape: (N, 1, H, W), logits

            # 모든 메트릭 계산기에 현재 배치의 예측과 정답을 전달하여 상태 업데이트
            ap_calculator.update(outputs.squeeze(1), binary_masks)
            for metric in metrics_at_0_5.values():
                metric.update(outputs.squeeze(1), binary_masks)

    # 5. 최종 결과 계산 및 출력
    final_results = {}
    for name, metric in metrics_at_0_5.items():
        final_results[name] = metric.compute().item()
    
    final_results["Average Precision"] = ap_calculator.compute().item()

    print("\n--- Test Set Performance (Calculated by TorchMetrics) ---")
    print(f"Dice Score (Threshold=0.5): {final_results['Dice']:.4f}")
    print(f"Accuracy (Threshold=0.5):   {final_results['Accuracy']:.4f}")
    print(f"Precision (Threshold=0.5):  {final_results['Precision']:.4f}")
    print(f"Recall (Threshold=0.5):     {final_results['Recall']:.4f}")
    print(f"Specificity (Threshold=0.5):{final_results['Specificity']:.4f}")
    print(f"Average Precision (AP):     {final_results['Average Precision']:.4f}")
    
    # 6. 테스트셋 기준 최적 임계값 탐색 및 결과 보고
    find_optimal_threshold(model, test_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PetMask Evaluation Script")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, default="/home/sehoon/workspace/PetMask/src/dataset", help="Dataset root directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Input batch size for evaluation")
    parser.add_argument("--img_size", type=int, default=256, help="Image size used during training")
    
    args = parser.parse_args()
    
    run_evaluation(args.checkpoint, args.data_dir, args.batch_size, args.img_size)