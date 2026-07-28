import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import random
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    BinaryAveragePrecision,
    BinaryPrecisionRecallCurve
)

def set_seed(seed=42):
    """Sets a random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_pr_curve(precision, recall, ap_score, save_path):
    """Plots and saves the Precision-Recall curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', lw=2)
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve (AP = {ap_score:.4f})')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"PR curve saved to {save_path}")

def plot_score_distribution(scores, save_path):
    """Plots and saves the distribution of Dice scores."""
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=30, color='skyblue', edgecolor='black')
    mean_score = np.mean(scores)
    plt.axvline(mean_score, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.4f}')
    plt.title('Distribution of Dice Scores Across Test Set')
    plt.xlabel('Dice Score')
    plt.ylabel('Number of Images')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(save_path)
    plt.close()
    print(f"Score distribution plot saved to {save_path}")

def save_qualitative_results(sample_data, threshold, save_path):
    """Saves a 4-panel image comparing original, GT, prediction, and error map."""

    image = sample_data["image"]
    gt_mask = sample_data["gt_mask"]
    pred_logits = sample_data["pred_logits"]
    score = sample_data["score"]

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = image.numpy().transpose((1, 2, 0))
    image = std * image + mean
    image = np.clip(image, 0, 1)

    gt_mask = gt_mask.numpy().squeeze()
    pred_mask = (
        torch.sigmoid(pred_logits) > threshold
    ).numpy().squeeze()

    # Create error map
    # TP: green, FP: red, FN: yellow
    error_map = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    error_map[(pred_mask == 1) & (gt_mask == 1)] = [0, 255, 0]  # Green (TP)
    error_map[(pred_mask == 1) & (gt_mask == 0)] = [255, 0, 0]  # Red (FP)
    error_map[(pred_mask == 0) & (gt_mask == 1)] = [255, 255, 0] # Yellow (FN)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Dice Score: {score:.4f}', fontsize=16)

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')

    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title(f"Prediction (Threshold={threshold:.2f})")
    axes[2].axis('off')
    
    axes[3].imshow(error_map)
    axes[3].set_title("Error Map (G: TP, R: FP, Y: FN)")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

  
def binary_dice_score(logits, target, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()
    target = target.float()

    intersection = (pred * target).sum()
    denominator = pred.sum() + target.sum()

    return (2 * intersection + eps) / (denominator + eps)


def run_evaluation(checkpoint_path, data_dir, batch_size, img_size, threshold, num_examples):
    set_seed(42)
    from dataset.dataset_load import OxfordIIITPetsAugmented
    from models.model import ResNetUNet

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_class = 1

    # --- 1. Setup Output Directory ---
    output_dir = "evaluation_results"
    qualitative_dir = os.path.join(output_dir, "qualitative_results")
    os.makedirs(qualitative_dir, exist_ok=True)
    print(f"Evaluation results will be saved in '{output_dir}/'")

    # --- 2. Load Model & Data ---
    model = ResNetUNet(num_class).to(device)
    if os.path.exists(checkpoint_path):
        #torch.load:저장된 가중치 파일을 읽어오는 역할
        #load_state_dict:모델에 가중치를 불러오는 역할
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Checkpoint not found at {checkpoint_path}"); return
    model.eval()

    val_transform = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.Normalize(),
        ToTensorV2(),
    ])
    
    pets_path_test = os.path.join(data_dir, 'OxfordPets', 'test')
    torchvision.datasets.OxfordIIITPet(root=pets_path_test, split="test", target_types="segmentation", download=True)
    
    # Use batch_size=1 for per-image analysis
    test_dataset = OxfordIIITPetsAugmented(root=pets_path_test, split="test", transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"\nEvaluating on {len(test_dataset)} test images...")

    # --- 3. Initialize Metric Calculators ---
    # Global metrics calculated over the entire dataset
    global_metrics = {
        "Accuracy": BinaryAccuracy(threshold=threshold).to(device),
        "Precision": BinaryPrecision(threshold=threshold).to(device),
        "Recall": BinaryRecall(threshold=threshold).to(device),
        "Specificity": BinarySpecificity(threshold=threshold).to(device),
        "AP": BinaryAveragePrecision(thresholds=101).to(device),
    }
    pr_curve_calculator = BinaryPrecisionRecallCurve(thresholds=101).to(device)
    
    # Per-image Dice calculator
   
    global_intersection = 0.0
    global_pred_sum = 0.0
    global_target_sum = 0.0
    # --- 4. Evaluation Loop ---
    all_results = []
    empty_gt_count = 0

    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            image, mask = image.to(device), mask.to(device)

            binary_mask = (mask.squeeze(1) > 0.5).int()

            if binary_mask.sum().item() == 0:
                empty_gt_count += 1
                print(
                    f"[Empty GT] index={i}, "
                    f"raw_unique={torch.unique(mask)}, "
                    f"mask_min={mask.min().item()}, "
                    f"mask_max={mask.max().item()}"
                )
            
            logits = model(image)
            pred_binary = (torch.sigmoid(logits.squeeze(1)) > threshold).float()
            target_binary = binary_mask.float()

            global_intersection += (pred_binary * target_binary).sum().item()
            global_pred_sum += pred_binary.sum().item()
            global_target_sum += target_binary.sum().item()
            # Update global metrics
            for metric in global_metrics.values():
                metric.update(logits.squeeze(1), binary_mask)
            pr_curve_calculator.update(logits.squeeze(1), binary_mask)

            # Calculate and store per-image score and data
            score = binary_dice_score( logits.squeeze(1), binary_mask, threshold=threshold).item()
            all_results.append({
                "score": score,
                "image": image.squeeze(0).detach().cpu(),
                "gt_mask": binary_mask.squeeze(0).detach().cpu(),
                "pred_logits": logits.squeeze(0).detach().cpu()
                })

    # --- 5. Compute, Print, and Visualize Results ---
    # Global Metrics
    global_dice = (2 * global_intersection + 1e-7) / (global_pred_sum + global_target_sum + 1e-7)
    final_scores = {name: metric.compute().item() for name, metric in global_metrics.items()}
    
    print("\n--- Test Set Performance (Calculated by TorchMetrics) ---")
    print(f"Metrics calculated at threshold: {threshold:.4f}")
    print("-" * 55)
    print(f"Dice Score:        {global_dice:.4f}")
    print(f"Accuracy:          {final_scores['Accuracy']:.4f}")
    print(f"Precision:         {final_scores['Precision']:.4f}")
    print(f"Recall:            {final_scores['Recall']:.4f}")
    print(f"Specificity:       {final_scores['Specificity']:.4f}")
    print(f"Average Precision: {final_scores['AP']:.4f} (Threshold-Independent)")
    print("Empty GT count:", empty_gt_count)
    # PR Curve
    precision, recall, _ = pr_curve_calculator.compute()
    plot_pr_curve(precision.cpu(), recall.cpu(), final_scores['AP'], os.path.join(output_dir, "pr_curve.png"))
    
    # Score Distribution
    dice_scores = [r['score'] for r in all_results]
    plot_score_distribution(dice_scores, os.path.join(output_dir, "score_distribution.png"))

    # Qualitative Analysis
    all_results.sort(key=lambda x: x['score'], reverse=True)
    best_cases = all_results[:num_examples]
    worst_cases = all_results[-num_examples:]

    print(f"\nSaving {num_examples} best and worst case qualitative examples...")
    for i in range(num_examples):
        save_qualitative_results(best_cases[i], threshold, os.path.join(qualitative_dir, f"best_case_{i+1}.png"))
        # Reverse worst cases for naming consistency (worst_case_1 is the absolute worst)
        save_qualitative_results(worst_cases[-(i+1)], threshold, os.path.join(qualitative_dir, f"worst_case_{i+1}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PetMask Comprehensive Evaluation Script")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Path to trained model checkpoint.")
    parser.add_argument("--data_dir", type=str, default="/content/PetMask/src/dataset", help="Dataset root directory.")
    parser.add_argument("--img_size", type=int, default=256, help="Image size used during training.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Optimal threshold found on the validation set.")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of best/worst examples to save.")
    
    args = parser.parse_args()
    
    run_evaluation(args.checkpoint, args.data_dir, 1, args.img_size, args.threshold, args.num_examples)