from collections import defaultdict
import argparse
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from evalution import BinaryMetrics  # Metrics 계산기 임포트
from dataset.dataset_load import OxfordIIITPetsAugmented, trimap2f, args_to_dict
from models.model import ResNetUNet
import time
import torchvision
import os
import torchvision.transforms as T
import wandb
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset
import random
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    """학습 재현성을 위해 모든 랜덤 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 멀티 GPU 사용 시
    # CUDA 연산의 결정론적 동작 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed fixed to: {seed}")

# Sigmoid + BCEWithLogits
# - 출력: [N, 6, H, W]
# - 6개의 마스크 채널을 각각 독립적인 binary 문제로 봄
# - 각 채널마다 "이 픽셀이 해당 클래스인가? yes/no"를 판단
# - 한 픽셀이 여러 클래스에 동시에 속할 수 있는 multi-label 방식
# - loss: F.binary_cross_entropy_with_logits(pred, target)
# - target shape: [N, 6, H, W], 값은 0 또는 1


# Softmax + CrossEntropy
# - 출력: [N, 6, H, W]
# - 한 픽셀이 6개 클래스 중 딱 하나에만 속한다고 봄
# - 채널 방향으로 softmax를 적용해 클래스끼리 경쟁시킴
# - 최종 예측은 보통 torch.argmax(pred, dim=1)
# - 한 픽셀당 하나의 클래스만 선택하는 multi-class 방식
# - loss: F.cross_entropy(pred, target)
# - target shape: [N, H, W], 값은 0~5 클래스 인덱스

#BCE: 한 픽셀이 잘 맞췄는가
#dice loss: 예측 마스크 전체와 정답 마스크 전체가 얼마나 겹치는가?
def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def calc_loss(pred, target, metrics, bce_weight):
    # The model outputs 3 channels, but for this binary problem,
    # we assume the first channel corresponds to the 'pet' class.
    pred_pet = pred[:, 0]
      # Ensure target is float for BCE and Dice loss

    # Target is already a float tensor (0.0, 0.5, 1.0)
    # Squeeze the channel dimension if it exists
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(pred_pet, target)
    dice = dice_loss(pred_pet, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics["bce"] += bce.detach().cpu().item() * target.size(0)
    metrics["dice"] += dice.detach().cpu().item() * target.size(0)
    metrics["loss"] += loss.detach().cpu().item() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))
    
def log_images_to_wandb(inputs, labels, outputs, epoch):
    """
    WandB에 예측 결과 이미지를 업로드하는 함수
    """
    # 한 배치에서 최대 4개까지만 시각화
    num_images = min(inputs.size(0), 4)
    
    probs = torch.sigmoid(outputs[:, 0])
    preds = (probs >= 0.5).long() # Prediction is already 0 or 1
    
    class_labels = {
        0: "background",
        1: "border", # Assuming 0.5 maps to border
        2: "pet"     # Assuming 1.0 maps to pet
    }

    images_to_log = []

    for i in range(num_images):
        img_np = inputs[i].cpu().permute(1, 2, 0).numpy()
        # Denormalize if necessary (assuming ToTensor() [0, 1] range)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)
        
        gt_float = labels[i].squeeze().cpu().numpy()
        
        # Convert float labels (0.0, 0.5, 1.0) to integer class labels (0, 1, 2)
        gt_int = np.zeros_like(gt_float, dtype=np.uint8)
        gt_int[gt_float == 0.5] = 1 # Border
        gt_int[gt_float == 1.0] = 2 # Pet
        # Background (0.0) remains 0

        # Create image with ground truth mask
        images_to_log.append(wandb.Image(
            img_np,
            masks={
                "ground_truth": {
                    "mask_data": gt_int,
                    "class_labels": class_labels
                },
                 "prediction": {
                    "mask_data": preds[i].cpu().numpy(),
                    "class_labels": {0: "background", 1: "pet"} # Prediction is binary
                }
            },
            caption=f"Sample {i} - GT and Prediction"
        ))

    wandb.log({
        "Visuals/Predictions": images_to_log
    }, step=epoch)




def train_model(model, dataloaders, optimizer, scheduler, checkpoint_path, num_epochs=25, patience=5, use_mock=False, args=None):
    # wandb initialization
    if not use_mock:
        wandb.init(
            project="PetMask",
            name = (
                f"bs{args.batch_size}"
                f"_lr{args.lr}"
                f"_ftlr{args.fine_tune_lr}"
                f"_wd{args.weight_decay}"
                f"_ce{args.ce_weight}"
                f"_min_delta{args.min_delta}"
                ),
            config={
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epochs": num_epochs,
                "batch_size": dataloaders['train'].batch_size,
                "patience": patience,
                "min_delta": args.min_delta,
                "model": "ResNet18UNet"
            }
        )

    best_loss = 1e10
    stop_count = 0

    # 메트릭 계산기 초기화 (평균 IoU 등 계산용)
    metric_calc = BinaryMetrics(activation='sigmoid')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # At epoch 5, unfreeze the backbone and create a new optimizer for fine-tuning
        if epoch == 10:
            print("\n" + "="*40)
            print("Epoch 5: Unfreezing backbone and switching to AdamW for fine-tuning.")
            print("="*40 + "\n")
            
            for layer in model.base_layers:
                for param in layer.parameters():
                    param.requires_grad = True

            # Create a new optimizer for fine-tuning that includes all parameters
            # Separate parameters for different learning rates
        
            all_model_params = list(model.parameters())
            base_model_params = list(model.base_model.parameters())
            decoder_params = [p for p in all_model_params if p not in base_model_params]

            optimizer = optim.AdamW([
                {'params': base_model_params, 'lr': args.fine_tune_lr},
                {'params': decoder_params, 'lr': args.lr}
            ], weight_decay=args.weight_decay)
            
            # Create a new scheduler for the new optimizer
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )

            # Log the new learning rate to wandb
            if not use_mock:
                wandb.config.update({"fine_tune_lr": args.fine_tune_lr}, allow_val_change=True)

            # Reset early stopping counter to give fine-tuning a fair chance
            stop_count = 0

        since = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            total_dice, total_pixel_acc = 0, 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train

                #torch.set_grad_enabled(조건)은 조건에 따라 PyTorch가
                #gradient 계산 기록을 할지 말지 정하는 기능

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics, bce_weight= args.ce_weight)

                    # 성능 메트릭 계산 (IoU, Acc)
                    with torch.no_grad():
                        # target을 0.5 기준으로 hard-label로 변환
                        target_for_metric = (labels.squeeze(1) > 0.5).long()
                        
                        # BinaryMetrics는 1채널 예측을 기대하므로 첫 채널만 전달
                        # 반환값: [pixel_acc, dice, precision, specificity, recall]
                        metrics_list = metric_calc(target_for_metric, outputs[:, 0:1])
                        pixel_acc = metrics_list[0]
                        dice = metrics_list[1]
                        
                        total_dice += dice * inputs.size(0)
                        total_pixel_acc += pixel_acc * inputs.size(0)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 시각화 로그: 검증 단계의 첫 번째 배치 이미지만 기록
                if phase == 'val' and epoch_samples == 0 and not use_mock:
                    log_images_to_wandb(inputs, labels, outputs, epoch)

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if not use_mock:
                wandb.log({
                    f"{phase}/loss": epoch_loss,
                    f"{phase}/bce": metrics['bce'] / epoch_samples,
                    f"{phase}/dice": metrics['dice'] / epoch_samples,
                    f"{phase}/dice_score": total_dice / epoch_samples,
                    f"{phase}/pixel_acc": total_pixel_acc / epoch_samples,
                }, step=epoch)

            if phase == 'train':
              # scheduler.step() is now called after the validation phase.

              # model.parameters()는 모델의 학습 가능한 파라미터(weight, bias 등)를 optimizer에 전달한다.
              # optimizer는 전달받은 파라미터만 loss.backward()로 계산된 gradient를 이용해 업데이트한다.
              # lr=1e-5는 초기 learning rate를 0.00001로 설정한다는 의미이다.
              # scheduler가 없으면 이 값이 유지되고, scheduler가 있으면 학습 중 변경될 수 있다.
            

              for param_group in optimizer.param_groups:
                  lr = param_group['lr']
                  print("LR", lr)
                  if not use_mock:
                      wandb.log({"learning_rate": lr}, step=epoch)

            if phase == 'val':
                # Update scheduler based on validation loss
                scheduler.step(epoch_loss)
                
                # save the model weights if validation loss improved
                if best_loss - epoch_loss > args.min_delta:
                    print(f"saving best model to {checkpoint_path}")
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), checkpoint_path)
                    stop_count = 0  # reset early stopping counter
                else:
                    stop_count += 1
                    print(f"EarlyStopping counter: {stop_count} out of {patience}")

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        if stop_count >= patience:
            print("Early stopping triggered. Training halted.")
            break
        
    if not use_mock:
        wandb.finish()

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(torch.load(checkpoint_path))
    return model


def get_args():
    parser = argparse.ArgumentParser(description="PetMask Training Script")
    parser.add_argument("--data_dir", type=str, default="/home/sehoon/workspace/PetMask/src/dataset", help="Dataset root directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Input batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5, help="Learning rate for fine-tuning after backbone unfreeze")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.0001, help="Minimum change to qualify as an improvement for early stopping")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Checkpoint file name")
    parser.add_argument("--use_mock", action="store_true", help="Use mock data for quick testing")
    parser.add_argument("--ce_weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    return parser.parse_args()

def main():
    args = get_args()

    # 학습 시작 전 시드 고정 (기본값 42)
    set_seed(args.seed)

    if args.use_mock:
        num_class = 1 # Binary segmentation (pet vs. background)
        mock_n, mock_h, mock_w = 8, 128, 128
        mock_inputs = torch.randn(mock_n, 3, mock_h, mock_w)
        # Generate binary mock targets (0 or 1)
        mock_targets = torch.randint(low=0, high=2, size=(mock_n, 1, mock_h, mock_w))
        mock_dataset = TensorDataset(mock_inputs, mock_targets)
        dataloaders = {
            "train": DataLoader(mock_dataset, batch_size=2, shuffle=True),
            "val": DataLoader(mock_dataset, batch_size=2, shuffle=False),
        }
        num_epochs, patience = 1, 1
    else:
        num_class = 1 # Binary segmentation (pet vs. background)
        working_dir = args.data_dir
        pets_path_train = os.path.join(working_dir, 'OxfordPets', 'train')
        pets_path_test = os.path.join(working_dir, 'OxfordPets', 'test')
        
        # Ensure data is downloaded
        torchvision.datasets.OxfordIIITPet(root=pets_path_train, split="trainval", target_types="segmentation", download=True)
        torchvision.datasets.OxfordIIITPet(root=pets_path_test, split="test", target_types="segmentation", download=True)

        # Define Albumentations pipelines
        IMG_SIZE = 256
        # Note: Albumentations' Normalize uses ImageNet stats by default
        train_transform = A.Compose([
            A.LongestMaxSize(max_size=IMG_SIZE),

            A.PadIfNeeded(
                min_height=IMG_SIZE,
                min_width=IMG_SIZE,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
            ),

            A.HorizontalFlip(p=0.5),

            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                shear=(-5, 5),
                p=0.5,
            ),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5)),
                A.GaussNoise(std_range=(0.01, 0.05)),
            ], p=0.2),

            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
                p=0.4,
            ),

            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                fill=0,
                fill_mask=None,
                p=0.2,
            ),

            A.Normalize(),
            ToTensorV2(),
        ])
    
        val_transform = A.Compose([
            A.LongestMaxSize(max_size=IMG_SIZE),
            A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
            A.Normalize(),
            ToTensorV2(),
        ])

        # Create full datasets with different transforms
        full_train_dataset = OxfordIIITPetsAugmented(root=pets_path_train, split="trainval", transform=train_transform)
        # The validation set should not have random augmentations, so we create a separate dataset instance for it
        full_val_dataset = OxfordIIITPetsAugmented(root=pets_path_train, split="trainval", transform=val_transform)
        pets_test = OxfordIIITPetsAugmented(root=pets_path_test, split="test", transform=val_transform)

        # Split the original training data into a new training set and a validation set (e.g., 90% train, 10% val)
        train_len = int(len(full_train_dataset) * 0.8)
        val_len = len(full_train_dataset) - train_len
        
        # Use a generator for reproducibility
        generator = torch.Generator().manual_seed(args.seed)
        # Important: Split the indices, then create Subsets with the correct transforms
        indices = torch.randperm(len(full_train_dataset), generator=generator).tolist()
        
        pets_train = torch.utils.data.Subset(full_train_dataset, indices[:train_len])
        pets_val = torch.utils.data.Subset(full_val_dataset, indices[train_len:])
        
        # The test set is now correctly separated from the validation set.

        # 전체 데이터셋 요약 정보 출력
        total_samples = len(pets_train) + len(pets_val) + len(pets_test)
        print("\n" + "="*40)
        print(f"{'Dataset Split Information':^40}")
        print("-" * 40)
        print(f" Train samples: {len(pets_train):>5} ({(len(pets_train)/total_samples*100):5.1f}%)")
        print(f" Val samples:   {len(pets_val):>5} ({(len(pets_val)/total_samples*100):5.1f}%)")
        print(f" Test samples:  {len(pets_test):>5} ({(len(pets_test)/total_samples*100):5.1f}%)")
        print(f" Total samples: {total_samples:>5} (100.0%)")
        print("="*40 + "\n")

        dataloaders = {
            "train": DataLoader(pets_train, batch_size=args.batch_size, shuffle=True),
            "val": DataLoader(pets_val, batch_size=args.batch_size, shuffle=False),
            "test": DataLoader(pets_test, batch_size=args.batch_size, shuffle=False),
        }
        num_epochs, patience = args.epochs, args.patience

    model = ResNetUNet(num_class).to(device)

    # freeze backbone layers
    for l in model.base_layers:
        for param in l.parameters():
            param.requires_grad = False

    optimizer_ft = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer_ft,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    train_model(
        model, 
        dataloaders, 
        optimizer_ft, 
        exp_lr_scheduler, 
        checkpoint_path=args.checkpoint,
        num_epochs=num_epochs, 
        patience=patience,
        use_mock=args.use_mock,
        args=args
    )

if __name__ == "__main__":
    main()