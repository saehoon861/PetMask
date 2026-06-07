from collections import defaultdict
import argparse
import torch
import torch.nn.functional as F
from losses.loss import multiclass_dice_loss
from evalution import SegmentationMetrics  # Metrics 계산기 임포트
from dataset.dataset_load import OxfordIIITPetsAugmented
from dataset.dataset_load import tensor_trimap
from dataset.dataset_load import args_to_dict
from models.model import ResNetUNet
import time
import torchvision
import os
import torchvision.transforms as T
import wandb
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
def calc_loss(pred, target, metrics, ce_weight=0.5):
    target = target.squeeze(1).long()

    ce = F.cross_entropy(pred, target)
    dice = multiclass_dice_loss(pred, target)

    loss = ce * ce_weight + dice * (1 - ce_weight)

    metrics["ce"] += ce.detach().cpu().item() * target.size(0)
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
    images_list = []
    
    preds = torch.argmax(outputs, dim=1) # (N, H, W)
    
    for i in range(num_images):
        img = inputs[i].cpu().permute(1, 2, 0).numpy()
        # Denormalize if necessary (assuming ToTensor() [0, 1] range)
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        
        gt = labels[i].squeeze().cpu().numpy()
        pred = preds[i].cpu().numpy()
        
        images_list.append(wandb.Image(img, caption=f"Original_{i}"))
        images_list.append(wandb.Image(gt / 2.0, caption=f"GT_{i}")) # 0,1,2 scale for better visibility
        images_list.append(wandb.Image(pred / 2.0, caption=f"Pred_{i}"))

    wandb.log({
        "Visuals/Predictions": images_list
    }, step=epoch)




def train_model(model, dataloaders, optimizer, scheduler, checkpoint_path, num_epochs=25, patience=5, use_mock=False):
    # wandb initialization
    if not use_mock:
        wandb.init(
            project="PetMask",
            config={
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epochs": num_epochs,
                "batch_size": dataloaders['train'].batch_size,
                "patience": patience,
                "model": "ResNet18UNet"
            }
        )

    best_loss = 1e10
    stop_count = 0

    # 메트릭 계산기 초기화 (평균 IoU 등 계산용)
    metric_calc = SegmentationMetrics(average=True, ignore_background=False, activation='0-1')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            total_iou, total_pixel_acc = 0, 0

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
                    loss = calc_loss(outputs, labels, metrics)

                    # 성능 메트릭 계산 (IoU, Acc)
                    with torch.no_grad():
                        # target은 (N, H, W) 형태여야 함
                        target_for_metric = labels.squeeze(1).long()
                        pixel_acc, _, iou, _, _ = metric_calc(target_for_metric, outputs)
                        total_iou += iou * inputs.size(0)
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
                    f"{phase}/ce": metrics['ce'] / epoch_samples,
                    f"{phase}/dice": metrics['dice'] / epoch_samples,
                    f"{phase}/mIoU": total_iou / epoch_samples,
                    f"{phase}/pixel_acc": total_pixel_acc / epoch_samples,
                }, step=epoch)

            if phase == 'train':
              scheduler.step()

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
                # save the model weights if validation loss improved
                if epoch_loss < best_loss:
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
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Checkpoint file name")
    parser.add_argument("--use_mock", action="store_true", help="Use mock data for quick testing")
    parser.add_argument("--ce_weight", type=float, default=0.5)
    return parser.parse_args()

def main():
    args = get_args()

    if args.use_mock:
        num_class = 3
        mock_n, mock_h, mock_w = 8, 128, 128
        mock_inputs = torch.randn(mock_n, 3, mock_h, mock_w)
        mock_targets = torch.randint(low=0, high=num_class, size=(mock_n, 1, mock_h, mock_w))
        mock_dataset = TensorDataset(mock_inputs, mock_targets)
        dataloaders = {
            "train": DataLoader(mock_dataset, batch_size=2, shuffle=True),
            "val": DataLoader(mock_dataset, batch_size=2, shuffle=False),
        }
        num_epochs, patience = 1, 1
    else:
        num_class = 3
        working_dir = args.data_dir
        pets_path_train = os.path.join(working_dir, 'OxfordPets', 'train')
        pets_path_test = os.path.join(working_dir, 'OxfordPets', 'test')
        
        # Ensure data is downloaded
        torchvision.datasets.OxfordIIITPet(root=pets_path_train, split="trainval", target_types="segmentation", download=True)
        torchvision.datasets.OxfordIIITPet(root=pets_path_test, split="test", target_types="segmentation", download=True)

        transform_dict = args_to_dict(
            pre_transform=T.ToTensor(),
            pre_target_transform=T.ToTensor(),
            common_transform=T.Compose([T.RandomHorizontalFlip(p=0.5)]),
            post_transform=T.Compose([
                T.Resize((128, 128), interpolation=T.InterpolationMode.BILINEAR),
                T.ColorJitter(contrast=0.3),
            ]),
            post_target_transform=T.Compose([
                T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST),
                T.Lambda(tensor_trimap),
            ]))

        # 1. 모든 데이터를 로드하여 하나로 합칩니다 (Pooling)
        ds_trainval = OxfordIIITPetsAugmented(root=pets_path_train, split="trainval", target_types="segmentation", download=False, **transform_dict)
        ds_test = OxfordIIITPetsAugmented(root=pets_path_test, split="test", target_types="segmentation", download=False, **transform_dict)
        all_data = ConcatDataset([ds_trainval, ds_test])

        # 2. 모든 길이를 전체 데이터(total_len) 기준으로 계산하여 70:10:20 비율을 맞춥니다.
        total_len = len(all_data)
        test_len = int(0.2 * total_len)
        val_len = int(0.1 * total_len)
        train_len = total_len - (test_len + val_len)
        pets_train, pets_val, pets_test = random_split(all_data, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))

        # 전체 데이터셋 요약 정보 출력
        total_samples = len(pets_train) + len(pets_val) + len(pets_test)
        print("\n" + "="*40)
        print(f"{'Dataset Split Information':^40}")
        print("-" * 40)
        print(f" Train samples: {len(pets_train):>5} ({len(pets_train)/total_samples*100:>5.1f}%)")
        print(f" Val samples:   {len(pets_val):>5} ({len(pets_val)/total_samples*100:>5.1f}%)")
        print(f" Test samples:  {len(pets_test):>5} ({len(pets_test)/total_samples*100:>5.1f}%)")
        print(f" Total:         {total_samples:>5} (100.0%)")
        print("="*40 + "\n")

        dataloaders = {
            "train": DataLoader(pets_train, batch_size=args.batch_size, shuffle=True),
            "val": DataLoader(pets_val, batch_size=args.batch_size, shuffle=False)
        }
        num_epochs, patience = args.epochs, args.patience

    model = ResNetUNet(num_class).to(device)

    # freeze backbone layers
    for l in model.base_layers:
        for param in l.parameters():
            param.requires_grad = False

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.1)

    train_model(
        model, 
        dataloaders, 
        optimizer_ft, 
        exp_lr_scheduler, 
        checkpoint_path=args.checkpoint,
        num_epochs=num_epochs, 
        patience=patience,
        use_mock=args.use_mock
    )

if __name__ == "__main__":
    main()