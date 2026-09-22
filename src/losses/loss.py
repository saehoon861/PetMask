import torch
import torch.nn as nn
import torch.nn.functional as F

#dice loss를 구하기 위해서 타켓을 원핫인코딩 진행: shape이 [N, H, W]인 타겟을 [N, C, H, W]로 바꿔주는 함수
def multiclass_dice_loss(pred, target, smooth=1.):
    # pred: [N, C, H, W] logits
    # target: [N, H, W] class index

    num_classes = pred.shape[1]

    pred = torch.softmax(pred, dim=1)

    target_onehot = F.one_hot(target, num_classes=num_classes)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()

    pred = pred.contiguous()
    target_onehot = target_onehot.contiguous()

    intersection = (pred * target_onehot).sum(dim=(2, 3))

    dice = (2. * intersection + smooth) / (
        pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3)) + smooth
    )

    return 1 - dice.mean()