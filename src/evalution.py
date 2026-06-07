import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torchvision.transforms as T
import random
import numpy as np


class SegmentationMetrics(object):
    r"""Calculate common metrics in semantic segmentation to evalueate model preformance.

    Supported metrics: Pixel accuracy, Dice Coeff, IoU, precision score and recall score.
    
    Pixel accuracy measures how many pixels in a image are predicted correctly.

    Dice Coeff is a measure function to measure similarity over 2 sets, which is usually used to
    calculate the similarity of two samples. Dice equals to f1 score in semantic segmentation tasks.
    
    It should be noted that Dice Coeff and Intersection over Union are highly related, so you need 
    NOT calculate these metrics both, the other can be calcultaed directly when knowing one of them.

    Precision describes the purity of our positive detections relative to the ground truth. Of all
    the objects that we predicted in a given image, precision score describes how many of those objects
    actually had a matching ground truth annotation.

    Recall describes the completeness of our positive predictions relative to the ground truth. Of
    all the objected annotated in our ground truth, recall score describes how many true positive instances
    we have captured in semantic segmentation.

    Args:
        eps: float, a value added to the denominator for numerical stability.
            Default: 1e-5

        average: bool. Default: ``True``
            When set to ``True``, average Dice Coeff, precision and recall are
            returned. Otherwise Dice Coeff, precision and recall of each class
            will be returned as a numpy array.

        ignore_background: bool. Default: ``True``
            When set to ``True``, the class will not calculate related metrics on
            background pixels. When the segmentation of background pixels is not
            important, set this value to ``True``.

        activation: [None, 'none', 'softmax' (default), 'sigmoid', '0-1']
            This parameter determines what kind of activation function that will be
            applied on model output.

    Input:
        y_true: :math:`(N, H, W)`, torch tensor, where we use int value between (0, num_class - 1)
        to denote every class, where ``0`` denotes background class.
        y_pred: :math:`(N, C, H, W)`, torch tensor.

    Examples::
        >>> metric_calculator = SegmentationMetrics(average=True, ignore_background=False)
        >>> pixel_accuracy, dice, precision, recall = metric_calculator(y_true, y_pred)
    """
    def __init__(self, eps=1e-5, average=True, ignore_background=False, activation='0-1'):
        self.eps = eps
        self.average = average
        self.ignore = ignore_background
        self.activation = activation

    @staticmethod
    def _one_hot(gt, pred, class_num):
        # transform sparse mask into one-hot mask
        # shape: (B, H, W) -> (B, C, H, W)
        input_shape = tuple(gt.shape)  # (N, H, W, ...)
        new_shape = (input_shape[0], class_num) + input_shape[1:]
        one_hot = torch.zeros(new_shape).to(pred.device, dtype=torch.float)
        target = one_hot.scatter_(1, gt.unsqueeze(1).long().data, 1.0)
        return target

    @staticmethod
    def _get_class_data(gt_onehot, pred, class_num):
        # perform calculation on a batch
        # for precise result in a single image, plz set batch size to 1
        matrix = np.zeros((3, class_num))

        # calculate tp, fp, fn per class
        for i in range(class_num):
            # pred shape: (N, H, W)
            class_pred = pred[:, i, :, :]
            # gt shape: (N, H, W), binary array where 0 denotes negative and 1 denotes positive
            class_gt = gt_onehot[:, i, :, :]

            pred_flat = class_pred.contiguous().view(-1, )  # shape: (N * H * W, )
            gt_flat = class_gt.contiguous().view(-1, )  # shape: (N * H * W, )

            tp = torch.sum(gt_flat * pred_flat)
            fp = torch.sum(pred_flat) - tp
            fn = torch.sum(gt_flat) - tp

            matrix[:, i] = tp.item(), fp.item(), fn.item()

        return matrix

    def _calculate_multi_metrics(self, gt, pred, class_num):
        # calculate metrics in multi-class segmentation
        matrix = self._get_class_data(gt, pred, class_num)
        if self.ignore:
            matrix = matrix[:, 1:]

        # tp = np.sum(matrix[0, :])
        # fp = np.sum(matrix[1, :])
        # fn = np.sum(matrix[2, :])

        # Global Pixel Accuracy: (Sum of all TPs) / (Total Number of Pixels)
        # Total Pixels = Total TP + Total FN (정답지 기준 모든 픽셀의 합)
        pixel_acc = (np.sum(matrix[0, :]) + self.eps) / (np.sum(matrix[0, :] + matrix[2, :]) + self.eps)
        dice = (2 * matrix[0] + self.eps) / (2 * matrix[0] + matrix[1] + matrix[2] + self.eps)
        iou = (matrix[0] + self.eps) / (matrix[0] + matrix[1] + matrix[2] + self.eps)
        precision = (matrix[0] + self.eps) / (matrix[0] + matrix[1] + self.eps)
        recall = (matrix[0] + self.eps) / (matrix[0] + matrix[2] + self.eps)

        if self.average:
            dice = np.average(dice)
            iou = np.average(iou)
            precision = np.average(precision)
            recall = np.average(recall)
        return pixel_acc, dice, iou, precision, recall

    def __call__(self, y_true, y_pred):
        class_num = y_pred.size(1)

        if self.activation in [None, 'none']:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "softmax":
            activation_fn = nn.Softmax(dim=1)
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            pred_argmax = torch.argmax(y_pred, dim=1)
            activated_pred = self._one_hot(pred_argmax, y_pred, class_num)
        else:
            raise NotImplementedError("Not a supported activation!")

        gt_onehot = self._one_hot(y_true, y_pred, class_num)
        pixel_acc, dice, iou, precision, recall = self._calculate_multi_metrics(gt_onehot, activated_pred, class_num)
        return pixel_acc, dice, iou, precision, recall


class BinaryMetrics():
    r"""Calculate common metrics in binary cases.
    In binary cases it should be noted that y_pred shape shall be like (N, 1, H, W), or an assertion 
    error will be raised.
    Also this calculator provides the function to calculate specificity, also known as true negative 
    rate, as specificity/TPR is meaningless in multiclass cases.
    """
    def __init__(self, eps=1e-5, activation='0-1'):
        self.eps = eps
        self.activation = activation

    def _calculate_overlap_metrics(self, gt, pred):
        output = pred.view(-1, )
        target = gt.view(-1, ).float()

        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP
        fn = torch.sum((1 - output) * target)  # FN
        tn = torch.sum((1 - output) * (1 - target))  # TN

        pixel_acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
        dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps)
        recall = (tp + self.eps) / (tp + fn + self.eps)
        specificity = (tn + self.eps) / (tn + fp + self.eps)

        return pixel_acc, dice, precision, specificity, recall

    def __call__(self, y_true, y_pred):
        # y_true: (N, H, W)
        # y_pred: (N, 1, H, W)
        if self.activation in [None, 'none']:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            sigmoid_pred = nn.Sigmoid()(y_pred)
            activated_pred = (sigmoid_pred > 0.5).float().to(y_pred.device)
        else:
            raise NotImplementedError("Not a supported activation!")

        assert activated_pred.shape[1] == 1, 'Predictions must contain only one channel' \
                                             ' when performing binary segmentation'
        pixel_acc, dice, precision, specificity, recall = self._calculate_overlap_metrics(y_true.to(y_pred.device,
                                                                                                    dtype=torch.float),
                                                                                          activated_pred)
        return [pixel_acc, dice, precision, specificity, recall]

def visualize_prediction(image, target, prediction, save_path=None):
    """
    시각적으로 모델의 예측 결과를 확인하기 위한 함수
    image: (3, H, W) Tensor
    target: (H, W) Tensor
    prediction: (C, H, W) Tensor (Logits)
    """
    image = image.cpu().permute(1, 2, 0).numpy()
    # Normalize image for visualization if needed
    image = (image - image.min()) / (image.max() - image.min())
    
    target = target.cpu().numpy()
    pred_mask = torch.argmax(prediction, dim=0).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(target, cmap='viridis')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(pred_mask, cmap='viridis')
    axes[2].set_title("Prediction")
    axes[2].axis('off')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def calculate_confusion_matrix(y_true, y_pred, num_classes):
    """픽셀 단위의 오차 행렬을 계산합니다."""
    y_true = y_true.view(-1).cpu().numpy().astype(int)
    y_pred = y_pred.view(-1).cpu().numpy().astype(int)
    cm = np.bincount(num_classes * y_true + y_pred, minlength=num_classes**2)
    return cm.reshape(num_classes, num_classes)

def plot_confusion_matrix(cm, class_names, save_path):
    """오차 행렬을 시각화하여 저장합니다."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Pixel-wise)')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # 행렬 내부에 수치 표시
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(int(cm[i, j]), 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_evaluation(checkpoint_path, data_dir, batch_size=21):
    # 평가 시에도 동일한 데이터 분할을 보장하기 위해 시드 설정
    set_seed(42)

    from dataset.dataset_load import OxfordIIITPetsAugmented, tensor_trimap, args_to_dict, working_dir
    from models.model import ResNetUNet

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 3

    # 0. Output 폴더 생성 (src/output)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 모델 로드
    model = ResNetUNet(num_classes).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    model.eval()

    # 2. 테스트셋 로드 (train.py 방식 참고)
    transform_dict = args_to_dict(
        pre_transform=T.ToTensor(),
        pre_target_transform=T.ToTensor(),
        common_transform=None,
        post_transform=T.Compose([
            T.Resize((128, 128), interpolation=T.InterpolationMode.BILINEAR),
        ]),
        post_target_transform=T.Compose([
            T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST),
            T.Lambda(tensor_trimap),
        ]))

    # train.py와 동일하게 전체 데이터를 합친 후 분할
    pets_path_train = os.path.join(working_dir, 'OxfordPets', 'train')
    pets_path_test = os.path.join(working_dir, 'OxfordPets', 'test')

    ds_trainval = OxfordIIITPetsAugmented(root=pets_path_train, split="trainval", target_types="segmentation", download=False, **transform_dict)
    ds_test = OxfordIIITPetsAugmented(root=pets_path_test, split="test", target_types="segmentation", download=False, **transform_dict)
    all_data = ConcatDataset([ds_trainval, ds_test])

    total_len = len(all_data)
    test_len = int(0.2 * total_len)
    val_len = int(0.1 * total_len)
    train_len = total_len - (test_len + val_len)

    # manual_seed(42)를 사용하여 train.py에서 떼어놓은 것과 '정확히 동일한' 20% 테스트셋을 가져옴
    _, _, pets_test = random_split(all_data, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))

    test_loader = DataLoader(pets_test, batch_size=batch_size, shuffle=False)

    # 3. 메트릭 계산기 초기화
    # Oxford Pets 라벨 0, 1, 2를 모두 평가하기 위해 ignore_background=False 설정
    metric_calc = SegmentationMetrics(average=True, ignore_background=False, activation='0-1')

    total_metrics = np.zeros(5) # pixel_acc, dice, iou, precision, recall
    global_cm = np.zeros((num_classes, num_classes))
    samples = 0

    print("Evaluating...")
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device).squeeze(1).long() # (N, H, W)

            outputs = model(images) # (N, C, H, W)
            preds = torch.argmax(outputs, dim=1)
            
            results = metric_calc(masks, outputs)
            global_cm += calculate_confusion_matrix(masks, preds, num_classes)
            total_metrics += np.array(results) * images.size(0)
            samples += images.size(0)

    avg_metrics = total_metrics / samples
    
    print("\n--- Test Set Performance ---")
    print(f"Pixel Accuracy: {avg_metrics[0]:.4f}")
    print(f"Dice Score:     {avg_metrics[1]:.4f}")
    print(f"mIoU:           {avg_metrics[2]:.4f}")
    print(f"Precision:      {avg_metrics[3]:.4f}")
    print(f"Recall:         {avg_metrics[4]:.4f}")

    # Confusion Matrix 저장
    class_names = ['Pet', 'Background', 'Border']
    plot_confusion_matrix(global_cm, class_names, os.path.join(output_dir, "confusion_matrix.png"))
    print(f"\nConfusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Path to trained model")
    parser.add_argument("--data_dir", type=str, default="/home/sehoon/workspace/PetMask/src/dataset", help="Dataset root")
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.data_dir)