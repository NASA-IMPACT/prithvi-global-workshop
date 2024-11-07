import numpy as np
import numpy as np
import os
import random
import rasterio
import torch
import torch.nn as nn

from consts import NO_DATA, NO_DATA_FLOAT
from PIL import Image


def load_raster(path, crop=None):
        with rasterio.open(path) as src:
            img = src.read()

            img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)

        return img

def random_crop(tensor, crop_size=(224, 224)):
    # Get original dimensions: channel, height (H), width (W)
    C, H, W = tensor["img"].shape

    # Ensure the crop size fits within the original dimensions
    crop_h, crop_w = crop_size
    if H < crop_h or W < crop_w:
        raise ValueError(f"Original size ({H}, {W}) is smaller than the crop size ({crop_h}, {crop_w})")

    # Randomly select the top-left corner for the crop
    top = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)

    # Perform the crop (channel dimension remains unchanged)
    tensor["img"] = tensor["img"][:, top:top + crop_h, left:left + crop_w]
    tensor["mask"] = tensor["mask"][:,top:top + crop_h, left:left + crop_w]

    return tensor



# Example processing function to simulate the pipeline
def process_input(input_array, mask, img_norm_cfg):

    input_array=input_array.astype(np.float32)
    img_tensor = torch.from_numpy(input_array)  # Assuming input_array is of type np.float32
    img_tensor=img_tensor.float()
    mask_tensor = torch.from_numpy(mask)

    processed_data = {}
    processed_data['img'] = img_tensor
    processed_data['mask'] = mask_tensor
    # step['type'] == "RandomFlip":
    p=np.random.rand()
    if p < 0.5:
        processed_data['img'] = torch.flip(img_tensor, [2])  # Flip along width
        processed_data['mask'] = torch.flip(mask_tensor, [2])  # Flip along width

    #print("flipped img shape",processed_data['img'].shape)
    #print("flipped mask shape",processed_data['mask'].shape)

    mean=torch.tensor(img_norm_cfg['mean']).view(-1, 1, 1)
    std=torch.tensor(img_norm_cfg['std']).view(-1, 1, 1)

    #print("mean shape",mean.shape)
    #print("std shape",std.shape)

    # step['type'] == "TorchNormalize":
    processed_data['img'] = (processed_data['img'] - mean)/ std
    #print("normalized img shape",processed_data['img'].shape)

    # step['type'] == "TorchRandomCrop":
    processed_data = random_crop(processed_data, (224, 224))
    #print("cropped img shape",processed_data['img'].shape)
    #print("cropped mask shape",processed_data['mask'].shape)

    return processed_data['img'],processed_data['mask']


def segmentation_loss(mask, pred, device, class_weights, ignore_index):

    mask=mask.long()

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)  # Define class weights

    # Initialize the CrossEntropyLoss with weights
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index,weight=class_weights).to(device)
    loss=criterion(pred,mask)

    return loss


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, filename):

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss':val_loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")


def compute_accuracy(labels, output):
    # TODO: Change to scikit or some other package
    # (batch_size, 2_class,time_frame,224, 224) ->  (batch_size, 224, 224)
    predicted = torch.argmax(output, dim=1)

    # Compare the predicted class with the true labels
    # correct = (predicted == labels).sum().item()
    total = labels.numel()  # Total number of elements in labels

    # accuracy = correct / total

    # True Positives (TP): Both predicted and true labels are 1
    TP = ((predicted == 1) & (labels == 1)).sum().item()

    # True Negatives (TN): Both predicted and true labels are 0
    TN = ((predicted == 0) & (labels == 0)).sum().item()

    # False Positives (FP): Predicted is 1, but true label is 0
    FP = ((predicted == 1) & (labels == 0)).sum().item()

    # False Negatives (FN): Predicted is 0, but true label is 1
    FN = ((predicted == 0) & (labels == 1)).sum().item()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return (TP + TN) / total, f1_score


def calculate_miou(output, target, device):

    eps=1e-6

    num_classes=output.shape[1]
    preds = torch.argmax(output, dim=1)

    # Flatten the tensors
    preds = preds.view(-1)  # Flatten predictions
    target = target.view(-1)  # Flatten target

    # Initialize intersection and union for each class
    intersection = torch.zeros(num_classes).to(device)
    union = torch.zeros(num_classes).to(device)

    for cls in range(num_classes):

        # Create binary masks for the current class
        pred_mask = (preds == cls).float()
        target_mask = (target == cls).float()

        # Calculate intersection and union
        intersection[cls] = (pred_mask * target_mask).sum()
        union[cls] = pred_mask.sum() + target_mask.sum() - intersection[cls]

    # Calculate IoU for each class for all images in one batch

    iou = intersection / (union + eps)  # Add eps to avoid division by zero

    # Calculate mean IoU (all class avergae)
    mean_iou = iou.mean().item()

    return mean_iou


def print_model_details(model):
    for name, module in model.named_modules():
        #print(f"Layer Name: {name}")
        #print(f"Layer Type: {module.__class__.__name__}")
        print("name",name)
