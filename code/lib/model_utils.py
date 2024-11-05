import os
import torch
import torch.nn as nn
import numpy as np

from glob import glob
from PIL import Image

from lib.utils import load_raster, preprocess_image


def upscaling_block(in_channels, out_channels):
    block=nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    return block

def upscaling_block2(in_channels, out_channels):

    block=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    return block

def data_path(mode,data_dir):
    tif_path=os.path.join(f"{data_dir}",f"{mode}/*tif")
    list_file=glob.glob(tif_path)
    path=[]

    for i in list_file:
        tag=i.split("_")[-1]
        if tag=="merged.tif":
            j=i.strip("_merged.tif")
            mask=j+".mask.tif"
            if os.path.exists(mask):
                path.append([i,mask])
    return path


def segmentation_loss(mask,pred,device,class_weights,ignore_index):

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
    # (batch_size, 2_class,time_frame,224, 224) ->  (batch_size, 224, 224)
    predicted = torch.argmax(output, dim=1)

    # Compare the predicted class with the true labels
    correct = (predicted == labels).sum().item()
    total = labels.numel()  # Total number of elements in labels

    accuracy = correct / total

    # True Positives (TP): Both predicted and true labels are 1
    TP = ((predicted == 1) & (labels == 1)).sum().item()

    # True Negatives (TN): Both predicted and true labels are 0
    TN = ((predicted == 0) & (labels == 0)).sum().item()

    # False Positives (FP): Predicted is 1, but true label is 0
    FP = ((predicted == 1) & (labels == 0)).sum().item()

    # False Negatives (FN): Predicted is 0, but true label is 1
    FN = ((predicted == 0) & (labels == 1)).sum().item()

    # Precision: TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    # Recall: TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    #miou_1=TP/(TP+FN+FP)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


    return accuracy,f1_score


def plot_output_image(model, device, epoch,means,stds,input_path,prediction_img_dir):

    model.eval()

    if_img=1
    img=load_raster(input_path,if_img,crop=(224, 224))

    final_image=preprocess_image(img,means,stds)
    final_image=final_image.to(device)


    with torch.no_grad():
        output = model(final_image)  # [1, n_segmentation_class, 224, 224]

    # Remove batch dimension
    output = output.squeeze(0)  # [n_segmentation_class, 224, 224]

    predicted_mask = torch.argmax(output, dim=0)  # shape [224, 224]
    predicted_mask = predicted_mask.cpu().numpy()
    binary_image = (predicted_mask * 255).astype(np.uint8)
    img = Image.fromarray(binary_image, mode='L') #PIL Image

    # Save the image
    output_image_path = os.path.join(prediction_img_dir,f"segmentation_output_epoch_{epoch}.png")
    img.save(output_image_path)


def calculate_miou(output, target, device):

    eps = 1e-6

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
