import os
import torch
import torch.nn as nn
import numpy as np
import random

from glob import glob
from PIL import Image

from lib.utils import load_raster


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

def data_path(data_dir,annot,config,mode):

    path=[]

    if config["case"]=="burn":

        tif_path = os.path.join(f"{data_dir[0]}",f"{mode}/*tif")
        list_file = glob(tif_path)

        for i in list_file:
            tag=i.split("_")[-1]
            if tag=="merged.tif":
                j=i.strip("_merged.tif")
                mask=j+".mask.tif"
                if os.path.exists(mask):
                    path.append([i,mask])

    if mode=="training":
        path_len=len(path)
        used_pp=config["dataset_used"]
        ten_pp_len=int(used_pp*path_len)
        random.seed(42)
        random_selection_path = random.sample(path, ten_pp_len)
    else:
        random_selection_path = path

    return random_selection_path


def segmentation_loss(mask,pred,device,class_weights,ignore_index):

    mask=mask.long()

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index,weight=class_weights).to(device)
    loss=criterion(pred,mask)

    return loss

def compute_accuracy_and_f1(target, output):
    num_classes = output.shape[1]

    # (batch_size, num_classes, time_frame, 224, 224) ->  (batch_size, 224, 224)
    preds = torch.argmax(output, dim=1)

    # Flatten the tensors
    preds = preds.view(-1)
    target = target.view(-1)

    hit_batch = []
    total_batch = []
    precision_batch = []
    recall_batch = []
    f1_batch = []

    for cls in range(num_classes):
        Hit_output = (preds == cls).float()
        Hit_target = (target == cls).float()

        # True Positives (TP): correct predictions for this class
        TP = (Hit_output * Hit_target).sum().item()

        # Predicted Positives: total predicted for this class (True Positives + False Positives)
        Predicted_positives = Hit_output.sum().item()

        # Actual Positives: total actual for this class (True Positives + False Negatives)
        Actual_positives = Hit_target.sum().item()

        # Accuracy for the class
        accuracy_cls = (TP + 1e-6) / (Actual_positives + 1e-6)

        # Precision and recall for the class
        precision_cls = (TP + 1e-6) / (Predicted_positives + 1e-6)
        recall_cls = (TP + 1e-6) / (Actual_positives + 1e-6)

        # F1 Score for the class
        f1_cls = 2 * (precision_cls * recall_cls) / (precision_cls + recall_cls + 1e-6)

        # Append metrics
        hit_batch.append(accuracy_cls)
        precision_batch.append(precision_cls)
        recall_batch.append(recall_cls)
        f1_batch.append(f1_cls)

    # Convert to numpy arrays
    accuracy_batch = np.array(hit_batch)
    precision_batch = np.array(precision_batch)
    recall_batch = np.array(recall_batch)
    f1_batch = np.array(f1_batch)

    return accuracy_batch, precision_batch, recall_batch, f1_batch


def plot_output_image(model, device, epoch, config, input_path, prediction_img_dir):

    model.eval()

    if_img = 1
    img = load_raster(input_path, if_img, crop=None)

    #normalize image
    mean = np.array(config["data"]["means"]).reshape(-1, 1, 1)  # Reshape to (6, 1, 1)
    std = np.array(config["data"]["stds"]).reshape(-1, 1, 1)    # Reshape to (6, 1, 1)

    final_image = (img - mean)/ std

    if config["case"]=="burn":
        #centre crop
        start = (512 - 224) // 2
        end = start + 224
        final_image=final_image[:, start:end, start:end]

    final_image=torch.from_numpy(final_image).float()

    final_image=final_image.to(device)
    final_image=final_image.unsqueeze(0).unsqueeze(2)
    #print("to be plotted input shape",final_image.shape)

    with torch.no_grad():
        output = model(final_image)  # [1, n_segmentation_class, 224, 224]

    # Remove batch dimension
    output = output.squeeze(0)  # [n_segmentation_class, 224, 224]

    predicted_mask = torch.argmax(output, dim=0)  # shape [224, 224]
    predicted_mask = predicted_mask.cpu().numpy()

    colormap = np.array(config['colormap'], dtype=np.uint8)

    # Apply the color map to the predicted mask
    colored_mask = colormap[predicted_mask]

    #binary_image = (predicted_mask * 255).astype(np.uint8)
    img = Image.fromarray(colored_mask) #PIL Image

    # Save the image
    output_image_path = os.path.join(prediction_img_dir,f"segmentation_output_epoch_{epoch}.png")
    img.save(output_image_path)


def calculate_miou(output, target, device):

    eps=1e-6
    #output.shape = B,n_classes,H,W
    num_classes=output.shape[1]
    preds = torch.argmax(output, dim=1)

    # Flatten the tensors
    preds = preds.view(-1)
    target = target.view(-1)

    # Initialize intersection and union for each class
    intersection = torch.zeros(num_classes).to(device)
    union = torch.zeros(num_classes).to(device)
    iou_cls = torch.zeros(num_classes).to(device)

    for cls in range(num_classes): #starts from 0

        # Create binary masks for the current class
        pred_mask = (preds == cls).float()
        target_mask = (target == cls).float()

        # Calculate intersection and union
        intersection[cls] = (pred_mask * target_mask).sum()
        union[cls] = pred_mask.sum() + target_mask.sum() - intersection[cls]
        iou_cls[cls] = intersection[cls] / (union[cls] + eps)

    # Calculate IoU for each class for all images in one batch
    iou = intersection / (union + eps)  # Add eps to avoid division by zero

    # Calculate mean IoU (all class avergae)
    mean_iou = iou.mean().item()

    return mean_iou,iou_cls


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
