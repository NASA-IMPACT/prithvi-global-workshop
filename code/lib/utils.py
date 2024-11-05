import numpy as np
import rasterio
import torch

from consts import NO_DATA, NO_DATA_FLOAT

import numpy as np
import torch.nn as nn
import os

from PIL import Image


def load_raster(path, crop=None):
    """Load raster data using rasterio and crop if needed
    """
    with rasterio.open(path) as src:
        img = src.read()

        img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)

        # Crop image if crop size is provided
        if crop:
            img = img[:, -crop[0]:, -crop[1]:]
    return img


def preprocess_image(image,means,stds):
    """Calculate normalized values of the read image.

    """
    # Mean across height and width, for each channel
    reshaped_means = means.reshape(-1, 1, 1)
    # Std deviation across height and width, for each channel
    reshaped_stds = stds.reshape(-1, 1, 1)

    normalized = image.copy()
    # Normalize
    normalized = ((image - reshaped_means) / reshaped_stds)

    return torch.from_numpy(
        normalized.reshape(
            1,
            normalized.shape[0],
            1,
            *normalized.shape[-2:]
        )
    ).to(torch.float32)


################################## useful class and functions ##################################################

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


def plot_output_image(model, device, epoch, means, stds, input_path, prediction_img_dir):

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
    output_image_path = os.path.join(
        prediction_img_dir,
        f"segmentation_output_epoch_{epoch}.png"
    )
    img.save(output_image_path)


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
