import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from lib.downstream_dataset import DownstreamDataset
from prithvi_wrapper import PrithviWrapper
import numpy as np
import yaml
import torch.nn as nn
import glob
import os
import wandb
import argparse
from Prithvi_global_v1.mae.config import get_config
from PIL import Image
from data_load_prithvi_burn import load_raster
#from data_load_prithvi_crop import preprocess_image
import random
from torch.optim import Adam
import csv



################################## useful class and functions ##################################################
def data_path(data_dir,annot,config,mode):

    path=[]

    if config["case"]=="burn":

        tif_path=os.path.join(f"{data_dir[0]}",f"{mode}/*tif")
        list_file=glob.glob(tif_path)

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

def plot_output_image(model, device, epoch,config,input_path,prediction_img_dir):

    model.eval()

    if_img=1
    img=load_raster(input_path,if_img,crop=None)

    #normalize image
    mean = np.array(config["data"]["means"]).reshape(-1, 1, 1)  # Reshape to (6, 1, 1)
    std = np.array(config["data"]["stds"]).reshape(-1, 1, 1)    # Reshape to (6, 1, 1)

    final_image = (img - mean)/ std

    if config["case"]=="burn":
        #centre crop
        start = (512 - 224) // 2
        end = start + 224
        final_image=final_image[:,start:end,start:end]

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

    if config["case"]=="burn" :

        colormap = np.array([
            [0, 0, 0],# Class 0
            [255, 255, 255], # Class 1
        ], dtype=np.uint8)

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


#######################################################################################

def main():

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device=config["device_name"]
    n_channel=config["model"]["n_channel"]
    n_class=config["model"]["n_class"]
    n_frame=config["data"]["n_frame"]
    n_iteration=config["n_iteration"]

    file=config["case"]+f'{config["dataset_used"]}'+ ".pth"
    checkpoint=os.path.join(config["logging"]["checkpoint_dir"],file)


    embed_size=config["model"]["encoder_embed_dim"]
    dec_embed_size=config["model"]["dec_embed_dim"]

    train_batch_size=config["training"]["train_batch_size"]
    val_batch_size=config["validation"]["val_batch_size"]
    learning_rate=config["training"]["learning_rate"]
    segment_input=config["segment_input_path"]
    predicted_mask_dir=config["predicted_mask_dir"]
    class_weights=config["class_weights"]
    ignore_index=config["ignore_index"]
    input_size=config["data"]["input_size"]
    patch_size=config["data"]["patch_size"]

    case=config["case"]
    used_data=config["dataset_used"]


    wandb.init(
            # set the wandb project where this run will be logged
            project=config["wandb_project"]
            )

    wandb.run.log_code(".")


    #initialize dataset class
    means=config["data"]["means"]
    stds=config["data"]["stds"]
    means=np.array(means)
    stds=np.array(stds)


    if config["case"]=="burn":
        data_dir=[config["data"]["data_dir_input"],config["data"]["data_dir_mask"]]
        path_train=data_path(data_dir,None,config,"training")
        path_val=data_path(data_dir,None,config,"validation")

    burn_dataset_train=burn_dataset(path_train,means,stds,config)
    burn_dataset_val=burn_dataset(path_val,means,stds,config)

    #initialize dataloader
    train_dataloader=DataLoader(burn_dataset_train,batch_size=train_batch_size,
                                shuffle=config["training"]["shuffle"],num_workers=1)
    val_dataloader=DataLoader(burn_dataset_val,batch_size=val_batch_size,
                              shuffle=config["validation"]["shuffle"],num_workers=1)

    #initialize model
    model_weights=config["prithvi_model_new_weight"]
    config["prithvi_model_new_config"]= get_config(None)
    prithvi_config=config["prithvi_model_new_config"]

    #wrapper of prithvi #initialization of prithvi is done by initializing prithvi_loader.py
    model=prithvi_wrapper(n_channel,n_class,n_frame,embed_size,input_size,
                          patch_size,model_weights,prithvi_config,
                          n_channel,config)
    model=model.to(device)


    optimizer = Adam(model.parameters(),lr=1e-5,betas=(0.9, 0.999), weight_decay=0.05)

    best_miou_val=0

    for i in range(n_iteration):

        loss_i=0.0
        miou_train=[]
        acc_dataset_train=[]
        f1_dataset_train=[]

        iou_train=torch.tensor([])
        iter=torch.tensor([i]).unsqueeze(0)
        print("iteration started")


        ####### train phase   ################################################################
        model.train()

        for j,(input,mask) in enumerate(train_dataloader):

            input=input.to(device)
            mask=mask.to(device)
            mask=mask.long()

            optimizer.zero_grad()
            out=model(input)
            loss=segmentation_loss(mask,out,device,class_weights,ignore_index)
            loss_i += loss.item() * input.size(0)  # Multiply by batch size

            #batch_acc=compute_accuracy(mask,out)
            accuracy_batch, precision_batch, recall_batch, f1_batch=compute_accuracy_and_f1(mask,out)
            acc_dataset_train.append(accuracy_batch)
            f1_dataset_train.append(f1_batch)
            miou_batch,iou_batch=calculate_miou(out, mask, device)
            miou_train.append(miou_batch)
            iou_batch=iou_batch.cpu().unsqueeze(0)
            iou_batch = torch.cat((iter, iou_batch), dim=1)

            if i==0:
                iou_train=iou_batch
            else:
                iou_train=torch.cat((iou_train,iou_batch),dim=0)


            loss.backward()

            #Gradient clipping by norm
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Gradient clipping by value (optional, use one at a time)
            #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)


            optimizer.step()

        acc_dataset_train=np.array(acc_dataset_train) #B,n_class
        acc_total_train=np.mean(acc_dataset_train,axis=0) #1,n_class
        mean_acc_train=np.mean(acc_total_train) #1
        #print("mean acc", mean_acc)

        f1_dataset_train=np.array(f1_dataset_train) #B,n_class
        f1_total_train=np.mean(f1_dataset_train,axis=0) #1,n_class
        mean_f1_train=np.mean(f1_total_train) #1
        #print("mean acc", mean_acc)

        #miou_train=np.mean(miou_train)
        iou_train = iou_train.numpy()
        iou_train=np.mean(iou_train,axis=0)
        print("iou_train",iou_train)
        miou_train=np.mean(iou_train[1:])
        print("miou train",miou_train)

        epoch_loss_train=(loss_i)/len(train_dataloader.dataset)

        with open(f'iou_train_{case}_{used_data}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            row=iou_train
            # Write the list as a row
            writer.writerow(row)

        with open(f'acc_train_{case}_{used_data}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            row=acc_total_train
            # Write the list as a row
            writer.writerow(row)

        with open(f'f1_train_{case}_{used_data}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            row=f1_total_train
            # Write the list as a row
            writer.writerow(row)


        wandb.log({"epoch": i + 1, "train_loss": epoch_loss_train,"acc_train":mean_acc_train,
                   "learning_rate": optimizer.param_groups[0]['lr'],"miou_train":miou_train,"mF1_train":mean_f1_train})

        ########### Validation Phase ############################################################
        model.eval()

        val_loss = 0.0
        miou_valid=[]
        acc_dataset_val=[]
        f1_dataset_val=[]

        iou_valid=torch.tensor([])

        with torch.no_grad():
            for j,(input,mask) in enumerate(val_dataloader):

                input=input.to(device)
                mask=mask.to(device)
                mask=mask.long()

                out=model(input)
                loss=segmentation_loss(mask,out,device,class_weights,ignore_index)
                val_loss += loss.item() * input.size(0)

                #batch_acc=compute_accuracy(mask,out)
                accuracy_batch, precision_batch, recall_batch, f1_batch=compute_accuracy_and_f1(mask,out)
                acc_dataset_val.append(accuracy_batch)
                f1_dataset_val.append(f1_batch)
                miou_batch,iou_batch=calculate_miou(out, mask, device)
                miou_valid.append(miou_batch)
                iou_batch=iou_batch.cpu().unsqueeze(0)
                iou_batch = torch.cat((iter, iou_batch), dim=1)

                if i==0:
                    iou_valid=iou_batch
                else:
                    iou_valid=torch.cat((iou_valid,iou_batch),dim=0)


        acc_dataset_val=np.array(acc_dataset_val)
        acc_total_val=np.mean(acc_dataset_val,axis=0)
        mean_acc_val=np.mean(acc_total_val)
        #print("mean acc", mean_acc_val)

        f1_dataset_val=np.array(f1_dataset_val)
        f1_total_val=np.mean(f1_dataset_val,axis=0)
        mean_f1_val=np.mean(f1_total_val)
        #print("mean acc", mean_acc_val)

        #miou_valid=np.mean(miou_valid)
        iou_valid = iou_valid.numpy()
        iou_valid=np.mean(iou_valid,axis=0)
        miou_valid=np.mean(iou_valid[1:])

        epoch_loss_val=(val_loss)/len(val_dataloader.dataset)


        with open(f'acc_val_{case}_{used_data}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            row=acc_total_val
            # Write the list as a row
            writer.writerow(row)

        with open(f'f1_val_{case}_{used_data}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            row=f1_total_val
            # Write the list as a row
            writer.writerow(row)

        with open(f'iou_val_{case}_{used_data}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            row=iou_valid
            # Write the list as a row
            writer.writerow(row)

        wandb.log({"epoch": i + 1, "val_loss": epoch_loss_val,"accuracy_val":mean_acc_val,
                   "miou_val":miou_valid, "mF1_val":mean_f1_val})


        print(f"Epoch: {i}, train loss: {epoch_loss_train}, val loss:{epoch_loss_val},accuracy_train:{mean_acc_train},accuracy_val:{mean_acc_val},miou_train:{miou_train},miou_val:{miou_valid},mF1_train:{mean_f1_train},mF1_val: {mean_f1_val}")


        if i==0:

            best_miou_val=miou_valid

        if miou_valid>best_miou_val:
            save_checkpoint(model, optimizer, i, epoch_loss_train, epoch_loss_val, checkpoint)
            best_miou_val=miou_valid

        if i%2==0:
            plot_output_image(model,device,i,config,segment_input,predicted_mask_dir)

    wandb.finish()

if __name__ == "__main__":
    main()
