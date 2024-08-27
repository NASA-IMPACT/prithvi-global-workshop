import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data_load_flood_new import burn_dataset
from unet import UNet
import numpy as np
import yaml
import torch.nn as nn
import glob
import os
import wandb
import argparse
from Prithvi_global_v1.mae.config import get_config
from PIL import Image
from data_load_flood_new import load_raster
from data_load_flood_new import preprocess_image

################################## useful class and functions ##################################################
def data_path(mode,data_dir_input,data_dir_mask,annot):
    
    path=[]
    
    if mode=="training":
        with open (annot,"r") as f:
            annot_file=f.readlines()
    else:
        with open (annot,"r") as f:
            annot_file=f.readlines()
    

    for i in annot_file:
        
        mask_name=i.split(",")[1].strip()
        mask_path=os.path.join(data_dir_mask,mask_name)
        
        name=i.split(",")[0].strip()
        img_name="_".join(name.split("_")[:-1])+"_S2Hand.tif"
        img_path=os.path.join(data_dir_input,img_name)
       
        path.append([img_path,mask_path])
        

    return path


def segmentation_loss(mask,pred,device,class_weights,ignore_index):
    
    mask=mask.long()
    pred=pred.squeeze(2) #single timestamp for burn_scar mask, time dim=1 not required
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)  # Data is imbalanced
    
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
    
    # output: (batch_size, 2_class,time_frame,224, 224) -> (batch_size, 224, 224)
    output=output.squeeze(2)
    predicted = torch.argmax(output, dim=1)

    # Compare the predicted class with the true labels
    correct = (predicted == labels).sum().item()
    total = labels.numel()  # Total number of elements in labels
    
    accuracy = correct / total 
    
    return accuracy


def plot_output_image(model, device, epoch,means,stds,input_path,prediction_img_dir):
    
    model.eval()  

    if_img=1
    img=load_raster(input_path,if_img,crop=(224, 224))
   
    final_image=preprocess_image(img,means,stds)
    final_image=final_image.to(device)
    
    with torch.no_grad():
        output = model(final_image)  # [1, n_segmentation_class, 224, 224]

    # Remove batch dimension, batch=1, and then the time_step dimension, which is 0 also

    output = output.squeeze(0).squeeze(1)  #[n_segmentation_class, 224, 224]  
    predicted_mask = torch.argmax(output, dim=0)  # shape [224, 224]
    predicted_mask = predicted_mask.cpu().numpy()
    binary_image = (predicted_mask * 255).astype(np.uint8)
    img = Image.fromarray(binary_image, mode='L') #PIL Image

    # Save the image
    output_image_path = os.path.join(prediction_img_dir,f"segmentation9_output_epoch_{epoch}.png")
    img.save(output_image_path)


def calculate_miou(output, target, device):


    eps=1e-6
    output=output.squeeze(2)
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

    # Calculate IoU for each class
    iou = intersection / (union + eps)  # Add eps to avoid division by zero

    # Calculate mean IoU
    mean_iou = iou.mean().item()

    return mean_iou

#######################################################################################

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device=config["device_name"]
    n_channel=config["model"]["n_channel"]
    n_class=config["model"]["n_class"]
    n_frame=config["data"]["n_frame"]
    n_iteration=config["n_iteration"]
    checkpoint=config["logging"]["checkpoint_dir"]
    data_dir_input=config["data"]["data_dir_input"]   
    data_dir_mask=config["data"]["data_dir_mask"]   
    train_annot=config["data"]["train_annot"]    
    val_annot=config["data"]["val_annot"]      
    train_batch_size=config["training"]["train_batch_size"]
    val_batch_size=config["validation"]["val_batch_size"]
    learning_rate=config["training"]["learning_rate"]
    segment_input=config["segment_input_path"]
    predicted_mask_dir=config["predicted_mask_dir"]
    class_weights=config["class_weights"]
    ignore_index=config["ignore_index"]
    input_size=config["data"]["input_size"]

    # Print some of the configuration parameters
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {train_batch_size}")
    print(f"Number of Epochs: {n_iteration}")
    print(f"Number of Input Channel: {n_channel}")
    print(f"Number of Segmentation Class: {n_class}")
    print(f"Used device name: {device}")
    print(f"Checkpoint Path: {checkpoint}")
    print(f"Data input dir:{data_dir_input}")
    print(f"Data mask dir:{data_dir_mask}")
    print(f"Train annotation file:{train_annot}")
    print(f"Val annotation file:{val_annot}")
    

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
    path_train=data_path("training",data_dir_input,data_dir_mask,train_annot)
    path_val=data_path("validation",data_dir_input,data_dir_mask,val_annot)
    flood_dataset_train=burn_dataset(path_train,means,stds)
    flood_dataset_val=burn_dataset(path_val,means,stds)

    #initialize dataloader
    train_dataloader=DataLoader(flood_dataset_train,batch_size=train_batch_size,shuffle=config["training"]["shuffle"],num_workers=1)
    val_dataloader=DataLoader(flood_dataset_val,batch_size=val_batch_size,shuffle=config["validation"]["shuffle"],num_workers=1)

    #initialize model    
    config["prithvi_model_new_weight"]="/rhome/rghosal/Rinki/rinki-hls-foundation-os/Prithvi_global.pt"
    config["prithvi_model_new_config"]= get_config(None)  
    model=UNet(n_channel,n_class,n_frame,config["prithvi_model_new_weight"],config["prithvi_model_new_config"],input_size) #wrapper of prithvi #initialization of prithvi is done by initializing prithvi_loader.py
    model=model.to(device)

    #while training, prithvi core model weights should be freezed
    for param in model.prithvi.parameters():
        param.requires_grad = False

    optimizer_params = config["training"]['optimizer']['params']
    optimizer = getattr(optim, config["training"]['optimizer']['name'])(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_params)

    scheduler_params = config["training"]['scheduler']['params']
    scheduler = getattr(lr_scheduler, config["training"]['scheduler']['name'])(optimizer, **scheduler_params)

    best_loss=0
    best_miou_val=0

    for i in range(n_iteration):

        loss_i=0.0
        acc_dataset_train=[]
        miou_train=[]
        print("iteration started")

        model.train()

        for j,(input,mask) in enumerate(train_dataloader):

            input=input.to(device)
            mask=mask.to(device)

            #print(torch.unique(mask)) #check unique classes in target mask #-1,0,1
        
            optimizer.zero_grad()
            out=model(input)

            loss=segmentation_loss(mask,out,device,class_weights,ignore_index)
            batch_acc=compute_accuracy(mask,out)
            
            loss_i += loss.item() * input.size(0)  # Multiply by batch size
            acc_dataset_train.append(batch_acc)

            miou_batch=calculate_miou(out, mask, device)
            miou_train.append(miou_batch)
    
            loss.backward()
            optimizer.step()

        acc_total_train=np.mean(acc_dataset_train)
        miou_train=np.mean(miou_train)
        epoch_loss_train=(loss_i)/len(train_dataloader.dataset)
        wandb.log({"epoch": i + 1, "train_loss": epoch_loss_train,"acc_train":acc_total_train,"learning_rate": optimizer.param_groups[0]['lr'],"miou_train":miou_train})

        # Validation Phase
        model.eval()
        val_loss = 0.0
        miou_valid=[]
        acc_dataset_val=[]
    
        with torch.no_grad():
            for j,(input,mask) in enumerate(val_dataloader):

                input=input.to(device)
                mask=mask.to(device)
            
                out=model(input)

                #loss
                loss=segmentation_loss(mask,out,device,class_weights,ignore_index)
                batch_acc=compute_accuracy(mask,out)

                acc_dataset_val.append(batch_acc)
                val_loss += loss.item() * input.size(0)  

                miou_batch=calculate_miou(out, mask, device)
                miou_valid.append(miou_batch) 
    
        acc_total_val=np.mean(acc_dataset_val)
        epoch_loss_val = val_loss / len(val_dataloader.dataset)
        miou_valid=np.mean(miou_valid)

        wandb.log({"epoch": i + 1, "val_loss": epoch_loss_val,"accuracy_val":acc_total_val,"miou_val":miou_valid})
        print(f"Epoch: {i}, train loss: {epoch_loss_train}, val loss:{epoch_loss_val},accuracy_train:{acc_total_train},accuracy_val:{acc_total_val},miou_val:{miou_valid}")


        scheduler.step()

        if i==0:
            best_loss=epoch_loss_val
            best_miou_val=miou_valid

        if miou_valid>best_miou_val:
            save_checkpoint(model, optimizer, i, epoch_loss_train, epoch_loss_val, checkpoint)
            best_miou_val=miou_valid

        if i%20==0:
            plot_output_image(model,device,i,means,stds,segment_input,predicted_mask_dir)

    wandb.finish()
    
if __name__ == "__main__":
    main()





