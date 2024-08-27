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
    
    #print("annot file:",annot_file)

    for i in annot_file:
        #print("i is:",i)
        mask_name=i.split(",")[1].strip()
        mask_path=os.path.join(data_dir_mask,mask_name)
        #print("mask path:",mask_path)
        name=i.split(",")[0].strip()
        img_name="_".join(name.split("_")[:-1])+"_S2Hand.tif"
        img_path=os.path.join(data_dir_input,img_name)
        #print("img path:",img_path)
        path.append([img_path,mask_path])
        
    #print(path)

    return path


def segmentation_loss(mask,pred,device,class_weights,ignore_index):
    
    mask=mask.long()
    pred=pred.squeeze(2) #single timestamp for burn_scar mask, time dim=1 not required
    #print("mask ",mask)
    #print("pred",pred)
            
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)  # Define class weights
    # Initialize the CrossEntropyLoss with weights
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index,weight=class_weights).to(device) 
    loss=criterion(pred,mask) 
    #print("loss",loss)

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
    # Get the predicted class by taking the argmax along the class dimension
    # output shape: (batch_size, 2_class,time_frame,224, 224) -> predicted shape: (batch_size, 224, 224)
    output=output.squeeze(2)
    predicted = torch.argmax(output, dim=1)

    # Compare the predicted class with the true labels
    correct = (predicted == labels).sum().item()
    total = labels.numel()  # Total number of elements in labels
    
    # Compute accuracy
    accuracy = correct / total 
    #print("accuracy:",accuracy)
    return accuracy

def plot_output_image(model, device, epoch,means,stds,input_path,prediction_img_dir):
    model.eval()  # Set model to evaluation mode

    if_img=1
    img=load_raster(input_path,if_img,crop=(224, 224))
    #print("img size",img.shape)
    final_image=preprocess_image(img,means,stds)
    final_image=final_image.to(device)
    #print("final img size",final_image.shape)
    
    with torch.no_grad():
        output = model(final_image)  # output shape should be [1, n_segmentation_class, 224, 224]

    # Remove batch dimension
    output = output.squeeze(0).squeeze(1)  # shape [n_segmentation_class, 224, 224]

    # Convert output to predicted class
    predicted_mask = torch.argmax(output, dim=0)  # shape [224, 224]
    #print("predicted shape:",predicted_mask.shape)
    
    # Convert to numpy for plotting
    predicted_mask = predicted_mask.cpu().numpy()

    binary_image = (predicted_mask * 255).astype(np.uint8)

    # Convert to PIL image
    img = Image.fromarray(binary_image, mode='L')

    # Save the image
    output_image_path = os.path.join(prediction_img_dir,f"segmentation9_output_epoch_{epoch}.png")
    img.save(output_image_path)


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

    # Print all the configuration parameters
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

    means=config["data"]["means"]
    stds=config["data"]["stds"]
    means=np.array(means)
    stds=np.array(stds)


    #initialize dataset class
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
    #print("model:",pr_config)

    #n_frame=3
    #model_prithvi=prithvi(pr_weight,pr_config,n_frame)

    #initialize the model
    model=UNet(n_channel,n_class,n_frame,config["prithvi_model_new_weight"],config["prithvi_model_new_config"],input_size) #wrapper of prithvi #initialization of prithvi is done by initializing prithvi_loader.py
    model=model.to(device)

    #while training, prithvi core model weights should be freezed
    for param in model.prithvi.parameters():
        param.requires_grad = False

    # optimizer=optim.Adam(model.parameters(),lr=0.0001)
    #optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)


    optimizer_params = config["training"]['optimizer']['params']
    optimizer = getattr(optim, config["training"]['optimizer']['name'])(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_params)

    scheduler_params = config["training"]['scheduler']['params']
    scheduler = getattr(lr_scheduler, config["training"]['scheduler']['name'])(optimizer, **scheduler_params)

    best_loss=0

    for i in range(n_iteration):
        loss_i=0.0

        acc_dataset_train=[]
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

            loss.backward()
            optimizer.step()

            #print("loss",loss)

        acc_total_train=np.mean(acc_dataset_train)
        epoch_loss_train=(loss_i)/len(train_dataloader.dataset)
        wandb.log({"epoch": i + 1, "train_loss": epoch_loss_train,"acc_train":acc_total_train,"learning_rate": optimizer.param_groups[0]['lr']})
    

        # Validation Phase
        model.eval()
        val_loss = 0.0
        
        acc_dataset_val=[]
    
        with torch.no_grad():
            for j,(input,mask) in enumerate(val_dataloader):

                input=input.to(device)
                mask=mask.to(device)
            
                out=model(input)

                #loss=dice_loss(out, mask)
                loss=segmentation_loss(mask,out,device,class_weights,ignore_index)
                batch_acc=compute_accuracy(mask,out)

                acc_dataset_val.append(batch_acc)
                # Update validation loss and accuracy
                val_loss += loss.item() * input.size(0)   
    
        acc_total_val=np.mean(acc_dataset_val)
        epoch_loss_val = val_loss / len(val_dataloader.dataset)

        wandb.log({"epoch": i + 1, "val_loss": epoch_loss_val,"accuracy_val":acc_total_val})
        print(f"Epoch: {i}, train loss: {epoch_loss_train}, val loss:{epoch_loss_val},accuracy_train:{acc_total_train},accuracy_val:{acc_total_val}")

        scheduler.step()

        if i==0:
            best_loss=epoch_loss_val

        if epoch_loss_val<best_loss:
            save_checkpoint(model, optimizer, i, epoch_loss_train, epoch_loss_val, checkpoint)
            best_loss=epoch_loss_val

        if i%20==0:
            plot_output_image(model,device,i,means,stds,segment_input,predicted_mask_dir)

    wandb.finish()
    
if __name__ == "__main__":
    main()





