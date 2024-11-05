import numpy as np
import torch
import yaml

from lib.downstream_dataset import DonwstreamDataset
from lib.model_utils import calculate_miou, compute_accuracy, plot_output_image, save_checkpoint, segmentation_loss
from lib.model_utils import data_path
from prithvi_global.mae.config import get_config
from prithvi_wrapper import PrithviWrapper
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


def main():

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device=config["device_name"]
    n_channel=config["model"]["n_channel"]
    n_class=config["model"]["n_class"]
    n_frame=config["data"]["n_frame"]
    n_iteration=config["n_iteration"]
    checkpoint=config["logging"]["checkpoint_dir"]
    embed_size=config["model"]["embed_dim"]
    dec_embed_size=config["model"]["dec_embed_dim"]
    data_dir=config["data"]["data_dir"]
    train_batch_size=config["training"]["train_batch_size"]
    val_batch_size=config["validation"]["val_batch_size"]
    learning_rate=config["training"]["learning_rate"]
    segment_input=config["segment_input_path"]
    predicted_mask_dir=config["predicted_mask_dir"]
    class_weights=config["class_weights"]
    ignore_index=config["ignore_index"]
    input_size=config["data"]["input_size"]
    patch_size=config["data"]["patch_size"]


    # Print all the configuration parameters
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {train_batch_size}")
    print(f"Number of Epochs: {n_iteration}")
    print(f"Number of Input Channel: {n_channel}")
    print(f"Number of Segmentation Class: {n_class}")
    print(f"Used device name: {device}")
    print(f"Checkpoint Path: {checkpoint}")
    print(f"Data input dir:{data_dir}")

    #initialize dataset class
    means=config["data"]["means"]
    stds=config["data"]["stds"]
    means=np.array(means)
    stds=np.array(stds)
    path_train=data_path("training",data_dir)
    path_val=data_path("validation",data_dir)
    flood_dataset_train=DonwstreamDataset(path_train,means,stds)
    flood_dataset_val=DonwstreamDataset(path_val,means,stds)


    #initialize dataloader
    train_dataloader=DataLoader(flood_dataset_train,batch_size=train_batch_size,shuffle=config["training"]["shuffle"],num_workers=1)
    val_dataloader=DataLoader(flood_dataset_val,batch_size=val_batch_size,shuffle=config["validation"]["shuffle"],num_workers=1)


    #initialize model
    config["prithvi_model_new_weight"]="/rhome/rghosal/Rinki/rinki-hls-foundation-os/Prithvi_global.pt"
    config["prithvi_model_new_config"]= get_config(None)
    model=PrithviWrapper(n_channel,n_class,n_frame,embed_size,input_size,patch_size,config["prithvi_model_new_weight"],config["prithvi_model_new_config"]) #wrapper of prithvi #initialization of prithvi is done by initializing prithvi_loader.py
    model=model.to(device)

    '''
    optimizer_params = config["training"]['optimizer']['params']
    optimizer = getattr(optim, config["training"]['optimizer']['name'])(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_params)

    scheduler_params = config["training"]['scheduler']['params']
    scheduler = getattr(lr_scheduler, config["training"]['scheduler']['name'])(optimizer, **scheduler_params)
    '''

    optimizer = AdamW(model.parameters(), lr=6e-5, weight_decay=0.05)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=True)

    best_loss=0
    best_miou_val=0

    for i in range(n_iteration):

        loss_i=0.0
        miou_train=[]
        acc_dataset_train=[]
        f1_dataset_train=[]

        print("iteration started")

        model.train()

        for j,(input,mask) in enumerate(train_dataloader):

            input=input.to(device)
            mask=mask.to(device)

            #print(torch.unique(mask)) #check unique classes in target mask #-1,0,1

            optimizer.zero_grad()
            out=model(input)

            loss=segmentation_loss(mask,out,device,class_weights,ignore_index)
            batch_acc,batch_f1=compute_accuracy(mask,out)

            loss_i += loss.item() * input.size(0)  # Multiply by batch size
            acc_dataset_train.append(batch_acc)
            f1_dataset_train.append(batch_f1)

            miou_batch=calculate_miou(out, mask, device)
            miou_train.append(miou_batch)

            loss.backward()
            optimizer.step()


        acc_total_train=np.mean(acc_dataset_train)
        f1_total_train=np.mean(f1_dataset_train)
        miou_train=np.mean(miou_train)
        epoch_loss_train=(loss_i)/len(train_dataloader.dataset)

        print({"epoch": i + 1, "train_loss": epoch_loss_train,"acc_train":acc_total_train,"learning_rate": optimizer.param_groups[0]['lr'],"miou_train":miou_train,"f1_train":f1_total_train})

        # Validation Phase
        model.eval()
        val_loss = 0.0
        miou_valid=[]
        acc_dataset_val=[]
        f1_dataset_val=[]

        with torch.no_grad():
            for j,(input,mask) in enumerate(val_dataloader):

                input=input.to(device)
                mask=mask.to(device)

                out=model(input)

                loss=segmentation_loss(mask,out,device,class_weights,ignore_index)
                batch_acc,batch_f1=compute_accuracy(mask,out)

                acc_dataset_val.append(batch_acc)
                f1_dataset_val.append(batch_f1)
                val_loss += loss.item() * input.size(0)

                miou_batch=calculate_miou(out, mask, device)
                miou_valid.append(miou_batch)

        acc_total_val=np.mean(acc_dataset_val)
        f1_total_val=np.mean(f1_dataset_val)
        epoch_loss_val = val_loss / len(val_dataloader.dataset)
        miou_valid=np.mean(miou_valid)

        print({"epoch": i + 1, "val_loss": epoch_loss_val,"accuracy_val":acc_total_val,"miou_val":miou_valid,"f1_val":f1_total_val})
        print(f"Epoch: {i}, train loss: {epoch_loss_train}, val loss:{epoch_loss_val},accuracy_train:{acc_total_train},accuracy_val:{acc_total_val},miou_train:{miou_train},miou_val:{miou_valid}")

        scheduler.step(epoch_loss_val)

        if i==0:
            best_loss=epoch_loss_val
            best_miou_val=miou_valid

        if miou_valid>best_miou_val:
            save_checkpoint(model, optimizer, i, epoch_loss_train, epoch_loss_val, checkpoint)
            best_miou_val=miou_valid

        if i%20==0:
            plot_output_image(model,device,i,means,stds,segment_input,predicted_mask_dir)

    # wandb.finish()

if __name__ == "__main__":
    main()
