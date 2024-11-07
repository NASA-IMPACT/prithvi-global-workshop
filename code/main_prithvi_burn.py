import torch
from torch.utils.data import DataLoader
from lib.downstream_dataset import DownstreamDataset
from prithvi_wrapper import PrithviWrapper
import numpy as np
import yaml
import os
from prithvi_global.mae.config import get_config
from lib.utils import load_raster

from torch.optim import Adam
import csv

from lib.consts import SPLITS

from lib.model_utils import *


class Trainer:
    def __init__(self, config_filename):
        if not(os.path.exists(config_filename)):
            raise(f"Config file: {config_filename} not found")
        with open(config_filename) as config:
            self.config = yaml.safe_load(config)

        self.load_parameters()
        self.load_datasets()


    def load_datasets(self):
        data_dir = self.config["data"]["data_dir_input"]
        mask_dir = self.config["data"]["data_dir_mask"]
        self.datasets = {}
        self.dataloaders = {}

        for split in SPLITS:
            if self.config.get(split):
                dataset_dir = [data_dir, mask_dir]
                dataset_path = data_path(
                    dataset_dir,
                    None,
                    self.config,
                    self.config[split]['relative_path']
                )
                self.datasets[split] = DownstreamDataset(
                    dataset_path,
                    self.means,
                    self.stds,
                    self.config
                )
                self.dataloaders[split] = DataLoader(
                    self.datasets[split],
                    batch_size=self.batch_size[split],
                    shuffle=self.config[split]["shuffle"],
                    num_workers=1
                )

    def load_parameters(self):
        self.device = self.config["device_name"]
        self.n_channel = self.config["model"]["n_channel"]
        self.n_class = self.config["model"]["n_class"]
        self.n_frame = self.config["data"]["n_frame"]
        self.n_iteration = self.config["n_iteration"]

        self.case = self.config["case"]

        model_filename = f"{self.case}{self.config['dataset_used']}.pth"
        self.checkpoint = os.path.join(
            self.config["logging"]["checkpoint_dir"],
            model_filename
        )

        self.embed_size = self.config["model"]["encoder_embed_dim"]
        self.dec_embed_size = self.config["model"]["dec_embed_dim"]

        self.batch_size = {}

        for split in SPLITS:
            if self.config.get(split):
                self.batch_size[split] = self.config[split]["batch_size"]

        self.learning_rate = self.config["training"]["learning_rate"]
        self.segment_input = self.config["segment_input_path"]
        self.predicted_mask_dir = self.config["predicted_mask_dir"]
        self.class_weights = self.config["class_weights"]
        self.ignore_index = self.config["ignore_index"]
        self.input_size = self.config["data"]["input_size"]
        self.patch_size = self.config["data"]["patch_size"]

        self.used_data = self.config["dataset_used"]

        self.means = self.config["data"]["means"]
        self.stds = self.config["data"]["stds"]
        self.means = np.array(self.means)
        self.stds = np.array(self.stds)

    def load_model(self):
       #initialize model
        model_weights = self.config["prithvi_model_new_weight"]
        self.config["prithvi_model_new_config"] = get_config(None)
        prithvi_config = self.config["prithvi_model_new_config"]

        #wrapper of prithvi #initialization of prithvi is done by initializing prithvi_loader.py
        self.model = PrithviWrapper(
            self.n_channel,
            self.n_class,
            self.n_frame,
            self.embed_size,
            self.input_size,
            self.patch_size,
            model_weights,
            prithvi_config,
            self.n_channel,
            self.config
        )
        self.model = self.model.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.05)


    def train(self):
        best_miou_val = 0

        for i in range(self.n_iteration):
            loss_i = 0.0
            miou_train = []
            acc_dataset_train = []
            f1_dataset_train = []

            iou_train = torch.tensor([])
            iter = torch.tensor([i]).unsqueeze(0)
            print("iteration started")


            ####### train phase ################################################################
            self.model.train()

            for input, mask in self.dataloaders['training']:

                input = input.to(self.device)
                mask = mask.to(self.device)
                mask = mask.long()

                self.optimizer.zero_grad()
                out = self.model(input)
                loss = segmentation_loss(
                    mask,
                    out,
                    self.device,
                    self.class_weights,
                    self.ignore_index
                )

                loss_i += loss.item() * input.size(0)  # Multiply by batch size

                #batch_acc=compute_accuracy(mask,out)
                accuracy_batch, precision_batch, recall_batch, f1_batch = compute_accuracy_and_f1(mask,out)
                acc_dataset_train.append(accuracy_batch)
                f1_dataset_train.append(f1_batch)
                miou_batch, iou_batch = calculate_miou(out, mask, self.device)
                miou_train.append(miou_batch)
                iou_batch = iou_batch.cpu().unsqueeze(0)
                iou_batch = torch.cat((iter, iou_batch), dim=1)

                if i == 0:
                    iou_train = iou_batch
                else:
                    iou_train = torch.cat(
                        (iou_train,iou_batch),
                        dim=0
                    )

                loss.backward()
                self.optimizer.step()

            acc_dataset_train = np.array(acc_dataset_train) #B,n_class
            acc_total_train = np.mean(acc_dataset_train,axis=0) #1,n_class
            mean_acc_train = np.mean(acc_total_train) #1

            f1_dataset_train = np.array(f1_dataset_train) #B,n_class
            f1_total_train = np.mean(f1_dataset_train,axis=0) #1,n_class
            mean_f1_train = np.mean(f1_total_train) #1

            iou_train = iou_train.numpy()
            iou_train = np.mean(iou_train,axis=0)

            miou_train = np.mean(iou_train[1:])

            epoch_loss_train = loss_i / len(self.dataloaders['training'].dataset)
            log_postfix = f"{self.case}_{self.used_data}.csv"

            with open(f'iou_train_{log_postfix}.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                row=iou_train
                # Write the list as a row
                writer.writerow(row)

            with open(f'acc_train_{log_postfix}.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                row=acc_total_train
                # Write the list as a row
                writer.writerow(row)

            with open(f'f1_train_{log_postfix}.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                row=f1_total_train
                # Write the list as a row
                writer.writerow(row)

            print({
                "epoch": i + 1,
                "train_loss": epoch_loss_train,
                "acc_train": mean_acc_train,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "miou_train": miou_train,
                "mF1_train": mean_f1_train
            })
            miou_valid, mean_acc_val, mean_f1_val, epoch_loss_val = self.validate()

            if i == 0:
                best_miou_val = miou_valid

            print(f"""
                Epoch: {i}
                Train loss: {epoch_loss_train}
                Val loss: {epoch_loss_val}
                Accuracy train: {mean_acc_train}
                Accuracy val: {mean_acc_val}
                MIOU train: {miou_train}
                MIOU val: {miou_valid}
                MF1 train: {mean_f1_train}
                MF1_val: {mean_f1_val}
            """
            )

            if miou_valid > best_miou_val:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    i,
                    epoch_loss_train,
                    epoch_loss_val,
                    self.checkpoint
                )
                best_miou_val = miou_valid

            if i % (self.config['check_output'] or 2) == 0:
                plot_output_image(
                    self.model,
                    self.device,
                    i,
                    self.config,
                    self.segment_input,
                    self.predicted_mask_dir
                )

    def validate(self):
        self.model.eval()

        val_loss = 0.0
        miou_valid=[]
        acc_dataset_val=[]
        f1_dataset_val=[]

        iou_valid=torch.tensor([])

        with torch.no_grad():
            for index, (input, mask) in enumerate(self.dataloaders['validation']):

                input = input.to(self.device)
                mask = mask.to(self.device)
                mask = mask.long()

                out = self.model(input)
                loss = segmentation_loss(
                    mask,
                    out,
                    self.device,
                    self.class_weights,
                    self.ignore_index
                )

                val_loss += loss.item() * input.size(0)

                accuracy_batch, precision_batch, recall_batch, f1_batch = compute_accuracy_and_f1(
                    mask,
                    out
                )
                acc_dataset_val.append(accuracy_batch)
                f1_dataset_val.append(f1_batch)
                miou_batch, iou_batch = calculate_miou(out, mask, self.device)
                miou_valid.append(miou_batch)
                iou_batch = iou_batch.cpu().unsqueeze(0)
                iou_batch = torch.cat((iter, iou_batch), dim=1)

                if index == 0:
                    iou_valid = iou_batch
                else:
                    iou_valid=torch.cat((iou_valid, iou_batch), dim=0)


        acc_dataset_val = np.array(acc_dataset_val)
        acc_total_val = np.mean(acc_dataset_val,axis=0)
        mean_acc_val = np.mean(acc_total_val)
        #print("mean acc", mean_acc_val)

        f1_dataset_val = np.array(f1_dataset_val)
        f1_total_val = np.mean(f1_dataset_val,axis=0)
        mean_f1_val = np.mean(f1_total_val)
        #print("mean acc", mean_acc_val)

        #miou_valid=np.mean(miou_valid)
        iou_valid = iou_valid.numpy()
        iou_valid = np.mean(iou_valid,axis=0)
        miou_valid = np.mean(iou_valid[1:])

        epoch_loss_val = val_loss /len(self.dataloaders['validation'].dataset)

        log_postfix = f"{self.case}_{self.used_data}.csv"
        with open(f'acc_val_{log_postfix}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            row=acc_total_val
            # Write the list as a row
            writer.writerow(row)

        with open(f'f1_val_{log_postfix}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            row=f1_total_val
            # Write the list as a row
            writer.writerow(row)

        with open(f'iou_val_{log_postfix}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            row=iou_valid
            # Write the list as a row
            writer.writerow(row)

        return miou_valid, mean_acc_val, mean_f1_val, epoch_loss_val

def main():
    trainer = Trainer('../configs/burn_scars.yaml')
    trainer.train()

if __name__ == "__main__":
    main()
