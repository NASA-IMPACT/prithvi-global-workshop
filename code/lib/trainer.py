import csv
import numpy as np
import os
import torch
import yaml

from lib.consts import SPLITS
from lib.downstream_dataset import DownstreamDataset
from lib.model_utils import *
from lib.prithvi_wrapper import PrithviWrapper
from lib.utils import dice_loss

from prithvi_global.mae.config import get_config

from torch.optim import Adam
from torch.utils.data import DataLoader

from tqdm import tqdm


class Trainer:
    def __init__(self, config_filename, model_path=None, model_only=False):
        if not(os.path.exists(config_filename)):
            raise(f"Config file: {config_filename} not found")
        with open(config_filename) as config:
            self.config = yaml.safe_load(config)
        self.model_path = model_path
        self.load_parameters()
        if not(model_only):
            self.load_datasets()
        self.load_model()


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
                    self.config[split].get('annot'),
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
        self.criterion = nn.BCEWithLogitsLoss()


    def load_model(self):
       #initialize model
        model_weights = self.model_path or self.config["prithvi_model_new_weight"]
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

    def forward_pass(self, input, mask):
        input = input.to(device=self.device, dtype=torch.float32)
        mask = mask.to(device=self.device, dtype=torch.float32)

        self.optimizer.zero_grad()
        out = self.model(input)
        out = out.squeeze(1)
        loss = self.criterion(out, mask)
        loss += dice_loss(
            torch.sigmoid(out), mask,
            multiclass=False,
        )
        return loss, input, out, mask



    def train(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)


        for iteration in range(self.n_iteration):
            loss_i = 0.0

            self.model.train()
            current_iteration = 0
            total_iteration = len(self.dataloaders['training'])
            pbar = tqdm(
                self.dataloaders['training'],
                total=total_iteration,
                desc=f"Iteration [{current_iteration}/{total_iteration}]",
                bar_format="{l_bar}{bar} [ time left: {remaining}, time spent: {elapsed}]"
            )
            for input, mask in pbar:
                loss, input, out, mask = self.forward_pass(input, mask)
                loss_i += loss.item() * input.size(0)
                self.optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                grad_scaler.step(self.optimizer)
                grad_scaler.update()

            epoch_loss_train = loss_i / len(self.dataloaders['training'].dataset)
            epoch_loss_val = self.validate()

            print(f"Epoch {iteration} - Train Loss: {epoch_loss_train}, Validation Loss: {epoch_loss_val}")
            state_dict = self.model.state_dict()
            torch.save(state_dict, self.checkpoint)

    def validate(self):
        self.model.eval()

        val_loss = 0.0
        total_samples = len(self.dataloaders['validation'])
        current_sample = 0
        pbar = tqdm(
            self.dataloaders['validation'],
            total=total_samples,
            desc=f"Iteration [{current_sample}/{total_samples}]",
            bar_format="{l_bar}{bar} [ time left: {remaining}, time spent: {elapsed}]"
        )
        with torch.no_grad():
            for input, mask in pbar:
                self.model.eval()
                loss, input, out, mask = self.forward_pass(input, mask)
                val_loss += loss.item() * input.size(0)

        epoch_loss_val = val_loss / len(self.dataloaders['validation'].dataset)

        return epoch_loss_val
