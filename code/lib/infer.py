import torch
import yaml
import numpy as np
import rasterio
from lib.trainer import Trainer
from lib.consts import NO_DATA, NO_DATA_FLOAT

class Infer:
    def __init__(self, config, checkpoint, backbone_path):
        self.config_filename = config
        with open(self.config_filename) as config:
            self.config = yaml.safe_load(config)
        self.checkpoint_filename = checkpoint
        self.backbone_path = backbone_path
        self.load_model()

    def load_model(self):
        model_weights = torch.load(self.checkpoint_filename)
        model_weights["model"] = model_weights.pop("model_state_dict")
        if 'prithvi_backbone.pos_embed' in model_weights['model']:
            del model_weights['model']['prithvi_backbone.pos_embed']

        trainer = Trainer(self.config_filename, model_path=model_weights, model_only=True)
        self.model = trainer.model
        self.model = self.model.to(self.config['device_name'])

        self.model.load_state_dict(model_weights["model"], strict=False)
        self.model.eval()

    def preprocess(self, images):
        images_array = []
        profiles = []

        mean = torch.tensor(self.config['data']['means']).view(-1, 1, 1)
        std = torch.tensor(self.config['data']['stds']).view(-1, 1, 1)

        for image in images:
            with rasterio.open(image) as raster_file:
                image = torch.from_numpy(raster_file.read())
                image = np.where(image == NO_DATA, NO_DATA_FLOAT, image)
                image = (image - mean) / std
                images_array.append(image)
                profiles.append(raster_file.profile)
                raster_file.close()
        # Example processing function to simulate the pipeline
        imgs_tensor = torch.from_numpy(np.asarray(images_array))  # Assuming input_array is of type np.float32
        imgs_tensor = imgs_tensor.float()

        # increase dimensions to match input size
        processed_images = imgs_tensor.unsqueeze(2)
        return processed_images, profiles

    def infer(self, images):
        """
        Infer on provided images
        Args:
            images (list): List of images
        """
        # forward the model
        with torch.no_grad():
            images, profiles = self.preprocess(images)
            result = self.model(images.to(self.config['device_name']))
            result = torch.argmax(result, dim=1)
        return result, profiles
