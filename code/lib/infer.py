import torch
import yaml
import numpy as np
import rasterio
from lib.trainer import Trainer

class Infer:
    def __init__(self, config, checkpoint, backbone_path):
        self.config_filename = config
        with open(self.config_filename) as config:
            self.config = yaml.safe_load(config)
        self.checkpoint_filename = checkpoint
        self.backbone_path = backbone_path
        self.load_model()

    def load_model(self):
        trainer = Trainer(self.config_filename, model_path=self.backbone_path, model_only=True)
        self.model = trainer.model
        self.model.load_state_dict(
            torch.load(self.checkpoint_filename)['model_state_dict']
        )
        self.model.eval()

    def preprocess(self, images):
        images_array = []
        profiles = []
        for image in images:
            with rasterio.open(image) as raster_file:
                images_array.append(raster_file.read())
                profiles.append(raster_file.profile)
                raster_file.close()
        # Example processing function to simulate the pipeline
        imgs_tensor = torch.from_numpy(np.asarray(images_array))  # Assuming input_array is of type np.float32
        imgs_tensor = imgs_tensor.float()

        mean = torch.tensor(self.config['data']['means']).view(-1, 1, 1)
        std = torch.tensor(self.config['data']['stds']).view(-1, 1, 1)

        processed_images = (imgs_tensor - mean) / std

        # increase dimensions to match input size
        processed_images = processed_images.unsqueeze(2)
        print(processed_images.shape)
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
        return result, profiles

