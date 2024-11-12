import torch
import yaml
import numpy as np
import rasterio

class Infer:
    def __init__(self, config, checkpoint, backbone_path):
        self.config_filename = config
        with open(self.config_filename) as config:
            self.config = yaml.safe_load(config)
        self.checkpoint_filename = checkpoint
        self.backbone_path = backbone_path
        self.load_model()

    def load_model(self):
        self.model = torch.load(self.backbone_path)
        self.model.load_state_dict(
            torch.load(self.checkpoint_filename, weights_only=True)
        )

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

        mean = torch.tensor(self.config['mean']).view(-1, 1, 1)
        std = torch.tensor(self.config['std']).view(-1, 1, 1)

        processed_images = (imgs_tensor - mean) / std

        # increase dimensions to match input size
        processed_images = processed_images.unsqueeze(2)
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
            result = self.model(images)
        return result, profiles

