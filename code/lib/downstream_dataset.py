from torch.utils.data import Dataset

from utils import load_raster, preprocess_image
from consts import CROP_SIZE

class DonwstreamDataset(Dataset):
    def __init__(self, path, means, stds):
        self.data_dir = path
        self.means = means
        self.stds = stds


    def __len__(self):
        #print("dataset length",len(self.input_plus_mask_path))
        return len(self.data_dir)


    def __getitem__(self, idx):
        image_path = self.data_dir[idx][0]
        mask_path = self.data_dir[idx][1]

        image = load_raster(image_path,crop=CROP_SIZE)

        final_image = preprocess_image(image,self.means,self.stds)

        final_mask = load_raster(mask_path,crop=CROP_SIZE)

        final_image = final_image.squeeze(0)
        final_mask = final_mask.squeeze(0)

        return final_image, final_mask
