#data_process.py
import torch
import numpy as np

img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Example values
crop_size = (128, 128)  
tile_size = 128 
bands = [0, 1, 2]  
num_frames = 1 

# Define the training pipeline
train_pipeline = [
    dict(type="LoadGeospatialImageFromFile", to_float32=True),
    dict(type="LoadGeospatialAnnotations", reduce_zero_label=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="ToTensor", keys=["img", "gt_semantic_seg"]),
    # Change to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),  # Assuming img initially has shape (H, W, C)
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(type="TorchRandomCrop", crop_size=crop_size),  # Randomly crop the image
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), num_frames, tile_size, tile_size),  # Reshape the image
    ),
    dict(type="Reshape", keys=["gt_semantic_seg"], new_shape=(1, tile_size, tile_size)),  # Reshape the target
    dict(type="CastTensor", keys=["gt_semantic_seg"], new_type="torch.LongTensor"),  # Cast target to LongTensor
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),  # Collect processed tensors
]

# Example processing function to simulate the pipeline
def process_input(input_array):
    
    img_tensor = torch.from_numpy(input_array).float()  # Assuming input_array is of type np.float32

    processed_data = {}
    
    # implementation of the pipeline
    for step in train_pipeline:

        if step['type'] == "LoadGeospatialImageFromFile":
            processed_data['img'] = img_tensor  
        elif step['type'] == "LoadGeospatialAnnotations":
            processed_data['gt_semantic_seg'] = torch.zeros((1, img_tensor.shape[2], img_tensor.shape[3]))  # Assuming a single class

        elif step['type'] == "RandomFlip":
            if np.random.rand() < step['prob']:
                processed_data['img'] = torch.flip(processed_data['img'], [3])  # Flip along width

        elif step['type'] == "ToTensor":
            # Already in tensor format from the previous steps
            continue

        elif step['type'] == "TorchPermute":
            processed_data['img'] = processed_data['img'].permute(*step['order'])

        elif step['type'] == "TorchNormalize":
            # Normalize using the provided mean and std
            processed_data['img'] = (processed_data['img'] - torch.tensor(img_norm_cfg['mean']).view(-1, 1, 1)) / \
                                     torch.tensor(img_norm_cfg['std']).view(-1, 1, 1)

        elif step['type'] == "TorchRandomCrop":
            # Random cropping logic would go here
            # This is a placeholder; you'll need to implement the cropping logic.
            pass

        elif step['type'] == "Reshape":
            processed_data[step['keys'][0]] = processed_data[step['keys'][0]].view(step['new_shape'])

        elif step['type'] == "CastTensor":
            processed_data[step['keys'][0]] = processed_data[step['keys'][0]].long()

        elif step['type'] == "Collect":
            # Final output can be collected here
            final_output = {
                'img': processed_data['img'],
                'gt_semantic_seg': processed_data['gt_semantic_seg']
            }

    return final_output

# Example usage
input_array = np.random.rand(4, 3, 1, 256, 256).astype(np.float32)  # Example input
processed_data = process_input(input_array)