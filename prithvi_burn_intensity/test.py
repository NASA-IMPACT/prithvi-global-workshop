import torch

weights_path ="/rhome/rghosal/Rinki/rinki-hls-foundation-os/flood_checkpoint/checkpoint_flood_new_sample.pth" 

checkpoint = torch.load(weights_path)

print(checkpoint.keys())