
## Prithvi HLS Global+ UNet (https://arxiv.org/pdf/2105.05633) : finetuning for flood map
<!---- Provide an overview of what is being achieved in this repo ----> 
Checkpoint: [ Pretrained Checkpoint](https://www.nsstc.uah.edu/data/sujit.roy/Prithvi_checkpoints/)

## Architecture:

MAE_VIT Encoder+Decoder (freezed) --> UNet(Unfreezed) 


## Instructions
1. Go to new_flood_v2 folder

2. Run slurm script using:
   ```python
   sbatch unet_flood_new.sh
   ```
3. config file is: config.yaml

4. unet_flood_new.sh runs main_flood_new.py, which initializes model by calling unet.py.

5. The unet.py calls prithvi_global_loader.py (wrapper around Prithvi_global model), which actually calls Prithvi_global_v1/mae/models_mae.py (i.e. the core Prithvi model architecture). 

