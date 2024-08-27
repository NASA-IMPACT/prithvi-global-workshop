## Prithvi HLS Global+UNet finetuning for flood
<!---- Provide an overview of what is being achieved in this repo ----> 
Checkpoint: [ Pretrained Checkpoint](https://www.nsstc.uah.edu/data/sujit.roy/Prithvi_checkpoints/)

## Architecture:

In this work we attached UNet down and up layers at input and output of prithvi global model. These in and out layer embeddings have skip connection in between them from input to output. This modification in architecture is done because, for CO2 downstream task, there will be MERRA data which the Prithvi_global model has never seen before. However in this repo the config file is provided for 6 channel input for flood mapping downstream task and so is the model trained for. To train this model for C02 downstream task, one have to modify the config.yaml file accordingly.


## Instructions
1. Go to new_flood folder

2. Run slurm script using:
   ```python
   sbatch unet_flood_new.sh
   ```
3. config file is: config.yaml

4. unet_flood_new.sh runs main_flood_new.py, which initializes model Unet by calling unet.py.

5. The unet.py calls prithvi_global_loader.py (wrapper around Prithvi_global model), which actually calls Prithvi_global_v1/mae/models_mae.py (i.e. the core Prithvi model architecture).

