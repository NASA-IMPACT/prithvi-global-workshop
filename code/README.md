
## Prithvi HLS Global+ Segmentation layers : finetuning for burn_scar
<!---- Provide an overview of what is being achieved in this repo ----> 
Checkpoint: [ Pretrained Checkpoint](https://www.nsstc.uah.edu/data/sujit.roy/Prithvi_checkpoints/)

## Architecture:

MAE_VIT Encoder (Unfreezed) --> Segmentation layers(Unfreezed) 


## Instructions
1. Go to prithvi_burn folder

2. Run slurm script using:
   ```python
   sbatch prithvi_burn.sh
   ```
3. config file is: config.yaml

4. prithvi_burn.sh runs main_prithvi_burn.py, which initializes model by calling model.py.

5. The model.py calls prithvi_global_loader.py (wrapper around Prithvi_global model), which actually calls Prithvi_global_v1/mae/models_mae.py (i.e. the core Prithvi model architecture) 

![IMG_3958](https://github.com/user-attachments/assets/609db6ba-5504-4a63-8d8a-cf869f25b939)
