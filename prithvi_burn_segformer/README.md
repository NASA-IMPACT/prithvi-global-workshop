
## Prithvi HLS Global+ Segmentor (https://arxiv.org/pdf/2105.05633) : finetuning for burn_scar
<!---- Provide an overview of what is being achieved in this repo ----> 
Checkpoint: [ Pretrained Checkpoint](https://www.nsstc.uah.edu/data/sujit.roy/Prithvi_checkpoints/)

## Architecture:

MAE_VIT Encoder (Unfreezed) --> Segmentator(Unfreezed) 


## Instructions
1. Go to prithvi_burn_segformer folder

2. Run slurm script using:
   ```python
   sbatch prithvi_burn.sh
   ```
3. config file is: config.yaml

4. prithvi_burn.sh runs main_prithvi_burn.py, which initializes model by calling model.py.

5. The model.py calls prithvi_global_loader.py (wrapper around Prithvi_global model), which actually calls Prithvi_global_v1/mae/models_mae.py (i.e. the core Prithvi model architecture) and segmentation head part i.e., head.py.

![IMG_3957](https://github.com/user-attachments/assets/c7a351dd-cc56-455b-a613-858667e12245)