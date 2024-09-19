For all resources: https://docs.google.com/spreadsheets/d/1Fkl0BG3eHujYGRiVm0l5RzC3LD92CjcA/edit?usp=drive_link&ouid=107344732858203799494&rtpof=true&sd=true

This code repo is created without using mmsegmentation purposefully

##  A. Other than Srija, rest of the people can use any of the fine tuning task folders:

>> prithvi_crop_10prcnt_mmseg_style (10% dataset adopted) 

### LORA model for burn_scar:

>> prithvi_burn_LORA_v2

If anyone wants to use LORA adapter for peft (precision efficient fine tuning), can use this model for any downstream

### Requirements:
 Set up environment (eg:hls2) following Setup section of https://github.com/NASA-IMPACT/hls-foundation-os

### Data:
Use instruction from https://anaconda.org/conda-forge/git-lfs/ to install git-lfs in the environemnt.

Use "install git-lfs" in your server before downloading data using "wget data_link", as the data files are in git-lfs format

For data links: Use https://docs.google.com/spreadsheets/d/1Fkl0BG3eHujYGRiVm0l5RzC3LD92CjcA/edit?usp=drive_link&ouid=107344732858203799494&rtpof=true&sd=true

### Architecture: 

1. For all  model except LORA: prithvi_encoder (not freezed) --> Upsampling_conv layers (not freezed)

2. For LORA model: prithvi_encoder (other than linear layers rest are freezed) --> Upsampling_conv layers (not freezed)

### Instructions:
1. Go to required folder

2. Create .sh file:

```python
#!/bin/bash
#SBATCH -p shared                 # Partition to submit to
#SBATCH --gres=gpu:a100:1         # Request 4 A100 GPUs
#SBATCH --ntasks-per-node=1       # Number of tasks per node
#SBATCH --cpus-per-task=24         # Number of CPU cores per task
#SBATCH --mem-per-cpu=10G         # Memory per CPU core
#SBATCH -t 00-05:00               # Runtime in D-HH:MM format
#SBATCH -J hls_rinki                 # Job name
#SBATCH -o slurm_logs_prithvi_burn/%j.txt   # Standard output and error log

# Activate your environment
source activate hls2

# Run your commands
torchrun \
  --standalone \
  --nnodes 1 \
  --nproc_per_node 1\
  main_prithvi_burn.py 
```



3. Run slurm script using:
   ```python
   sbatch prithvi_crop.sh
   ```
4. config file is: config.yaml

5. prithvi_burn.sh runs main_prithvi_crop.py, which initializes model by calling model.py 

6. The model.py calls Head.py, Neck.py and Seg_head.py

7. Example instruction is provided for crop_segmentation only.
   Naming convention is similar for rest of the downstreams. 

![image](https://github.com/user-attachments/assets/d31c0a58-17f3-44a3-9d24-9347cbc95aac)
***************************************************************************************************************************************

## B. For Srija to adapt the model for CO2 flux fine tuning:
### new_flood_v2 Model details: Prithvi HLS Global+ UNet: finetuning for flood map
<!---- Provide an overview of what is being achieved in this repo ----> 
Checkpoint: [ Pretrained Checkpoint](https://www.nsstc.uah.edu/data/sujit.roy/Prithvi_checkpoints/)

### Architecture:

MAE_VIT Encoder+Decoder (freezed) --> UNet(Unfreezed) 


### Instructions:
1. Go to new_flood_v2 folder

2. Run slurm script using:
   ```python
   sbatch unet_flood_new.sh
   ```
3. config file is: config.yaml

4. unet_flood_new.sh runs main_flood_new.py, which initializes model by calling unet.py.

5. The unet.py calls prithvi_global_loader.py (wrapper around Prithvi_global model), which actually calls Prithvi_global_v1/mae/models_mae.py (i.e. the core Prithvi model architecture). 

![IMG_3948](https://github.com/user-attachments/assets/ae5c0b64-31e3-495c-b485-6e4cb9eecb06)
