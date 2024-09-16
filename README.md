For all resourcses: https://docs.google.com/spreadsheets/d/1Fkl0BG3eHujYGRiVm0l5RzC3LD92CjcA/edit?usp=drive_link&ouid=107344732858203799494&rtpof=true&sd=true

This code repo is created without using mmsegmentation purposefully

##  A. Other than Srija, rest of the people can use any of the fine tuning task folders:
>> prithvi_burn or

>> prithvi_crop_10prcnt (10% dataset adopted) or


>> prithvi_burn_intensity

### LORA model for burn_scar:

>> prithvi_burn_LORA_v2

If anyone wants to use LORA adapter for peft (precision efficient fine tuning), can use this model for any downstream

### Architecture: 

1. For all  model except LORA: prithvi_encoder (not freezed) --> Upsampling_conv layers (not freezed)

2. For LORA model: prithvi_encoder (other than linear layers rest are freezed) --> Upsampling_conv layers (not freezed)

### Instructions:
1. Go to required prithvi_burn_scar folder

2. Run slurm script using:
   ```python
   sbatch prithvi_burn.sh
   ```
3. config file is: config.yaml

4. prithvi_burn.sh runs main_burn_scar.py, which initializes model by calling model.py or model_old.py

5. The model.py or model_old.py calls prithvi_global_loader.py (wrapper around Prithvi_global model), which actually calls Prithvi_global_v1/mae/models_mae.py (i.e. the core Prithvi model architecture).

6. Example instruction is provided for burn_scar only.
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
