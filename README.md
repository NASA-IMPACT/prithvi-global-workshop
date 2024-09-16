For all resourcses: https://docs.google.com/spreadsheets/d/1Fkl0BG3eHujYGRiVm0l5RzC3LD92CjcA/edit?usp=drive_link&ouid=107344732858203799494&rtpof=true&sd=true

###  Other than Srija, rest of the people can use :
>> prithvi_burn or

>> prithvi_crop_10prcnt (10% dataset adopted) or


>> prithvi_burn_intensity

folders for fine tuning


![image](https://github.com/user-attachments/assets/d31c0a58-17f3-44a3-9d24-9347cbc95aac)
***************************************************************************************************************************************

## For Srija to adapt the model for CO2 flux fine tuning:
### new_flood_v2 Model details: Prithvi HLS Global+ UNet: finetuning for flood map
<!---- Provide an overview of what is being achieved in this repo ----> 
Checkpoint: [ Pretrained Checkpoint](https://www.nsstc.uah.edu/data/sujit.roy/Prithvi_checkpoints/)

### Architecture:

MAE_VIT Encoder+Decoder (freezed) --> UNet(Unfreezed) 


### Instructions
1. Go to new_flood_v2 folder

2. Run slurm script using:
   ```python
   sbatch unet_flood_new.sh
   ```
3. config file is: config.yaml

4. unet_flood_new.sh runs main_flood_new.py, which initializes model by calling unet.py.

5. The unet.py calls prithvi_global_loader.py (wrapper around Prithvi_global model), which actually calls Prithvi_global_v1/mae/models_mae.py (i.e. the core Prithvi model architecture). 

![IMG_3948](https://github.com/user-attachments/assets/ae5c0b64-31e3-495c-b485-6e4cb9eecb06)
