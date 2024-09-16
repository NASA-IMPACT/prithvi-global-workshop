'''Experiments on Parameter-Efficient Finetuning (PEFT)
with Low-Rank Adaptation (LoRA)
Uses HuggingFace's peft library.
References:
- https://huggingface.co/docs/peft/developer_guides/custom_models#timm-model
'''
import peft
import torch
from model_old import prithvi_wrapper
from Prithvi_global_v1.mae.config import get_config

n_channels=6
n_classes=2
n_frame=1
embed_size=1024
input_size=[1,224,224]
patch_size=[1,16,16]
prithvi_weight= "/rhome/rghosal/Rinki/rinki-hls-foundation-os/prithvi_merra/checkpoint_global.pt"
prithvi_config=get_config(None)

model = prithvi_wrapper(n_channels,n_classes,n_frame,embed_size,input_size,patch_size,prithvi_weight,prithvi_config)
weights_path =prithvi_weight
checkpoint = torch.load(weights_path)

print(f"Model loaded from checkpoint")

# Calculate percentage of trainable parameters
def print_trainable_parameters(model: torch.nn.Module):
    
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )

'''for name, module in model.named_modules():
    print(f"  Layer: {name}, {type(module)}")'''

'''for name, param in model.named_parameters():
    
    dec_name="_".join(name.split("_")[:2])
   
    if dec_name=="prithvi.prithvi_model.decoder":
        name_n=".".join(name.split(".")[2:])
        del checkpoint["model"][name_n] 
        del model.name'''

#print("checkpoint",checkpoint["model"].keys())  

'''for name, param in model.named_parameters():
    print("model",name)  '''


#print_trainable_parameters(model=model)
for name, module in model.named_modules():
    print(f"  Layer: {name}, {type(module)}")
########################################################################
# %%
# Create LoraConfig and apply to neural network model at select layers
target_modules: list[str] = ["prithvi.prithvi_model.blocks.0.mlp.fc1",
                             "prithvi.prithvi_model.blocks.0.mlp.fc2"]

peft_config = peft.LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    bias="none",
    target_modules=target_modules,
    # modules_to_save=["model.head"],
)
lora_model = peft.get_peft_model(model=model, peft_config=peft_config)
print(f"LoRA config applied to model at layers: {target_modules}")
print_trainable_parameters(model=lora_model)
print(lora_model)

# %%
# Finetune LoRA model
# TODO