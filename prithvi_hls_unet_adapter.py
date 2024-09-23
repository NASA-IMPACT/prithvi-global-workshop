"""
UNet adapter + LoRA for Prithvi HLS model.

Code adapted from
https://github.com/NASA-IMPACT/WxFM-fine-tuning/blob/fdcc46393a433465dc40a64a15da11e8d1a0698f/src/unet_wxfm_without_pool.py
"""

import peft
import torch
from torch import nn

from Prithvi import MaskedAutoencoderViT

# %%
class UNetEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(UNetEncoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 8, hidden_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        return enc1, enc2, enc3, enc4


class UNetDecoder(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(UNetDecoder, self).__init__()

        self.decoder4 = nn.Sequential(
            nn.Conv2d(hidden_channels * 16, hidden_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 8, hidden_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.ReLU(inplace=True),
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(hidden_channels * 12, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(hidden_channels * 6, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, enc1, enc2, enc3, enc4, bottleneck):
        dec4 = torch.cat((bottleneck, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = torch.cat((dec4, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = torch.cat((dec3, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = torch.cat((dec2, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        output = self.final_conv(dec1)
        return output


# %%
class UNetWithTransformer(nn.Module):
    """
    U-Net adapter with the following flow:
    x -> unet_enc -> adapter -> unet_dec
    x -> embed -> backbone
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 8):
        super().__init__()

        self.unet_encoder = UNetEncoder(in_channels=in_channels, hidden_channels=hidden_channels)
        self.backbone: torch.nn.Module = MaskedAutoencoderViT(
            in_chans=hidden_channels * 8, num_frames=1
        )
        self.unet_decoder = UNetDecoder(hidden_channels=hidden_channels, out_channels=in_channels)

        # Freeze backbone weights
        self.backbone.requires_grad_(requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 4  # BCHW
        # assert input.shape == torch.Size([2, 3, 224, 224])

        enc1, enc2, enc3, enc4 = self.unet_encoder(input)
        imgs: torch.Tensor = enc4.unsqueeze(dim=2)  # Add time dimension, so BCHW -> BCTHW
        # assert imgs.shape == torch.Size([2, 64, 1, 224, 224])

        with torch.no_grad():
            _, pred, mask = self.backbone(imgs=imgs)
            pred_image: torch.Tensor = self.backbone.unpatchify(x=pred)
        # assert pred_image.shape == torch.Size([2, 64, 1, 224, 224])

        output: torch.Tensor = self.unet_decoder(
            enc1=enc1, enc2=enc2, enc3=enc3, enc4=enc4, bottleneck=pred_image.squeeze(dim=2)
        )
        # assert output.shape == torch.Size([2, 3, 224, 224])
        return output


# %%
if __name__ == "__main__":
    model = UNetWithTransformer()

    # for name, module in model.named_modules():
    #     print(f"  Layer: {name}, {type(module)}")

    # Create LoraConfig and apply to neural network model at select layers
    target_modules: list[str] = [
        "backbone.blocks.0.mlp.fc1",
        "backbone.blocks.0.mlp.fc2",
        "backbone.blocks.1.mlp.fc1",
        "backbone.blocks.1.mlp.fc2",
        "backbone.blocks.2.mlp.fc1",
        "backbone.blocks.2.mlp.fc2",
        "backbone.blocks.3.mlp.fc1",
        "backbone.blocks.3.mlp.fc2",
        "backbone.blocks.4.mlp.fc1",
        "backbone.blocks.4.mlp.fc2",
        "backbone.blocks.5.mlp.fc1",
        "backbone.blocks.5.mlp.fc2",
        "backbone.blocks.6.mlp.fc1",
        "backbone.blocks.6.mlp.fc2",
        "backbone.blocks.7.mlp.fc1",
        "backbone.blocks.7.mlp.fc2",
        "backbone.blocks.8.mlp.fc1",
        "backbone.blocks.8.mlp.fc2",
        "backbone.blocks.9.mlp.fc1",
        "backbone.blocks.9.mlp.fc2",
        "backbone.blocks.10.mlp.fc1",
        "backbone.blocks.10.mlp.fc2",
        "backbone.blocks.11.mlp.fc1",
        "backbone.blocks.11.mlp.fc2",
        "backbone.blocks.12.mlp.fc1",
        "backbone.blocks.12.mlp.fc2",
        "backbone.blocks.13.mlp.fc1",
        "backbone.blocks.13.mlp.fc2",
        "backbone.blocks.14.mlp.fc1",
        "backbone.blocks.14.mlp.fc2",
        "backbone.blocks.15.mlp.fc1",
        "backbone.blocks.15.mlp.fc2",
        "backbone.blocks.16.mlp.fc1",
        "backbone.blocks.16.mlp.fc2",
        "backbone.blocks.17.mlp.fc1",
        "backbone.blocks.17.mlp.fc2",
        "backbone.blocks.18.mlp.fc1",
        "backbone.blocks.18.mlp.fc2",
        "backbone.blocks.19.mlp.fc1",
        "backbone.blocks.19.mlp.fc2",
        "backbone.blocks.20.mlp.fc1",
        "backbone.blocks.20.mlp.fc2",
        "backbone.blocks.21.mlp.fc1",
        "backbone.blocks.21.mlp.fc2",
        "backbone.blocks.22.mlp.fc1",
        "backbone.blocks.22.mlp.fc2",
        "backbone.blocks.23.mlp.fc1",
        "backbone.blocks.23.mlp.fc2",
        "backbone.decoder_blocks.0.mlp.fc1",
        "backbone.decoder_blocks.0.mlp.fc2",
        "backbone.decoder_blocks.1.mlp.fc1",
        "backbone.decoder_blocks.1.mlp.fc2",
        "backbone.decoder_blocks.2.mlp.fc1",
        "backbone.decoder_blocks.2.mlp.fc2",
        "backbone.decoder_blocks.3.mlp.fc1",
        "backbone.decoder_blocks.3.mlp.fc2",
        "backbone.decoder_blocks.4.mlp.fc1",
        "backbone.decoder_blocks.4.mlp.fc2",
        "backbone.decoder_blocks.5.mlp.fc1",
        "backbone.decoder_blocks.5.mlp.fc2",
        "backbone.decoder_blocks.6.mlp.fc1",
        "backbone.decoder_blocks.6.mlp.fc2",
        "backbone.decoder_blocks.7.mlp.fc1",
        "backbone.decoder_blocks.7.mlp.fc2",
    ]
    peft_config = peft.LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
        target_modules=target_modules,
    )
    lora_model: nn.Module = peft.get_peft_model(model=model, peft_config=peft_config)
    print(f"LoRA config applied to model at layers: {target_modules}")
    print(lora_model)
    lora_model.print_trainable_parameters()
