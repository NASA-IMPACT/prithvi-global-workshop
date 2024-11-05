import torch
import torch.nn as nn

from prithvi_global.mae.models_mae import MaskedAutoencoderViT
from functools import partial


class Prithvi(nn.Module):
    def __init__(self, prithvi_weight, prithvi_config, n_frame,input_size):
        super(Prithvi, self).__init__()

        self.weights_path = prithvi_weight
        self.checkpoint = torch.load(self.weights_path)

        self.config = prithvi_config

        self.prithvi_model = MaskedAutoencoderViT(
            input_size=input_size,
            patch_size=self.config.MODEL.PATCH_SIZE,
            in_chans=len(self.config.DATA.BANDS),
            embed_dim=self.config.MODEL.EMBED_DIM,
            depth=self.config.MODEL.DEPTH,
            num_heads=self.config.MODEL.NUM_HEADS,
            decoder_embed_dim=self.config.MODEL.DECODER_EMBED_DIM,
            decoder_depth=self.config.MODEL.DECODER_DEPTH,
            decoder_num_heads=self.config.MODEL.DECODER_NUM_HEADS,
            mlp_ratio=self.config.MODEL.MLP_RATIO,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            norm_pix_loss=self.config.MODEL.NORM_PIX_LOSS,
            coords_encoding=self.config.MODEL.COORDS_ENCODING,
            coords_drop_rate=self.config.MODEL.COORDS_DROP_RATE,
            coords_scale_learn=self.config.MODEL.COORDS_SCALE_LEARN,
            drop_channels_rate=self.config.MODEL.DROP_CHANNELS_RATE
        )


        print("checkpoint names:",self.checkpoint.keys())

        _ = self.prithvi_model.load_state_dict(self.checkpoint, strict=False)
        #print("initialized model")

    def forward(self,x,temp,loc,mask):

        latent, mask, ids_restore = self.prithvi_model.forward_encoder(x, temp, loc, 0)#mask_ratio=0

        pred = latent

        return pred
