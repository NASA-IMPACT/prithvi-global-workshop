import torch
import torch.nn as nn
from block import Block, FeedForward, init_weights
from timm.models.layers import trunc_normal_
from einops import rearrange
import random
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout
    ):
        super().__init__()
        self.d_encoder = d_encoder #dim of encoder embed
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model #dim of decoder embed
        self.d_ff = d_ff #mlp_dim
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)] #drop_path
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size

        x = self.proj_dec(x) #8,196,512
        
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1) #8,2,512
        
        x = torch.cat((x, cls_emb), 1)
        
        for blk in self.blocks:
            x = blk(x)
           
        x = self.decoder_norm(x)
        
        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        
        patches = patches @ self.proj_patch
        
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        # Bilinearly upsample to the original image size
        upsampled_map = F.interpolate(masks, size=(224, 224), mode='bilinear', align_corners=False)

        # Apply softmax on the class dimension
        #segmentation_map = F.softmax(upsampled_map,dim=1)

        return upsampled_map
    
