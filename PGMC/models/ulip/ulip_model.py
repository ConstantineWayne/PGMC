import torch
from torch import nn

from .pointbert.point_encoder import PointTransformer


class ULIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cache_type = args.cache_type
        
        # --- point encoder
        self.point_encoder = PointTransformer(args)
        self.pc_projection = nn.Parameter(torch.empty(args.pc_feat_dim, 512))

    def forward(self, pc,conf=False):
        if self.cache_type == 'global':
            if conf:
                pc_feats,patch_centers = self.point_encoder.get_logits(pc)
                pc_embed = pc_feats @ self.pc_projection
                patch_embed = patch_centers @ self.pc_projection
                return pc_embed,patch_embed
            else:
                pc_feat = self.point_encoder(pc)
                pc_embed = pc_feat @ self.pc_projection
                return pc_embed
        elif self.cache_type == 'local':
            patch_centers = self.point_encoder(pc)
            patch_embed = patch_centers @ self.pc_projection
            return patch_embed
        elif self.cache_type == 'hierarchical': # NOTE 'hierarchical' caches
            pc_feat, patch_centers = self.point_encoder(pc)    
            pc_embed = pc_feat @ self.pc_projection
            patch_embed = patch_centers @ self.pc_projection
            return pc_embed, patch_embed
        else:   # NOTE for visualization purpose
            pc_feat, all_patches, patch_centers = self.point_encoder(pc)    
            pc_embed = pc_feat @ self.pc_projection
            all_patch_embed = all_patches @ self.pc_projection
            patch_embed = patch_centers @ self.pc_projection
            return pc_embed, all_patch_embed, patch_embed
