import torch
import timm
import numpy as np
from torch import nn

from .point_encoder import PointcloudEncoder


class Uni3D(nn.Module):
    def __init__(self, args, point_encoder):
        super().__init__()
        self.cache_type = args.cache_type
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder

    def encode_pc(self, pc):
        xyz = pc[:,:,:3].contiguous()
        color = pc[:,:,3:].contiguous()
        
        if self.cache_type == 'global':
            pc_feat = self.point_encoder(xyz, color)
            return pc_feat
        elif self.cache_type == 'local':
            patch_centers = self.point_encoder(xyz, color)
            return patch_centers
        elif self.cache_type == 'hierarchical':
            pc_feat, patch_centers = self.point_encoder(xyz, color)
            return pc_feat, patch_centers
        else:
            pc_feat, all_patches, patch_centers = self.point_encoder(xyz, color)
            return pc_feat, all_patches, patch_centers

    def forward(self, pc, text, image):
        text_embed_all = text
        image_embed = image   
        pc_embed = self.encode_pc(pc)
        return {'text_embed': text_embed_all,
                'pc_embed': pc_embed,
                'image_embed': image_embed,
                'logit_scale': self.logit_scale.exp()}


def create_uni3d(args):  
    # create transformer blocks for point cloud via timm
    # NOTE 1. pc_model: model name  2. pretrained_pc: model weights
    point_transformer = timm.create_model(args.pc_model, checkpoint_path=args.pretrained_pc, drop_path_rate=args.drop_path_rate)

    # create whole point cloud encoder
    point_encoder = PointcloudEncoder(point_transformer, args)

    # uni3d model
    model = Uni3D(args, point_encoder=point_encoder,)
    return model
