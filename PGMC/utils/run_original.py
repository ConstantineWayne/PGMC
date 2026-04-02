import os
import re
import yaml
import math
import numpy as np
import clip
import random
import argparse
import open_clip
from collections import OrderedDict
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.modelnet_c import ModelNet_C
from datasets.modelnet_c_views import ModelNet_C_Views
from datasets.modelnet40_sdxl import ModelNet40_SDXL
from datasets.modelnet40_c import ModelNet40_C
from datasets.modelnet40 import ModelNet40
from datasets.modelnet40_views import ModelNet40_Views
from datasets.scanobjnn import ScanObjNN
from datasets.scanobjectnn import ScanObjectNN
from datasets.sonn_c import SONN_C
from datasets.snv2_c import SNV2_C
from datasets.objaverse_lvis import Objaverse_LVIS
from datasets.omniobject3d import OmniObject3D
from datasets.sim2real_sonn import Sim2Real_SONN
from datasets.pointda_modelnet import PointDA_ModelNet
from datasets.pointda_scannet import PointDA_ScanNet
from datasets.pointda_shapenet import PointDA_ShapeNet

from datasets.utils import AugMixAugmenter
import torchvision.transforms as transforms

from models import uni3d
from models import openshape
from models import ulip
from utils import *
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()

    # system settings
    parser.add_argument('--config', dest='config', help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--lm3d', default='uni3d', type=str, help='which large multi-modal 3d model to use')
    parser.add_argument('--seed', type=int, default=1, help='experiment seed')
    parser.add_argument("--device", default=0, type=int, help="The GPU device id to use.", )
    parser.add_argument('--distributed', action='store_true', default=False, help='whether use distributed inference')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true',
                        help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--print-freq', type=int, default=500, help='result print frequency')
    parser.add_argument('--cache-type', type=str, default='hierarchical',
                        choices=['global', 'local', 'hierarchical', 'vis'])
    parser.add_argument("--k_shot", type=int, default=3, help="number of shots cached in per class")
    parser.add_argument("--n_cluster", type=int, default=3, help="number of local clustered parts for a 3D object")
    parser.add_argument('--alpha', default=4.0, type=float,
                        help="a balance factor to adjust the weights of cached logits")
    parser.add_argument('--beta', default=3.0, type=float,
                        help="a sharpness factor to adjust query-key attention computation")

    # uni3d
    parser.add_argument("--pc-model", type=str, default="eva_giant_patch14_560",
                        help="Name of pointcloud backbone to use.", )
    parser.add_argument("--pretrained-pc", default='', type=str,
                        help="Use a pretrained CLIP model vision weights with the specified tag or file path.", )
    parser.add_argument("--clip-model", type=str, default="EVA02-E-14-plus",
                        help="Name of the vision and text backbone to use.", )
    parser.add_argument("--pretrained", default='weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin', type=str,
                        help="open clip version", )
    parser.add_argument('--ckpt_path', default='weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt',
                        help='the ckpt to test 3d zero shot')
    parser.add_argument('--drop-path-rate', default=0.0, type=float, help="passed by uni3d and ulip")

    # openshape
    parser.add_argument("--oshape-version", type=str, choices=["vitg14", "vitl14"], default="vitg14")
    parser.add_argument('--npoints', default=1024, type=int, help='number of points used for pre-train and test.')
    parser.add_argument("--pc-feat-dim", type=int, default=768, help="Pointcloud feature dimension.")
    parser.add_argument("--group-size", type=int, default=32, help="Pointcloud Transformer group size.")
    parser.add_argument("--num-group", type=int, default=512, help="Pointcloud Transformer number of groups.")
    parser.add_argument("--pc-encoder-dim", type=int, default=512, help="Pointcloud Transformer encoder dimension.")
    parser.add_argument("--embed-dim", type=int, default=512, help="teacher embedding dimension.")
    parser.add_argument("--patch-dropout", type=float, default=0., help="flip patch dropout.")

    # ulip: Share ***point encoder*** config with openshape since both of them use `PointBERT`
    parser.add_argument("--ulip-version", type=str, choices=["ulip1", "ulip2"], default="ulip2")
    parser.add_argument("--pc-depth", type=int, default=12, help="number of layers of PointTransformer")
    parser.add_argument("--num-head", type=int, default=6, help="number of heads in PointTransformer attention")
    parser.add_argument("--encoder-dim", type=int, default=256,
                        help="dimensions of the encoder before feeding  PointTransformer")
    parser.add_argument("--slip-ckpt-path", type=str, default="/data/hdd2/hudisen/3D_data/slip_base_100ep.pt")

    # data
    parser.add_argument('--dataset', default='modelnet40', type=str, help="Datasets to process")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./data/',
                        help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--objaverse_lvis_root', type=str, default='/data/hdd2/hudisen/3D_data/objaverse_lvis', help='')
    parser.add_argument('--omniobject3d_root', type=str, default='/data/hdd2/hudisen/3D_data/omniobject3d', help='')
    parser.add_argument('--scanobjnn_root', type=str, default='/data/hdd2/hudisen/3D_data/scanobjnn', help='')
    parser.add_argument('--scanobjectnn_root', type=str, default='/data/hdd2/hudisen/3D_data/scanobjectnn', help='')
    parser.add_argument('--sonn_c_root', type=str, default='/data/hdd2/hudisen/3D_data/sonn_c', help='')
    parser.add_argument('--sonn_variant', type=str, default='hardest', help='')
    parser.add_argument('--modelnet40_root', type=str, default='/data/hdd2/hudisen/3D_data/modelnet40', help='')
    parser.add_argument('--modelnet_c_root', type=str, default='/data/hdd2/hudisen/3D_data/modelnet_c', help='')
    parser.add_argument('--modelnet40_c_root', type=str, default='/data/hdd2/hudisen/3D_data/modelnet40_c', help='')
    parser.add_argument('--snv2_c_root', type=str, default='/data/hdd2/hudisen/3D_data/snv2_c', help='')
    parser.add_argument('--cor_type', type=str, default='add_global_2', help='data corruption type')
    parser.add_argument('--sim2real_type', type=str, default='so_obj_only_9',
                        choices=['so_obj_only_9', 'so_obj_only_11',
                                 'so_obj_bg_9', 'so_obj_bg_11', 'so_hardest_9', 'so_hardest_11'])
    parser.add_argument('--pointda_type', type=str, default='so_obj_only_9',
                        choices=['modelnet', 'scannet', 'shapenet'])
    parser.add_argument('--cname', type=str, default='airplane', help='specify class name for visualization')

    parser.add_argument("--p_thres", type=float, default=0.1, help="take how many confident images from all images")
    parser.add_argument("--obj-id", type=int, default=0,
                        help="object id when visualizing all patches and the clustering"
                             "centers of a 3D object")

    args = parser.parse_args()

    return args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_original_dataloader(args,):
    if "modelnet" in args.dataset:
        original_dataset = ModelNet40(args)
        original_test_loader = DataLoader(original_dataset, batch_size=1, num_workers=2, shuffle=True)
    else:
        original_dataset = ScanObjectNN(args)
        original_test_loader = DataLoader(original_dataset)
    return original_test_loader







