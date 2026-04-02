import h5py
import os
import copy
import json
import numpy as np

import torch
from torch.utils.data import Dataset

from .templates import text_prompts, mn40_gpt35_prompts, mn40_gpt4_prompts, mn40_pointllm_prompts

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def normalize_pc(pc):
    # normalize pc to [-1, 1]
    pc = pc - np.mean(pc, axis=0)
    if np.max(np.linalg.norm(pc, axis=1)) < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / np.max(np.linalg.norm(pc, axis=1))
    return pc

def add_global_noise(xyz, sigma=0.01):
    noise = np.random.normal(0, sigma, xyz.shape)
    return xyz + noise

def add_local_noise(xyz, sigma=0.01, ratio=0.3):
    N = xyz.shape[0]
    idx = np.random.choice(N, int(N*ratio), replace=False)
    xyz[idx] += np.random.normal(0, sigma, (len(idx), 3))
    return xyz

def drop_global(xyz, drop_ratio=0.2):
    N = xyz.shape[0]
    keep_idx = np.random.choice(N, int(N*(1-drop_ratio)), replace=False)
    xyz = xyz[keep_idx]
    return xyz

def drop_local(xyz, drop_ratio=0.1, radius_ratio=0.25):
    N = xyz.shape[0]
    center_idx = np.random.randint(N)
    center = xyz[center_idx]
    dist = np.linalg.norm(xyz - center, axis=1)
    radius = radius_ratio * np.max(dist)
    mask = dist > radius
    xyz = xyz[mask]
    if xyz.shape[0] < N:
        idx = np.random.choice(xyz.shape[0], N - xyz.shape[0], replace=True)
        xyz = np.vstack([xyz, xyz[idx]])
    return xyz

def rotate_xyz(xyz, max_deg=60):
    theta = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0,              0,             1]])
    return xyz @ R.T

def scale_xyz(xyz, scale_low=0.9, scale_high=1.1):
    scale = np.random.uniform(scale_low, scale_high)
    return xyz * scale

def jitter_xyz(xyz, sigma=0.01, clip=0.03):
    noise = np.clip(np.random.normal(0, sigma, xyz.shape), -clip, clip)
    return xyz + noise

def apply_random_modelnet_c_like(xyz, corruption_prob=0.5,noise_type="add_global"):
    if np.random.rand() > corruption_prob:
        return xyz
    if noise_type == 'add_global':
        func = add_global_noise
    elif noise_type == 'add_local':
        func = add_local_noise
    elif noise_type == 'drop_global':
        func = drop_global
    elif noise_type == 'drop_local':
        func = drop_local
    elif noise_type == 'rotate':
        func = rotate_xyz
    elif noise_type == 'scale':
        func = scale_xyz
    elif noise_type == 'jitter':
        func = jitter_xyz

    xyz_aug = func(xyz)

    N = xyz.shape[0]
    if xyz_aug.shape[0] < N:
        idx = np.random.choice(xyz_aug.shape[0], N - xyz_aug.shape[0], replace=True)
        xyz_aug = np.vstack([xyz_aug, xyz_aug[idx]])
    elif xyz_aug.shape[0] > N:
        xyz_aug = xyz_aug[:N]

    return xyz_aug


class ModelNet40_Train(Dataset):
    def __init__(self, config, intensity=1, noise_type='add_global'):
        self.lm3d = config.lm3d
        self.template = text_prompts
        self.npoints = config.npoints
        self.data_path = config.modelnet40_root
        self.catfile = os.path.join(self.data_path, "shape_names.txt")
        self.classnames = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.classnames, range(len(self.classnames))))

        # 多个 h5 文件路径
        if self.lm3d != 'openshape':
            h5_files = [
                os.path.join(self.data_path, "ply_data_trainminusval.h5"),
                os.path.join(self.data_path, "ply_data_valid.h5"),
                os.path.join(self.data_path, "ply_data_test.h5"),
            ]
        else:
            h5_files = [
                os.path.join(self.data_path, "ply_data_test.h5")
            ]

        # 合并读取
        data_list, label_list = [], []
        for h5_path in h5_files:
            if not os.path.exists(h5_path):
                print(f"[Warning] Missing file: {h5_path}")
                continue
            with h5py.File(h5_path, 'r') as f:
                data_list.append(f['data'][:])  # [N, npoints, 6]
                label_list.append(f['label'][:])  # [N, 1]

        self.data = np.concatenate(data_list, axis=0)
        self.labels = np.concatenate(label_list, axis=0).squeeze()
        print(f"Loaded {len(self.labels)} samples from {len(h5_files)} h5 files.")

        # 其他元信息
        self.openshape_split = json.load(open(os.path.join(self.data_path, "test_split.json"), "r"))
        self.cate_to_id = {c: str(i) for i, c in enumerate(self.classnames)}
        self.intensity = intensity
        self.noise_type = noise_type

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx][:self.npoints]
        xyz = sample[:, :3]

        if self.intensity > 0:
            xyz = apply_random_modelnet_c_like(xyz, self.intensity, self.noise_type)

        # 坐标处理
        if self.lm3d == 'openshape':
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
            xyz = normalize_pc(xyz)
        else:
            xyz[:, 0:3] = pc_normalize(xyz[:, 0:3])

        # 转 tensor
        rgb = np.ones_like(xyz) * 0.4
        xyz = torch.from_numpy(xyz).float()
        rgb = torch.from_numpy(rgb).float()

        label = int(self.labels[idx])
        label_name = self.classnames[label]

        return xyz, label, label_name, rgb