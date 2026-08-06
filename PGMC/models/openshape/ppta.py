import torch
import torch.nn as nn
import torch_redstone as rst

from einops import rearrange
from .pointnet_util import PointNetSetAbstraction


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, *extra_args, **kwargs):
        return self.fn(self.norm(x), *extra_args, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., rel_pe = False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.rel_pe = rel_pe
        if rel_pe:
            self.pe = nn.Sequential(nn.Conv2d(3, 64, 1), nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(self, x, centroid_delta):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        pe = self.pe(centroid_delta) if self.rel_pe else 0
        dots = (torch.matmul(q, k.transpose(-1, -2)) + pe) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., rel_pe = False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, rel_pe = rel_pe)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, centroid_delta):
        for attn, ff in self.layers:
            x = attn(x, centroid_delta) + x
            x = ff(x) + x
        return x


class PointPatchTransformer(nn.Module):
    def __init__(self, cache_type, n_cluster, dim, depth, heads, mlp_dim, sa_dim, patches, prad, nsamp, in_dim=3, dim_head=64, rel_pe=False, patch_dropout=0) -> None:
        super().__init__()
        self.cache_type = cache_type
        if cache_type != 'global':
            self.n_cluster = 5
        self.patches = patches
        self.patch_dropout = patch_dropout
        self.sa = PointNetSetAbstraction(npoint=patches, radius=prad, nsample=nsamp, in_channel=in_dim + 3, mlp=[64, 64, sa_dim], group_all=False)
        self.lift = nn.Sequential(nn.Conv1d(sa_dim + 3, dim, 1), rst.Lambda(lambda x: torch.permute(x, [0, 2, 1])), nn.LayerNorm([dim]))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, 0.0, rel_pe)

    @staticmethod
    @torch.no_grad()
    def cluster_patches(local_patches, n_cluster, n_iter=20, tol=1e-4, verbose=False, seed=None):
        if local_patches.dim() == 2:
            local_patches = local_patches.unsqueeze(0)  # [1, N, D]
        B, N, D = local_patches.shape
        device = local_patches.device

        if seed is not None:
            torch.manual_seed(seed)
        idx = torch.randint(0, N, (B, n_cluster), device=device)
        centers = torch.gather(local_patches, 1, idx.unsqueeze(-1).expand(-1, -1, D))  # [B, n_cluster, D]

        for i in range(n_iter):
            dist = torch.cdist(local_patches, centers, p=2)

            labels = dist.argmin(dim=-1)

            new_centers = torch.zeros_like(centers)
            for b in range(B):
                for k in range(n_cluster):
                    mask = (labels[b] == k)
                    if mask.any():
                        new_centers[b, k] = local_patches[b, mask].mean(dim=0)
                    else:
                        new_centers[b, k] = local_patches[b, torch.randint(0, N, (1,), device=device)]

            shift = (centers - new_centers).pow(2).sum(dim=-1).mean()
            centers = new_centers

            if verbose:
                print(f"Iter {i + 1}/{n_iter}  shift={shift.item():.6f}")
            if shift < tol:
                break

        return centers.half()


    def forward(self, xyz: torch.Tensor, features):
        self.sa.npoint = self.patches
        if self.training:
            self.sa.npoint -= self.patch_dropout
        centroids, feature = self.sa(xyz, features)

        x = self.lift(torch.cat([centroids, feature], dim=1))

        x = rst.supercat([self.cls_token, x], dim=-2)
        centroids = rst.supercat([centroids.new_zeros(1), centroids], dim=-1)

        centroid_delta = centroids.unsqueeze(-1) - centroids.unsqueeze(-2)
        x = self.transformer(x, centroid_delta)
        
        if self.cache_type == 'global':
            return x[:, 0]
        else:
            # (5, trans_dim)
            patch_centers = self.__class__.cluster_patches(x[:, 1:], self.n_cluster)
            if self.cache_type == 'local':
                return patch_centers
            elif self.cache_type == 'hierarchical':   # NOTE hierarchical caches
                return x[:, 0], patch_centers
            else:   # NOTE for visualization purpose
                return x[:, 0], x[:, 1:], patch_centers


class Projected(nn.Module):
    def __init__(self, cache_type, ppat, proj) -> None:
        super().__init__()
        self.cache_type = cache_type
        
        self.ppat = ppat
        self.proj = proj

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        if self.cache_type == 'global':
            cls_token = self.ppat(xyz.transpose(-1, -2).contiguous(), features.transpose(-1, -2).contiguous())
            return self.proj(cls_token)
        elif self.cache_type == 'local':
            patch_tokens = self.ppat(xyz.transpose(-1, -2).contiguous(), features.transpose(-1, -2).contiguous())
            return self.proj(patch_tokens)
        elif self.cache_type == 'hierarchical':   # NOTE 'hierarchical' caches
            cls_token, patch_tokens = self.ppat(xyz.transpose(-1, -2).contiguous(), features.transpose(-1, -2).contiguous())
            return self.proj(cls_token), self.proj(patch_tokens)
        else:   # NOTE for visualization purpose
            cls_token, all_patch_tokens, patch_tokens = self.ppat(xyz.transpose(-1, -2).contiguous(), features.transpose(-1, -2).contiguous())
            return self.proj(cls_token), self.proj(all_patch_tokens), self.proj(patch_tokens)


def make(cfg):
    scaling = cfg.model.scaling
    cache_type = cfg.cache_type
    n_cluster = cfg.n_cluster
    if scaling == 1:
        return Projected(
            cache_type,
            PointPatchTransformer(cache_type, n_cluster, 256, 6, 4, 1024, 96, 64, 0.4, 256, cfg.model.in_channel),
            nn.Linear(256, cfg.model.out_channel)
        )
    if scaling == 2:
        return Projected(
            cache_type,
            PointPatchTransformer(cache_type, n_cluster, 512, 6, 8, 1024, 128, 64, 0.4, 256, cfg.model.in_channel),
            nn.Linear(512, cfg.model.out_channel)
        )
    if scaling == 3:
        return Projected(
            cache_type,
            PointPatchTransformer(cache_type, n_cluster, 512, 12, 8, 1024, 128, 128, 0.35, 128, cfg.model.in_channel),
            nn.Linear(512, cfg.model.out_channel)
        )
    if scaling == 4:
        return Projected(
            cache_type,
            PointPatchTransformer(cache_type, n_cluster, 512, 12, 8, 512*3, 256, 384, 0.2, 64, cfg.model.in_channel),
            nn.Linear(512, cfg.model.out_channel)
        )
    if scaling == 5:
        return Projected(
            cache_type,
            PointPatchTransformer(cache_type, n_cluster, 768, 12, 12, 768*3, 256, 512, 0.2, 64, cfg.model.in_channel),
            nn.Linear(768, cfg.model.out_channel)
        )
    if scaling == 6:
        return Projected(
            cache_type,
            PointPatchTransformer(cache_type, n_cluster, 768, 24, 12, 768*4, 256, 512, 0.2, 64, cfg.model.in_channel),
            nn.Linear(768, cfg.model.out_channel)
        )
    raise ValueError(scaling)
