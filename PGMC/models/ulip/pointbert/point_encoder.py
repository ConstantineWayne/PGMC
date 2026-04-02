from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from .dvae import Group
from .dvae import Encoder
import faiss

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class PointTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cache_type = args.cache_type
        if self.cache_type != 'global':
            self.n_cluster = args.n_cluster

        self.trans_dim = args.pc_feat_dim // 2 # 384
        self.depth = args.pc_depth             # 12
        self.drop_path_rate = args.drop_path_rate # 0.1
        self.num_heads = args.num_head # 6

        self.group_size = args.group_size         # 32
        self.num_group = args.num_group           # 512
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dim = args.encoder_dim     # 256
        self.encoder = Encoder(encoder_channel=self.encoder_dim)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dim, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

    def get_loss_acc(self, pred, gt, smoothing=True):
        # import pdb; pdb.set_trace()
        gt = gt.contiguous().view(-1).long()

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = self.loss_ce(pred, gt.long())

        pred = pred.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        return loss, acc * 100

    # @staticmethod
    # def cluster_patches(local_patches, n_cluster):
    #     # NOTE here squeeze is vital since KMeans expected dim(features) <= 2.
    #     features = local_patches.squeeze().cpu().numpy()
    #     # Initialize KMeans with 5 clusters
    #     kmeans = KMeans(n_clusters=n_cluster, n_init='auto', random_state=1,)
    #     # Fit the KMeans model to the feature data
    #     kmeans.fit(features)
    #     # Get the cluster centers
    #     cluster_centers = torch.from_numpy(kmeans.cluster_centers_).half().cuda()
    #
    #     return cluster_centers
    @staticmethod
    @torch.no_grad()
    def cluster_patches(features, n_cluster, n_iter=20, tol=1e-4, verbose=False, seed=None):
        """
        Batched KMeans clustering in PyTorch, GPU-friendly, close to sklearn results.

        Args:
            features: [B, N, D] float tensor
            n_cluster: number of clusters
            n_iter: max iterations
            tol: convergence threshold
            verbose: print loss if True
            seed: random seed for reproducibility

        Returns:
            centers: [B, n_cluster, D] cluster centers
            labels: [B, N] cluster assignments
        """
        if features.dim() == 2:
            features = features.unsqueeze(0)  # [1, N, D]

        B, N, D = features.shape
        device = features.device
        if seed is not None:
            torch.manual_seed(seed)

        # ===== kmeans++ 初始化 =====
        centers = torch.zeros(B, n_cluster, D, device=device, dtype=features.dtype)
        for b in range(B):
            # 第一个中心随机选
            indices = torch.randint(0, N, (1,), device=device)
            centers[b, 0] = features[b, indices]

            for k in range(1, n_cluster):
                dist = torch.cdist(features[b].unsqueeze(0), centers[b, :k].unsqueeze(0)).squeeze(0)
                min_dist2 = dist.min(dim=1)[0]
                probs = min_dist2 / min_dist2.sum()
                idx = torch.multinomial(probs, 1)
                centers[b, k] = features[b, idx]

        # ===== Lloyd 迭代 =====
        for it in range(n_iter):
            # 计算距离 [B, N, n_cluster]
            dist = torch.cdist(features, centers)
            labels = dist.argmin(dim=2)  # [B, N]

            # 更新中心
            new_centers = torch.zeros_like(centers)
            for b in range(B):
                for k in range(n_cluster):
                    mask = labels[b] == k
                    if mask.any():
                        new_centers[b, k] = features[b, mask].mean(dim=0)
                    else:
                        # 如果该 cluster 没样本，保持原中心
                        new_centers[b, k] = centers[b, k]

            shift = (new_centers - centers).norm(dim=2).max()
            centers = new_centers

            if verbose:
                print(f"Iteration {it}, max center shift: {shift.item():.6f}")

            if shift < tol:
                break

        return centers
    # @staticmethod
    # @torch.no_grad()
    # def cluster_patches(features, n_cluster, n_iter=10):
    #     """
    #     Fast GPU-based k-means clustering for batched data.
    #     Args:
    #         features: [B, N, D]
    #         n_cluster: number of clusters
    #         n_iter: iteration steps
    #     Returns:
    #         patch_centers: [B, n_cluster, D]
    #     """
    #     if features.dim() == 2:
    #         features = features.unsqueeze(0)  # [1, N, D]
    #
    #     B, N, D = features.shape
    #
    #     # 随机初始化中心 [B, n_cluster, D]
    #     idx = torch.randint(0, N, (B, n_cluster), device=features.device)
    #     centers = torch.gather(
    #         features, 1, idx.unsqueeze(-1).expand(-1, -1, D)
    #     ).clone()  # 随机选点作为初始中心
    #
    #     for _ in range(n_iter):
    #         # [B, N, n_cluster] -> pairwise distance
    #         dist = (
    #                 (features.unsqueeze(2) - centers.unsqueeze(1)) ** 2
    #         ).sum(dim=-1)  # L2 distance
    #
    #         # 每个点所属的聚类索引
    #         cluster_ids = dist.argmin(dim=-1)  # [B, N]
    #
    #         # 更新聚类中心
    #         new_centers = torch.zeros_like(centers)
    #         for b in range(B):
    #             for k in range(n_cluster):
    #                 mask = cluster_ids[b] == k
    #                 if mask.any():
    #                     new_centers[b, k] = features[b, mask].mean(dim=0)
    #                 else:
    #                     # 如果该cluster没有样本，则保持原中心
    #                     new_centers[b, k] = centers[b, k]
    #
    #         centers = new_centers
    #
    #     return centers  # [B, n_cluster, D]
    def forward(self, pts):
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts) #neighborhood:[num_groups,group_size,3],center:[num_groups,3]
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  # B G N #(B, num_group, encoder_dim)
        group_input_tokens = self.reduce_dim(group_input_tokens) #(B, num_group, trans_dim) trans_dim = pc_feat_dim // 2
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1) #(B, 1, trans_dim)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        pos = self.pos_embed(center) #(B, num_group, trans_dim)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1) #(B, 1 + num_group, trans_dim)
        pos = torch.cat((cls_pos, pos), dim=1) #(B, 1 + num_group, trans_dim)
        # transformer
        x = self.blocks(x, pos) #(B, 1 + num_group, trans_dim)
        x = self.norm(x) #(B, 1 + num_group, trans_dim)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1) #x[:,0] -> (B, trans_dim)->global feature, 后面一项也是(B, trans_dim)-> partial feature，concat了过后就是(B, 2 * trans_dim)

        if self.cache_type == 'global':
            return concat_f
        else:
            # (5, trans_dim)
            # patch_centers = self.__class__.cluster_patches(x[:, 1:], self.n_cluster)
            # # (5, trans_dim)
            # # print(patch_centers.size())
            # # cls_token = x[:, 0].repeat(self.n_cluster, 1)
            # # print(cls_token.size())
            # cls_token = x[:, 0].unsqueeze(1).repeat(self.n_cluster, 1)
            # # (5, 2*trans_dim)  NOTE concatenate the two tensors to match the required project dimension
            # # patch_centers = torch.cat([cls_token, patch_centers], dim=1)
            # patch_centers = torch.cat([cls_token, patch_centers], dim=2)



            # 聚类（假设返回 [B, n_cluster, D]）
            patch_centers = self.__class__.cluster_patches(x[:, 1:], self.n_cluster)
            if patch_centers.dim() == 2:  # 防止 batch_size=1 时掉维
                patch_centers = patch_centers.unsqueeze(0)  # [1, n_cluster, D]

            # [B, 1, D] -> [B, n_cluster, D]
            cls_token = x[:, 0].unsqueeze(1).repeat(1, self.n_cluster, 1)

            # 拼接 [B, n_cluster, 2D]
            patch_centers = torch.cat([cls_token, patch_centers], dim=-1)

            if self.cache_type == 'local':
                return patch_centers
            else:   # NOTE 'hierarchical' caches
                if self.cache_type == 'hierarchical':
                    return concat_f, patch_centers
                else:   # NOTE for visualization purpose
                    # (1, seq_len-1, emb_dim)
                    all_patches = x[:, 1:]
                    # (1, seq_len-1, emb_dim)
                    cls_token = x[:, 0].unsqueeze(1).repeat(1, all_patches.shape[1], 1)
                    # (1, seq_len-1, 2*emb_dim)
                    all_patches = torch.cat([cls_token, all_patches], dim=2)
                    return concat_f, all_patches, patch_centers
    def get_logits(self,pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N #(B, num_group, encoder_dim)
        group_input_tokens = self.reduce_dim(group_input_tokens)
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)  # (B, 1 + num_group, trans_dim)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = self.blocks(x, pos)  # (B, 1 + num_group, trans_dim)
        x = self.norm(x)
        patch_score = torch.einsum('bd,bnd->bn',x[:,0],x[:,1:])
        partial_pc_weighted = x[:, 1:] * torch.softmax(patch_score, dim=1).unsqueeze(-1)
        x_partial_mean = partial_pc_weighted.mean(dim=1)
        x_partial_var = partial_pc_weighted.var(dim=1,unbiased=False)
        epsilon = torch.randn_like(x_partial_var)
        x_partial = x_partial_mean + epsilon * x_partial_var.sqrt()
        concat_f = torch.cat([x[:,0],x_partial],dim=-1)
        patch_centers = self.__class__.cluster_patches(x[:, 1:], 5)
        # (5, trans_dim)
        cls_token = x[:, 0].repeat(5, 1)
        # (5, 2*trans_dim)  NOTE concatenate the two tensors to match the required project dimension
        patch_centers = torch.cat([cls_token, patch_centers], dim=1)
        return concat_f,patch_centers.mean(dim=0,keepdim=True)

