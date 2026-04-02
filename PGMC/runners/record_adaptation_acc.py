''' Hierarchical caches
        global + local
'''
import os
import sys
import wandb

import torch
import torch.nn.functional as F
import operator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *


@torch.no_grad()
def build_cache_in_advance(args, test_loader, lm3d_model, clip_weights, shot_capacity, include_prob_map=False):
    """Build cache with new features and loss, maintaining the maximum shot capacity."""
    '''
    NOTE 1. for positive cache, `include_prob_map=False`
                feature_loss = [pc_feats, loss]

         2. for negative cache, `include_prob_map=True`
                feature_loss = [pc_feats, loss, prob_map]
    '''
    # NOTE `cache` is empty initially
    if include_prob_map:
        print('*'*10, 'Building [global] neg. cache ...', '*'*10, '\n')
    else:
        print('*'*10, 'Building [global] and [local] pos. cache ...', '*'*10, '\n')
    cache, local_cache = {}, {}
    
    for pc, _, _, rgb in test_loader:
        # pc: (1, n, 3)     rgb: (1, n, 3)
        feature = torch.cat([pc, rgb], dim=-1).half()
        
        # `pred` indicates the class index
        # patch_centers: (5, emb_dim)
        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)
        
        item = [pc_feats, loss] if not include_prob_map else [pc_feats, loss, prob_map]
        local_item = [patch_centers, loss] if not include_prob_map else [patch_centers, loss, prob_map]
        
        if pred in cache:   
            if len(cache[pred]) < shot_capacity:
                # cache[pred] -> a list of: 
                #   [[feats, loss], ..., [feats, loss]] or [[feats, loss, prob_map], ..., [feats, loss, prob_map]]
                cache[pred].append(item)
                local_cache[pred].append(local_item)
        else:
            cache[pred] = [item]
            local_cache[pred] = [local_item]
            
        cache_num = sum([len(cache[key]) for key in cache])
        # clip_logits: (1, num_classes)
        num_classes = clip_logits.size(1)
        full_num = shot_capacity * num_classes
        
        # NOTE check if cache is full
        if cache_num == full_num:
            if include_prob_map:
                print('*'*10, 'Building [global] neg. cache is Done!', '*'*10, '\n')
            else:
                print('*'*10, 'Building [global] and [local] pos. cache is Done!', '*'*10, '\n')
            break
        
    return cache, local_cache
            

@torch.no_grad()
def update_cache(cache, local_cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    '''
    NOTE 1. for positive cache, `include_prob_map=False`
                feature_loss = [pc_feats, loss]

         2. for negative cache, `include_prob_map=True`
                feature_loss = [pc_feats, loss, prob_map]
    '''
    
    item = [features_loss[0]] + features_loss[2:] # pc_feats + loss
    local_item = [features_loss[1]] + features_loss[2:] # patch_centers + loss
    
    # NOTE `cache` is empty initially
    #   `pred` indicates the class index
    if pred in cache:
        if len(cache[pred]) < shot_capacity:
            # cache[pred] -> a list of: 
            #   [[feats, loss], ..., [feats, loss]] or [[feats, loss, prob_map], ..., [feats, loss, prob_map]]
            cache[pred].append(item)
            if not include_prob_map:
                local_cache[pred].append(local_item)
        elif features_loss[2] < cache[pred][-1][1]:
            # NOTE *** if the cache is full, and current `feats` has smaller entropy, \
            #   then replace the last item of cache with current `feats`
            cache[pred][-1] = item
            if not include_prob_map:
                local_cache[pred][-1] = local_item
                
        cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        if not include_prob_map:
            local_cache[pred] = sorted(local_cache[pred], key=operator.itemgetter(1))
        
    else:
        cache[pred] = [item]
        if not include_prob_map:
            local_cache[pred] = [local_item]


@torch.no_grad()
def compute_cache_logits(pc_feats, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    cache_keys = []
    cache_values = []
    for class_index in sorted(cache.keys()):
        for item in cache[class_index]:
            # item[0] -> `pc_feats` of shape (1, emb_dim)
            cache_keys.append(item[0])
            if neg_mask_thresholds:
                # NOTE *** for negative cache, item[2] -> `prob_map` is a class probability distribution
                cache_values.append(item[2])
            else:
                # NOTE class_index is an integer
                cache_values.append(class_index)

    # cache_keys: (emb_dim, n_cls * k_shot)
    cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)

    if neg_mask_thresholds:
        # (n_cls * k_shot, emb_dim)
        cache_values = torch.cat(cache_values, dim=0)
        cache_values = ((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).half().cuda()
    else:
        # NOTE 
        # 1. clip_weights: (emb_dim, n_cls)
        # 2. cache_values: (n_cls * k_shot, n_cls)
        cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).half().cuda()

    # affinity: (1, n_cls * k_shot)
    affinity = pc_feats @ cache_keys
    # cache_logits: (1, n_cls)
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits


@torch.no_grad()
def compute_local_cache_logits(patch_centers, local_cache, alpha, beta, clip_weights):
    """Compute logits using positive local cache."""
    local_cache_keys = []
    local_cache_values = []
    for class_index in sorted(local_cache.keys()):
        for item in local_cache[class_index]:
            # item[0] -> `patch_centers` of shape (5, emb_dim)
            local_cache_keys.append(item[0])
            n_cluster = item[0].shape[0]
            # NOTE ** class_index is an integer, each point cloud is clustered into **5** parts
            local_cache_values.append([class_index]*n_cluster)

    # local_cache_keys: (emb_dim, 5 * n_cls * k_shot)
    local_cache_keys = torch.cat(local_cache_keys, dim=0).permute(1, 0)

    # NOTE 
    # 1. clip_weights: (emb_dim, n_cls)
    # 2. local_cache_values: (5 * n_cls * k_shot, n_cls)
    local_cache_values = (F.one_hot(torch.Tensor(local_cache_values).to(torch.int64), num_classes=clip_weights.size(1))).half().cuda()
    local_cache_values = local_cache_values.view(-1, clip_weights.size(1))
    
    # patch_centers: (5, emb_dim)
    # affinity: (1, 5 * n_cls * k_shot)
    affinity = patch_centers.mean(dim=0, keepdim=True) @ local_cache_keys
    # local_cache_logits: (1, n_cls)
    local_cache_logits = ((-1) * (beta - beta * affinity)).exp() @ local_cache_values
    return alpha * local_cache_logits


@torch.no_grad()
def run_test_tda(args, pos_cfg, neg_cfg, test_loader, lm3d_model, clip_weights):
    ''' NOTE Build cache in advance '''
    pos_cache, pos_local_cache = build_cache_in_advance(args, test_loader, lm3d_model, clip_weights, pos_cfg['shot_capacity'])
    print('len(pos_cache):', len(pos_cache))
    print('len(pos_local_cache):', len(pos_local_cache))
    
    neg_cache, neg_local_cache = {}, {}
    
    zs_acc_li, global_acc_li, hier_acc_li = [], [], []
    
    #Unpack all hyperparameters
    pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
    if pos_enabled:
        pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
    if neg_enabled:
        neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}
        
    file_acc = open(f'outputs/{args.lm3d}_{args.dataset}_{args.sonn_variant}_{args.cor_type}_acc.log', 'w')

    #Test-time adaptation
    for i, (pc, target, _, rgb) in enumerate(test_loader):
        # pc: (1, n, 3)     rgb: (1, n, 3)
        feature = torch.cat([pc, rgb], dim=-1).half()

        # pc_feats: (1, emb_dim)
        # patch_centers: (5, emb_dim)
        # clip_logits: (1, n_cls)
        # loss: a scalar
        # prob_map: (1, n_cls)
        # pred: a scalar, class index
        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)

        # NOTE *** why compute `get_entropy()` ??? -> scale `loss` with `log_2(n_cls)`
        #   what's the difference between `get_entropy()` and `avg_entropy()` ???
        # target: (1, )     prop_entropy: (1, )
        target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights)
        
        zs_acc = cls_acc(clip_logits, target)
        zs_acc_li.append(zs_acc)
        file_acc.write(f'{i}\t{sum(zs_acc_li)/len(zs_acc_li)}\t')

        if pos_enabled:
            update_cache(pos_cache, pos_local_cache, pred, [pc_feats, patch_centers, loss], pos_params['shot_capacity'])

        if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
            update_cache(neg_cache, neg_local_cache, pred, [pc_feats, None, loss, prob_map], neg_params['shot_capacity'], True)

        final_logits = clip_logits.clone()
        if pos_enabled and pos_cache:
            final_logits += compute_cache_logits(pc_feats, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
            g_acc = cls_acc(final_logits, target)
            global_acc_li.append(g_acc)
            file_acc.write(f'{sum(global_acc_li)/len(global_acc_li)}\t')
            
            final_logits += compute_local_cache_logits(patch_centers, pos_local_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
        if neg_enabled and neg_cache:
            final_logits -= compute_cache_logits(pc_feats, neg_cache, neg_params['alpha'], neg_params['beta'], clip_weights, (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))
            
        h_acc = cls_acc(final_logits, target)  
        hier_acc_li.append(h_acc)
        file_acc.write(f'{sum(hier_acc_li)/len(hier_acc_li)}\n')

        if i % args.print_freq == 0:
            print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(hier_acc_li)/len(hier_acc_li)))
    
    file_acc.close()
    print("---- ***Final*** TDA's test accuracy: {:.2f}. ----\n".format(sum(hier_acc_li)/len(hier_acc_li)))


def main():
    args = get_arguments()
    # Set random seed
    set_random_seed(args.seed)
    
    # NOTE config
    config_path = args.config

    clip_model, lm3d_model = load_models(args)

    # NOTE *** need to be implemented
    preprocess = None

    # Run TDA on each dataset
    dataset_name = args.dataset
    print(f"Processing {dataset_name} dataset.")
    
    cfg = get_config_file(args, config_path, dataset_name)
    print("\nRunning dataset configurations:")
    print(cfg, "\n")
    
    test_loader, classnames, template = build_test_data_loader(args, dataset_name, args.data_root, preprocess)
    
    print(f'>>> classnames:', classnames)
    
    # `clip_weights` are text features of shape (emb_dim, n_cls)
    clip_weights = clip_classifier(args, classnames, template, clip_model)

    run_test_tda(args, cfg['positive'], cfg['negative'], test_loader, lm3d_model, clip_weights)


if __name__ == "__main__":
    main()