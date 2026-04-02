import os
import sys
import wandb
import logging
import torch
import torch.nn.functional as F
import operator
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.subnet import *

from utils.utils import *

wandb.init(mode="disabled")
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "train.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger()


def log_metrics(metrics, step=None):
    msg = f"Step {step}: " if step is not None else ""
    msg += ", ".join(f"{k}={v}" for k, v in metrics.items())
    logger.info(msg)
    print(msg)

@torch.no_grad()
def get_features(args,feat,lm3d_model):
    if isinstance(feat, list):
        feat = torch.cat(feat, dim=0).cuda()
    else:
        feat = feat.cuda()
    xyz = feat[:, :, :3]
    if args.cache_type == 'global':
        if args.lm3d == 'ulip':
            pc_feats = lm3d_model(xyz)
        elif args.lm3d == 'openshape':
            pc_feats = lm3d_model(xyz,feat)
        else:
            pc_feats = lm3d_model.encode_pc(feat)
        return pc_feats
    elif args.cache_type == 'local':
        if args.lm3d == 'ulip':
            patch_centers = lm3d_model(xyz)
        elif args.lm3d == 'openshape':
            patch_centers = lm3d_model(xyz,feat)
        else:
            patch_centers = lm3d_model.encode_pc(feat)
        return patch_centers
    else:
        if args.lm3d == 'ulip':
            pc_feats, patch_centers = lm3d_model(xyz)
        elif args.lm3d == 'openshape':
            pc_feats, patch_centers = lm3d_model(xyz,feat)
        else:
            pc_feats, patch_centers = lm3d_model.encode_pc(feat)
        return (pc_feats,patch_centers)

@torch.no_grad()
def direct_update(
    cache,
    preds,
    feats,
    uncertainties,
    belief_masses,
    prob_maps=None,
    shot_capacity=30,
    mode="positive",
    prt=None
):
    if isinstance(preds, int):
        preds = torch.tensor([preds], device=feats.device)
        feats = feats.unsqueeze(0)
        uncertainties = torch.tensor([uncertainties], device=feats.device)
        belief_masses = belief_masses.unsqueeze(0)
        if prob_maps is not None:
            prob_maps = prob_maps.unsqueeze(0)

    B = preds.shape[0]

    for i in range(B):
        pred = int(preds[i])
        feat = feats[i]
        belief_mass = belief_masses[i]
        if belief_mass.dim() == 1:
            belief_mass = belief_mass.unsqueeze(0)
        uncertainty = float(uncertainties[i])
        prob_map = prob_maps[i] if prob_maps is not None else None
        if prob_map is not None and prob_map.dim() == 1:
            prob_map = prob_map.unsqueeze(0)

        if feat.dim() == 1:
            item_feat = feat.unsqueeze(0)
        elif feat.dim() == 2:
            item_feat = feat
        else:
            item_feat = feat.squeeze(0)

        item = [item_feat, uncertainty, belief_mass, prob_map]

        if pred not in cache:
            cache[pred] = [item]
        else:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            else:
                uncertainties_cache = [it[1] for it in cache[pred]]

                if mode == "positive":
                    replace_idx = uncertainties_cache.index(max(uncertainties_cache))
                    if uncertainty < cache[pred][replace_idx][1]:
                        cache[pred][replace_idx] = item
                    cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
                elif mode == "negative":
                    replace_idx = uncertainties_cache.index(min(uncertainties_cache))
                    if uncertainty > cache[pred][replace_idx][1]:
                        cache[pred][replace_idx] = item
                    cache[pred] = sorted(cache[pred], key=operator.itemgetter(1), reverse=True)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

    if prt is not None:
        all_uncertainties = []
        for cls_items in cache.values():
            all_uncertainties.extend([item[1] for item in cls_items])
        if len(all_uncertainties) > 0:
            avg_uncertainty = sum(all_uncertainties) / len(all_uncertainties)
            if isinstance(avg_uncertainty, torch.Tensor):
                avg_uncertainty = avg_uncertainty.item()
        else:
            avg_uncertainty = 0.0

        print(f"[Cache] Current average uncertainty for {prt}: {avg_uncertainty:.4f}")

    return cache


def entropy_minimization_loss(clip_logits):
    clip_logits = clip_logits
    p = F.softmax(clip_logits, dim=1) + 1e-6
    entropy = -torch.sum(p * torch.log(p), dim=1)
    return entropy.mean()

def calculate_belief(logits):
    evidence = F.softplus(logits)
    alpha = evidence + 1
    concentration = torch.sum(alpha,dim=1,keepdim=True)
    belief_mass = evidence / concentration
    uncertainty = logits.size(1) / concentration
    return belief_mass, uncertainty

def test_time_training(args, dataloader, l3md_model, clip_weight, subnet, criterion, optimizer, num_epochs=8):
    l3md_model.eval()
    for p in l3md_model.parameters():
        p.requires_grad = False

    subnet = subnet.to(args.device)
    subnet.train()

    print(f"<<<<<<<<< Start Test-Time Training ({num_epochs} epochs) >>>>>>>>>")

    for epoch in range(num_epochs):
        print(f"\n===== Epoch [{epoch+1}/{num_epochs}] =====")
        epoch_loss = 0.0
        clean_cache, noisy_cache, recon_cache = {}, {}, {}
        clean_cache_neg, noisy_cache_neg, recon_cache_neg = {}, {}, {}
        clean_patch_cache, noisy_patch_cache, recon_patch_cache = {}, {}, {}

        for i, (clean_item, noisy_item, target) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}"):
            clean_pc, clean_rgb = clean_item[0], clean_item[1]
            noisy_pc, noisy_rgb = noisy_item[0], noisy_item[1]

            clean_features = torch.cat([clean_pc, clean_rgb], dim=-1).to(args.device).half()
            noisy_features = torch.cat([noisy_pc, noisy_rgb], dim=-1).to(args.device).half()

            with torch.no_grad():
                c_features = get_features(args, clean_features, l3md_model)  # (1, embed)
                n_features = get_features(args, noisy_features, l3md_model)

            c_feat,c_patch = c_features[0],c_features[1]
            n_feat,n_patch = n_features[0],n_features[1]

            # 子网络前向
            recon_feat = subnet(n_feat.float(),mode='point')
            r_feat = recon_feat / (recon_feat.norm(dim=-1, keepdim=True) + 1e-6)
            r_feat = r_feat.half()

            c_patch = c_patch / (c_patch.norm(dim=-1,keepdim=True) + 1e-6)

            clip_logits = 100. * r_feat @ clip_weight

            clean_feat = c_feat / (c_feat.norm(dim=-1, keepdim=True) + 1e-6)
            with torch.no_grad():
                teacher_logits = 100. * clean_feat @ clip_weight

            T = 2.0
            teacher_probs = F.softmax(teacher_logits / T, dim=1)
            student_log_probs = F.log_softmax(clip_logits / T, dim=1)
            loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

            cls_loss = F.cross_entropy(clip_logits.float(), target.long().cuda())
            loss_entropy = entropy_minimization_loss(clip_logits.float())
            loss = criterion(recon_feat, c_feat.float()) + cls_loss + loss_kd + loss_entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            c_feat /= (c_feat.norm(dim=-1, keepdim=True) + 1e-6)
            n_feat /= (n_feat.norm(dim=-1, keepdim=True) + 1e-6)
            recon_feat = recon_feat.half()
            recon_feat /= (recon_feat.norm(dim=-1, keepdim=True) + 1e-6)


            c_logits = 100 * c_feat @ clip_weight
            n_logits = 100 * n_feat @ clip_weight
            r_logits = 100 * recon_feat @ clip_weight

            c_prob, n_prob, r_prob = c_logits.softmax(dim=1), n_logits.softmax(dim=1), r_logits.softmax(dim=1)
            c_pred = c_logits.argmax(dim=1)
            n_pred = n_logits.argmax(dim=1)
            r_pred = r_logits.argmax(dim=1)

            c_belief, c_uncertainty = calculate_belief(c_logits)
            n_belief, n_uncertainty = calculate_belief(n_logits)
            r_belief, r_uncertainty = calculate_belief(r_logits)

            clean_cache = direct_update(clean_cache, c_pred, c_feat, c_uncertainty, c_belief, prob_maps=c_prob, shot_capacity=25)#25 #2 for ulip2_obj_bg "add_global"
            noisy_cache = direct_update(noisy_cache, n_pred, n_feat, n_uncertainty, n_belief, prob_maps=n_prob, shot_capacity=25)#25 #2 for ulip2_obj_bg "add_global"
            recon_cache = direct_update(recon_cache, r_pred, recon_feat, r_uncertainty, r_belief, prob_maps=r_prob, shot_capacity=25)#25 # 2 for ulip2_obj_bg "add_global"

            clean_patch_cache = direct_update(clean_patch_cache, c_pred, c_patch.half(), c_uncertainty, c_belief,
                                              prob_maps=c_prob, shot_capacity=25) #2 for ulip2_obj_bg "add_global"


            clean_cache_neg = direct_update(clean_cache_neg, c_pred, c_feat, c_uncertainty, c_belief, prob_maps=c_prob, shot_capacity=10, mode='negative') #1for ulip2_obj_bg "add_global"
            noisy_cache_neg = direct_update(noisy_cache_neg, n_pred, n_feat, n_uncertainty, n_belief, prob_maps=n_prob, shot_capacity=10, mode='negative') #1for ulip2_obj_bg "add_global"
            recon_cache_neg = direct_update(recon_cache_neg, r_pred, r_feat, r_uncertainty, r_belief, prob_maps=r_prob, shot_capacity=10, mode='negative')#1for ulip2_obj_bg "add_global"


        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

    print("<<<<<<<<< Test-Time Training Finished >>>>>>>>>")
    return subnet, [clean_cache, noisy_cache, recon_cache, clean_cache_neg, noisy_cache_neg, recon_cache_neg,clean_patch_cache,noisy_patch_cache,recon_patch_cache]


@torch.no_grad()
def cache_logits(pc_feats, cache, alpha, beta,mode='positive',neg_mask_thresholds=[0.03,1]):
    """Compute logits using positive/negative cache."""
    cache_keys = []
    cache_values = []
    for class_index in sorted(cache.keys()):
        for item in cache[class_index]:
            # item[0] -> `pc_feats` of shape (1, emb_dim)
            cache_keys.append(item[0])
            if mode == 'positive':
                cache_values.append(class_index)
            else:
                cache_values.append(item[-1])

    cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)

    if mode == 'positive':
        cache_values = (
            F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=40)).half().cuda()
    else:
        cache_values = torch.cat(cache_values, dim=0)
        cache_values = ((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).half().cuda()
    affinity = pc_feats @ cache_keys
    weight = torch.exp(beta * affinity)
    cache_logits = weight @ cache_values
    return alpha * cache_logits


@torch.no_grad()
def compute_local_cache_logits(patch_centers, local_cache, alpha, beta, num_classes=40,prob_map=False,neg_mask_thresholds=[0.03,1]):
    """Compute logits using positive local cache."""
    embed_dim = patch_centers.size(-1)
    patch_centers = patch_centers.view(-1,embed_dim)
    local_cache_keys = []
    local_cache_values = []
    for class_index in sorted(local_cache.keys()):
        for item in local_cache[class_index]:

            local_cache_keys.append(item[0])
            n_cluster = item[0].shape[0]
            if not prob_map:
                local_cache_values.append([class_index] * n_cluster)
            else:
                local_cache_values.append(item[-1])

    local_cache_keys = torch.cat(local_cache_keys, dim=0).permute(1, 0)

    if not prob_map:
        local_cache_values = (
            F.one_hot(torch.Tensor(local_cache_values).to(torch.int64), num_classes=num_classes)).half().cuda()
    else:

        local_cache_values = torch.cat(local_cache_values, dim=0)
        local_cache_values = local_cache_values.repeat(n_cluster,1)
        local_cache_values = ((local_cache_values > neg_mask_thresholds[0]) & (local_cache_values < neg_mask_thresholds[1])).half().cuda()
    local_cache_values = local_cache_values.view(-1, num_classes)

    affinity = patch_centers.mean(dim=0, keepdim=True) @ local_cache_keys
    local_cache_logits = ((-1) * (beta - beta * affinity)).exp() @ local_cache_values
    return alpha * local_cache_logits






@torch.no_grad()
def build_advance(args, test_loader, lm3d_model, clip_weights, shot_capacity,mode='point',include_prob_map=True):
    cache = {}

    for pc, _, _, rgb in tqdm(test_loader):
        feature = torch.cat([pc, rgb], dim=-1).half()
        pc_feats,patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights) #已经归一化了

        belief_mass,uncertainty = calculate_belief(clip_logits)

        if mode == 'point':
            item = [pc_feats, uncertainty,belief_mass,prob_map]
        elif mode == 'patch':
            item = [patch_centers.squeeze(0), uncertainty, belief_mass, prob_map]


        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
        else:
            cache[pred] = [item]

        cache_num = sum([len(cache[key]) for key in cache])
        num_classes = clip_logits.size(1)
        full_num = shot_capacity * num_classes

        if cache_num == full_num:
            if include_prob_map:
                print('*' * 10, 'Building neg. cache is Done!', '*' * 10, '\n')
            else:
                print('*' * 10, 'Building pos. cache is Done!', '*' * 10, '\n')
            break

    return cache

@torch.no_grad()
def run_test(args, pos_cfg, neg_cfg, test_loader, lm3d_model, clip_weights, recon_nn, train_caches):
    ''' NOTE Build cache in advance '''
    test_cache= build_advance(args,test_loader,lm3d_model,clip_weights,3)
    test_recon_cache = {}
    test_cache_n, test_recon_cache_n = {}, {}
    test_patch = build_advance(args,test_loader,lm3d_model,clip_weights,3,mode='patch')
    test_patch_n,test_recon_patch_n = {},{}

    recon_nn = recon_nn.cuda()
    recon_nn.eval()
    accuracies = []
    others = []
    train_clean_cache, train_noisy_cache, train_recon_cache = train_caches[0],train_caches[1],train_caches[2]
    train_clean_cache_n, train_noisy_cache_n, train_recon_cache_n = train_caches[3],train_caches[4],train_caches[5]
    train_patch_cache, train_noisy_patch_cache, train_recon_patch_cache = train_caches[6], train_caches[7], train_caches[8]

    pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
    if pos_enabled:
        pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
    if neg_enabled:
        neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}


    for i, (pc, target, _, rgb) in tqdm(enumerate(test_loader)):
        feature = torch.cat([pc, rgb], dim=-1).half()
        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights) #pc_feats已经归一化了
        test_pc = get_features(args,feature,lm3d_model)
        test_pc_feats, test_pc_patch = test_pc[0],test_pc[1]

        target = target.cuda()
        test_belief, test_uncertainty = calculate_belief(clip_logits)

        recon_feats = recon_nn(test_pc_feats.float())

        recon_feats = recon_feats / recon_feats.norm(dim=-1, keepdim=True)
        recon_feats = recon_feats.half()
        recon_logits = 100. * recon_feats @ clip_weights
        recon_prob = F.softmax(recon_logits,dim=1)
        recon_pred = int(recon_logits.topk(1, dim=1, largest=True, sorted=True)[1].t()[0])
        recon_belief, recon_uncertainty = calculate_belief(recon_logits)

        direct_update(test_cache,pred,pc_feats,test_uncertainty,test_belief,shot_capacity=25,prob_maps=prob_map,prt='test_cache') #25/10,,,2for ulip2_obj_bg "add_global"
        direct_update(test_recon_cache,recon_pred,recon_feats,recon_uncertainty,recon_belief,shot_capacity=25,prob_maps=recon_prob,prt='test_recon_cache') #25/10,,,,,,2for ulip2_obj_bg "add_global"
        direct_update(test_cache_n,pred,pc_feats,test_uncertainty,test_belief,prob_maps=prob_map,shot_capacity=10,mode='negative',prt='neg_test_cache') #25/10,,,,,,,,1for ulip2_obj_bg "add_global"
        direct_update(test_recon_cache_n, recon_pred, recon_feats, recon_uncertainty, recon_belief, prob_maps=recon_prob,shot_capacity=10,mode='negative',prt='neg_test_recon_cache') #1for ulip2_obj_bg "add_global"

        direct_update(test_patch,pred,patch_centers,test_uncertainty,test_belief,shot_capacity=25,prob_maps=prob_map,prt='test_patch_cache')#2for ulip2_obj_bg "add_global"

        direct_update(test_patch_n, pred, patch_centers, test_uncertainty, test_belief, shot_capacity=10,
                      prob_maps=prob_map,mode='negative') #1for ulip2_obj_bg "add_global"
            
        final_logits = clip_logits.clone()
        test_logits = cache_logits(pc_feats, test_cache, pos_params['alpha'], pos_params['beta'],mode='positive')
        test_logits_neg = cache_logits(pc_feats,test_cache_n,neg_params['alpha'],neg_params['beta'],mode='negative')

        test_logits = test_logits-test_logits_neg


        test_recon_logits = cache_logits(recon_feats, test_recon_cache, pos_params['alpha'], pos_params['beta'],mode='positive')
        test_recon_logits_neg = cache_logits(recon_feats, test_recon_cache_n, neg_params['alpha'], neg_params['beta'],mode='negative')

        test_recon_logits = test_recon_logits-test_recon_logits_neg


        train_logits = cache_logits(pc_feats,train_clean_cache,pos_params['alpha'], pos_params['beta'],mode='positive')
        train_logits_neg =  cache_logits(pc_feats,train_clean_cache_n,neg_params['alpha'], neg_params['beta'],mode='negative')

        train_logits = train_logits-train_logits_neg


        train_recon_logits = cache_logits(recon_feats,train_recon_cache,pos_params['alpha'], pos_params['beta'],mode='positive')
        train_recon_logits_neg = cache_logits(recon_feats, train_recon_cache_n, neg_params['alpha'], neg_params['beta'],mode='negative')

        train_recon_logits = train_recon_logits-train_recon_logits_neg
        train_patch_logits = compute_local_cache_logits(patch_centers,train_patch_cache,pos_params['alpha'], pos_params['beta'])
        test_patch_logits = compute_local_cache_logits(patch_centers,test_patch,pos_params['alpha'], pos_params['beta'])

        final_logits = test_recon_logits + clip_logits.clone() + test_logits + 2*train_logits + 2*train_recon_logits + train_patch_logits  + test_patch_logits
        acc = cls_acc(final_logits, target)
        accuracies.append(acc)
        log_metrics({"Averaged test accuracy": sum(accuracies) / len(accuracies)})
        log_metrics({"Averaged test accuracy for clean": sum(others) / len(others)})
        if i % args.print_freq == 0:
            print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))
    print("---- ***Final*** TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))
    return sum(accuracies) / len(accuracies)

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
    org_loader, _, _ = get_original_dataloader(args,intensity=1)
    print(f'>>> classnames:', classnames)
    if args.lm3d == 'openshape':
        embed_dim = 1280
    elif args.lm3d == 'uni3d':
        embed_dim = 1024
    else:
        embed_dim = 512
    subnet = ResidualTransformer(embed_dim=embed_dim)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(subnet.parameters(),lr=1e-4, weight_decay=1e-5)
    clip_weights = clip_classifier(args, classnames, template, clip_model)
    sub_nn,train_caches = test_time_training(args, org_loader, lm3d_model, clip_weights,subnet, criterion, optimizer)

    if args.lm3d == 'openshape':
        prefix = f"[test]/{args.cache_type}_cache/{args.lm3d}-{args.oshape_version}"
    elif args.lm3d == 'ulip':
        prefix = f"[test]/{args.cache_type}_cache/{args.ulip_version}"
    else:
        prefix = f"[test]/{args.cache_type}_cache/{args.lm3d}"

    if '_c' in dataset_name and 'sonn' in dataset_name:
        run_name = f"{prefix}/{dataset_name}-{args.sonn_variant}-{args.npoints}/{args.cor_type}"
    elif '_c' in dataset_name:
        run_name = f"{prefix}/{dataset_name}-{args.npoints}/{args.cor_type}"
    elif 'scanobjnn' in dataset_name or 'scanobjectnn' in dataset_name:
        run_name = f"{prefix}/{dataset_name}-{args.sonn_variant}-{args.npoints}"
    elif 'sim2real_sonn' in dataset_name:
        run_name = f"{prefix}/{dataset_name}-{args.sim2real_type}-{args.npoints}"
    elif 'pointda' in dataset_name:
        run_name = f"{prefix}/{dataset_name}-{args.npoints}"
    else:
        run_name = f"{prefix}/{dataset_name}-{args.npoints}"


    acc = run_test(args, cfg['positive'], cfg['negative'], test_loader, lm3d_model, clip_weights,sub_nn,train_caches)

    # 用 logger 打印/保存结果
    logger.info(f"{run_name} - Accuracy: {acc}")
    print(f"{run_name} - Accuracy: {acc}")


if __name__ == "__main__":
    main()