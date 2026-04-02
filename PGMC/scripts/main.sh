#!/bin/bash

gpu=$1
lm3d=$2         # uni3d, openshape, ulip
ckpt_path=$3    # weights/uni3d/lvis/model.pt, weights/uni3d/modelnet40/model.pt, weights/uni3d/scanobjnn/model.pt
                # weights/openshape/openshape-pointbert-vitg14-rgb/model.pt, weights/ulip/pointbert_ulip2.pt
dataset=$4     # modelnet_c, sonn_c, snv2_c, omniobject3d
sonn_variant=$5 # obj_only, obj_bg, hardest
cor_type=$6     # add_global_2, jitter_2
npoints=$7      # 1024/4096/16384 for `omniobject3d`
os_version=$8   # vitl14, vitg14 for `openshape`
ulip_version=$9 # ulip1, ulip2
cache_type=${10}  # 'global', 'local', 'hierarchical'
s2r_type=${11}  # so_obj_only_9', 'so_obj_only_11'

export CUDA_VISIBLE_DEVICES=${gpu}

if [ "$lm3d" = "uni3d" ]; then
    pc_feat_dim=1408
    num_group=512
    group_size=64
    pc_encoder_dim=512
    embed_dim=1024

    python runners/main2.py \
    --config configs \
    --wandb-log \
    --lm3d ${lm3d} \
    --cache-type ${cache_type} \
    --pc-feat-dim ${pc_feat_dim} \
    --num-group ${num_group} \
    --group-size ${group_size} \
    --pc-encoder-dim ${pc_encoder_dim} \
    --embed-dim ${embed_dim} \
    --ckpt_path ${ckpt_path} \
    --dataset ${dataset} \
    --sonn_variant ${sonn_variant} \
    --cor_type ${cor_type} \
    --sim2real_type ${s2r_type} \
    --npoints ${npoints}

elif [ "$lm3d" = "openshape" ]; then
    python runners/main2.py \
    --config configs \
    --wandb-log \
    --lm3d ${lm3d} \
    --cache-type ${cache_type} \
    --ckpt_path ${ckpt_path} \
    --dataset ${dataset} \
    --sonn_variant ${sonn_variant} \
    --cor_type ${cor_type} \
    --sim2real_type ${s2r_type} \
    --npoints ${npoints} \
    --oshape-version ${os_version}

elif [ "$lm3d" = "ulip" ]; then
    python runners/main2.py \
    --config configs \
    --wandb-log \
    --lm3d ${lm3d} \
    --cache-type ${cache_type} \
    --ckpt_path ${ckpt_path} \
    --dataset ${dataset} \
    --sonn_variant ${sonn_variant} \
    --cor_type ${cor_type} \
    --sim2real_type ${s2r_type} \
    --npoints ${npoints} \
    --ulip-version ${ulip_version}

else
    echo "The model does not match any of the supported ones."
fi