# PGMC

## Overview

In point-cloud representation learning, an encoder maps raw observations to high-dimensional feature manifolds and subsequently projects them onto low-dimensional semantic manifolds. Point-cloud corruptions disrupt this geometric consistency and induce manifold shifts that require robust calibration.

We propose **P**robabilistic **G**eometry-based **M**anifold **C**alibration (**PGMC**), a robust point-cloud representation framework that exploits geometric consistency on the probability simplex to dynamically restore distorted manifold mappings.

PGMC comprises three principal components:

1. The **Manifold Shift Degree Quantification** module maps distinct manifolds onto the probability simplex and selects reliable manifold anchors.
2. The **Hybrid Calibration Memory** module stores global discriminative anchors in a discrete memory while continuously refining local geometry.
3. The **Dual Manifold Calibration** strategy uses a geometry-preserved manifold to anchor the desired semantic structure and a shift-aware manifold to characterize distortion.

PGMC is evaluated with multiple pretrained models on challenging point-cloud corruption and domain-shift benchmarks. The experimental results demonstrate substantial improvements over existing state-of-the-art methods in the robustness of pretrained point-cloud representation models.

<img src="./e81143a259dd65d077b378c891ea697d.png" alt="Overview of PGMC">

## Framework

The Discrete Calibration Memory (DCM) and Continuous Calibration Memory (CCM) preserve discrete and continuous geometric semantics, respectively. The Geometry-Preserved Manifold (GPM) and Shift-Aware Manifold (SAM) provide complementary calibration spaces for refining the final manifold representation.

<img src="./5eeeaecab36b1d9bf16e71b807ae55b8.png" alt="PGMC framework">

## Supported Evaluations

The unified entry point, `main.py`, supports the following evaluations:

- ModelNet clean and ModelNet-C;
- ScanObjectNN clean and SONN-C;
- ShapeNet-to-ScanObjectNN-C Sim2Real transfer;
- FGSM, FGM-L2, PGD-L∞, and PGD-L2 adversarial attacks;
- ULIP-1, ULIP-2, OpenShape, and Uni3D backbones.

All commands in this document are executed from the project root:

```bash
cd PGMC
python main.py --help
```

The adversarial attack implementations are located in `pgmc/attacks.py`. The corresponding launcher is `scripts/run_attack.sh`.

## 1. Environment Setup

The reference environment uses Python 3.8.16, PyTorch 1.12.0, and CUDA 11.6.

```bash
conda env create -f environment.yaml
conda activate pgmc
```

If a compatible PyTorch and CUDA environment is already available, install the remaining dependencies with:

```bash
pip install -r requirements.txt
```

CUDA is required for the ULIP, OpenShape, and Uni3D backbones.

## 2. Pretrained Weights

The following pretrained weights are recommended for reproducing the reported experiments.

| Backbone component | Required file | Download source |
| --- | --- | --- |
| ULIP-1 point encoder | `pointbert_ulip1.pt` | [Point-PRC: ULIP-1](https://huggingface.co/datasets/auniquesun/Point-PRC/tree/main/pretrained-weights/ulip/point-encoder) |
| ULIP-2 point encoder | `pointbert_ulip2.pt` | [Point-PRC: ULIP-2](https://huggingface.co/datasets/auniquesun/Point-PRC/tree/main/pretrained-weights/ulip-2/point-encoder) |
| ULIP text encoder | `slip_base_100ep.pt` | [Point-PRC: ULIP text encoder](https://huggingface.co/datasets/auniquesun/Point-PRC/tree/main/pretrained-weights/ulip/image-text-encoder) |
| OpenShape point encoder | `model.pt` (`pointbert-vitg14-rgb`) | [OpenShape](https://huggingface.co/OpenShape/openshape-pointbert-vitg14-rgb/tree/main) |
| OpenShape text encoder | `open_clip_pytorch_model.bin` (`ViT-bigG-14`) | [LAION OpenCLIP](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/tree/main) |
| Uni3D point encoder | `model.pt` (`uni3d-g`) | [BAAI Uni3D](https://huggingface.co/BAAI/Uni3D/tree/main/modelzoo/uni3d-g) |
| Uni3D text encoder | `open_clip_pytorch_model.bin` (`EVA02-E-14-plus`) | [EVA02 OpenCLIP](https://huggingface.co/timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k/tree/main) |

The upstream implementations are available from the following repositories:

- [ULIP and ULIP-2](https://github.com/salesforce/ULIP)
- [OpenShape](https://github.com/Colin97/OpenShape_code)
- [Uni3D](https://github.com/baaivision/Uni3D)

The recommended weight layout is:

```text
PGMC/
`-- weights/
    |-- ulip/
    |   |-- pointbert_ulip1.pt
    |   |-- pointbert_ulip2.pt
    |   `-- slip_base_100ep.pt
    |-- openshape/
    |   |-- model.pt
    |   `-- open_clip_pytorch_model.bin
    `-- uni3d/
        |-- model.pt
        `-- open_clip_pytorch_model.bin
```

## 3. Datasets

PGMC requires a target benchmark for evaluation and a clean source dataset for training the feature-reconstruction adapter and constructing the source memory.

| Purpose | Dataset | Download source | Access information |
| --- | --- | --- | --- |
| ModelNet-C target | `modelnet_c`, including `clean.h5` and corruption H5 files | [Point-PRC ModelNet-C](https://huggingface.co/datasets/auniquesun/Point-PRC/tree/main/new-3ddg-benchmarks/xset/corruption/modelnet_c) | Direct download |
| SONN-C target | `sonn_c/{obj_only,obj_bg,hardest}` | [Point-PRC SONN-C](https://huggingface.co/datasets/auniquesun/Point-PRC/tree/main/new-3ddg-benchmarks/xset/corruption/sonn_c) | Direct download |
| ModelNet40 source | ModelNet40 HDF5 | [PointNet repository](https://github.com/charlesq34/pointnet) or [HDF5 archive](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) | Direct download |
| ScanObjectNN source | Processed as `obj_only/obj_bg/hardest/{trainminusval.h5,valid.h5}` | [PGMC_data on Baidu Netdisk](https://pan.baidu.com/s/1NNsqEkLJ5lFNde6QUlVVyQ?pwd=1222) | Extraction code: `1222` |
| Sim2Real target | `so_obj_only_9`, `so_obj_bg_9`, and `so_hardest_9` | [PGMC_data on Baidu Netdisk](https://pan.baidu.com/s/1NNsqEkLJ5lFNde6QUlVVyQ?pwd=1222) | Extraction code: `1222` |
| Sim2Real source | `shapenet_9` | [PGMC_data on Baidu Netdisk](https://pan.baidu.com/s/1NNsqEkLJ5lFNde6QUlVVyQ?pwd=1222) | Extraction code: `1222` |

The `PGMC_data` package contains the processed ScanObjectNN source data, the Sim2Real target data, and the ShapeNet source data used by PGMC. The Sim2Real evaluation uses ShapeNet as the source domain and does not use ModelNet as a source domain.

## 4. Data Layout

The recommended directory structure is:

```text
PGMC/
`-- data/
    |-- modelnet_c/
    |   |-- shape_names.txt
    |   |-- clean.h5
    |   |-- add_global_2.h5
    |   |-- add_local_2.h5
    |   |-- dropout_global_2.h5
    |   |-- dropout_local_2.h5
    |   |-- rotate_2.h5
    |   |-- scale_2.h5
    |   `-- jitter_2.h5
    |-- modelnet40/
    |   |-- shape_names.txt
    |   `-- ply_data_*.h5
    |-- sonn_c/
    |   |-- shape_names.txt
    |   |-- obj_only/{clean.h5,*_2.h5}
    |   |-- obj_bg/{clean.h5,*_2.h5}
    |   `-- hardest/{clean.h5,*_2.h5}
    |-- scanobjectnn/
    |   |-- obj_only/{trainminusval.h5,valid.h5}
    |   |-- obj_bg/{trainminusval.h5,valid.h5}
    |   `-- hardest/{trainminusval.h5,valid.h5}
    `-- sim2real/
        |-- so_obj_only_9/
        |-- so_obj_bg_9/
        |-- so_hardest_9/
        `-- shapenet_9/
```

Each H5 file must contain at least the following datasets:

```text
data:  [N, P, 3] or [N, P, C]
label: [N] or [N, 1]
```

## 5. Clean Evaluation

Clean evaluation reads `clean.h5` from the selected target dataset.

### 5.1 ModelNet Clean with ULIP-2

```bash
python main.py \
  --benchmark modelnet_c \
  --corruption clean \
  --backbone ulip \
  --ulip-version ulip2 \
  --checkpoint weights/ulip/pointbert_ulip2.pt \
  --text-checkpoint weights/ulip/slip_base_100ep.pt \
  --data-root data/modelnet_c \
  --source-root data/modelnet40 \
  --device cuda:0
```

For a clean target, PGMC uses `dropout_global_2` by default to construct the clean/noisy source pairs. To retain a clean source in both elements of each pair, specify:

```bash
--source-corruption clean
```

### 5.2 ScanObjectNN Clean with ULIP-2

```bash
python main.py \
  --benchmark sonn_c \
  --corruption clean \
  --sonn-variant obj_only \
  --backbone ulip \
  --ulip-version ulip2 \
  --checkpoint weights/ulip/pointbert_ulip2.pt \
  --text-checkpoint weights/ulip/slip_base_100ep.pt \
  --data-root data/sonn_c \
  --source-root data/scanobjectnn \
  --device cuda:0
```

The valid values of `--sonn-variant` are `obj_only`, `obj_bg`, and `hardest`.

## 6. Corruption Evaluation

PGMC evaluates the following seven corruption types at severity level 2:

```text
add_global_2
add_local_2
dropout_global_2
dropout_local_2
rotate_2
scale_2
jitter_2
```

### 6.1 Single ModelNet-C Corruption

```bash
python main.py \
  --benchmark modelnet_c \
  --corruption jitter_2 \
  --backbone ulip \
  --ulip-version ulip2 \
  --checkpoint weights/ulip/pointbert_ulip2.pt \
  --text-checkpoint weights/ulip/slip_base_100ep.pt \
  --data-root data/modelnet_c \
  --source-root data/modelnet40 \
  --device cuda:0
```

### 6.2 All ModelNet-C Corruptions

```bash
python main.py \
  --benchmark modelnet_c \
  --all-corruptions \
  --backbone ulip \
  --ulip-version ulip2 \
  --checkpoint weights/ulip/pointbert_ulip2.pt \
  --text-checkpoint weights/ulip/slip_base_100ep.pt \
  --data-root data/modelnet_c \
  --source-root data/modelnet40 \
  --device cuda:0
```

The `--all-corruptions` option evaluates the seven severity-2 corruptions and excludes the clean split.

### 6.3 SONN-C

```bash
python main.py \
  --benchmark sonn_c \
  --sonn-variant hardest \
  --corruption rotate_2 \
  --backbone openshape \
  --checkpoint weights/openshape/model.pt \
  --text-checkpoint weights/openshape/open_clip_pytorch_model.bin \
  --data-root data/sonn_c \
  --source-root data/scanobjectnn \
  --device cuda:0
```

Replacing `--corruption rotate_2` with `--all-corruptions` evaluates all seven corruptions for the selected SONN-C variant.

## 7. Sim2Real Evaluation

The Sim2Real protocol uses ShapeNet as the source domain and ScanObjectNN-C as the target domain. It comprises three transfer evaluations:

- ShapeNet → ScanObjectNN-C (`Obj_Only`);
- ShapeNet → ScanObjectNN-C (`Obj_BG`);
- ShapeNet → ScanObjectNN-C (`Hardest`).

Target-domain labels are excluded from test-time adaptation and are accessed only when calculating the final accuracy. The data loader supports either H5 files or class-organized `.npy` files.

Example H5 target layout:

```text
data/sim2real/so_obj_only_9/
|-- clean.h5
`-- shape_names.txt
```

Example class-folder target layout:

```text
data/sim2real/so_obj_only_9/
|-- bed/test/*.npy
|-- chair/test/*.npy
`-- ...
```

Example evaluation command:

```bash
python main.py \
  --benchmark sim2real \
  --sim2real-type so_obj_only_9 \
  --backbone ulip \
  --ulip-version ulip2 \
  --checkpoint weights/ulip/pointbert_ulip2.pt \
  --text-checkpoint weights/ulip/slip_base_100ep.pt \
  --data-root data/sim2real \
  --source-root data/sim2real \
  --source-domain shapenet_9 \
  --source-corruption jitter_2 \
  --device cuda:0
```

The valid values of `--sim2real-type` are `so_obj_only_9`, `so_obj_bg_9`, and `so_hardest_9`. The default source domain and source corruption are `shapenet_9` and `jitter_2`, respectively; the corresponding command-line arguments may therefore be omitted.

The `_9` suffix is retained for compatibility with the original data-package naming convention. PGMC does not infer the number of classes from this suffix. For a class-folder dataset, class names are determined from the directory names. For an H5 dataset, class names are read from `shape_names.txt` or `classnames.txt`. This procedure maintains a consistent label space across the source and target domains.

## 8. Adversarial Evaluation

The adversarial attack implementations are provided in `pgmc/attacks.py`. Supported attacks are:

```text
fgsm
fgm_l2
pgd_linf
pgd_l2
```

The attacks perturb only the XYZ coordinates and do not modify RGB features.

### 8.1 FGSM

Append the following arguments to any clean, corruption, or Sim2Real command:

```bash
--attack fgsm \
--attack-epsilon 0.05
```

### 8.2 PGD-L∞

```bash
--attack pgd_linf \
--attack-epsilon 0.05 \
--attack-steps 10 \
--attack-step-size 0.01
```

Complete example:

```bash
python main.py \
  --benchmark modelnet_c \
  --corruption jitter_2 \
  --backbone ulip \
  --ulip-version ulip2 \
  --checkpoint weights/ulip/pointbert_ulip2.pt \
  --text-checkpoint weights/ulip/slip_base_100ep.pt \
  --data-root data/modelnet_c \
  --source-root data/modelnet40 \
  --attack pgd_linf \
  --attack-epsilon 0.05 \
  --attack-steps 10 \
  --attack-step-size 0.01 \
  --device cuda:0
```

## 9. Backbone Selection

The dataset and PGMC configuration remain unchanged when the backbone is replaced. Only the backbone and checkpoint arguments require modification.

### 9.1 ULIP-1

```bash
--backbone ulip \
--ulip-version ulip1 \
--checkpoint weights/ulip/pointbert_ulip1.pt \
--text-checkpoint weights/ulip/slip_base_100ep.pt
```

### 9.2 ULIP-2

```bash
--backbone ulip \
--ulip-version ulip2 \
--checkpoint weights/ulip/pointbert_ulip2.pt \
--text-checkpoint weights/ulip/slip_base_100ep.pt
```

### 9.3 OpenShape

```bash
--backbone openshape \
--checkpoint weights/openshape/model.pt \
--text-checkpoint weights/openshape/open_clip_pytorch_model.bin \
--openclip-model ViT-bigG-14
```

### 9.4 Uni3D

```bash
--backbone uni3d \
--checkpoint weights/uni3d/model.pt \
--text-checkpoint weights/uni3d/open_clip_pytorch_model.bin \
--pc-model eva_giant_patch14_560 \
--clip-model EVA02-E-14-plus
```

## 10. Configuration and Hyperparameters

PGMC uses a single configuration file:

```text
configs/pgmc.yaml
```

Shared settings are defined once, while benchmark-specific overrides are placed under the `benchmarks` section. This structure avoids duplicated configuration files and ensures consistent defaults across ModelNet-C, SONN-C, and Sim2Real evaluations.

### 10.1 Memory Capacity

```yaml
cache:
  initial_target_shots: 3
  source_positive_capacity: 25
  source_negative_capacity: 10
  target_positive_capacity: 25
  target_negative_capacity: 10
```

Each capacity denotes the maximum number of entries stored per predicted class. Increasing a capacity generally increases both GPU memory consumption and execution time.

### 10.2 Memory Logits

```yaml
cache:
  positive:
    alpha: 2.0
    beta: 3.0
  negative:
    alpha: 0.117
    beta: 1.0
    mask_threshold: [0.03, 1.0]
```

- `alpha` controls the overall contribution of the corresponding memory logits.
- `beta` controls the sharpness of the similarity-based weighting.
- `mask_threshold` defines the probability interval used by the negative-memory mask.

### 10.3 Reconstruction Adapter

```yaml
adapter:
  epochs: 8
  learning_rate: 0.0001
  weight_decay: 0.00001
  num_heads: 8
  expansion: 4
  dropout: 0.2
```

The number of adapter-training epochs may be overridden for an individual run with `--epochs`, for example `--epochs 12`.

### 10.4 Logit Fusion

```yaml
fusion:
  zero_shot: 1.0
  target_global: 1.0
  target_reconstructed: 1.0
  source_global: 2.0
  source_reconstructed: 2.0
  source_local: 1.0
  target_local: 1.0
```

Each value specifies the weight assigned to the corresponding logit component. For ablation experiments, a component may be disabled by assigning it a weight of `0.0`; no Python modification is required.

For separate experiments, create a copy of the default configuration and pass it through `--config`:

```bash
cp configs/pgmc.yaml configs/pgmc_custom.yaml
python main.py ... --config configs/pgmc_custom.yaml
```

## 11. Outputs

By default, results are written to `outputs/`:

```text
pgmc.log
<run_id>_adapter.pt
<run_id>_metrics.json
pgmc_summary.json
```

An alternative output directory may be specified with:

```bash
python main.py ... --output-dir outputs/ulip2_modelnet_c
```
