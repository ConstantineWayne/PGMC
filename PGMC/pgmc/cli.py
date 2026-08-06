"""Command-line interface for PGMC."""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

from .backbones import load_backbone
from .config import load_pgmc_config
from .constants import CORRUPTIONS_SEVERITY_2, SIM2REAL_TYPES
from .data import build_source_dataset, build_target_dataset
from .engine import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PGMC robustness evaluation for severity-2 corruptions, Sim2Real, and adversarial attacks"
    )
    parser.add_argument(
        "--benchmark", required=True, choices=("modelnet_c", "sonn_c", "sim2real")
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-domain", default="")
    parser.add_argument("--backbone", required=True, choices=("ulip", "openshape", "uni3d"))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--text-checkpoint", default="")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "pgmc.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))

    parser.add_argument(
        "--corruption",
        default="add_global_2",
        choices=("clean",) + CORRUPTIONS_SEVERITY_2,
    )
    parser.add_argument("--all-corruptions", action="store_true")
    parser.add_argument(
        "--source-corruption", default=None, choices=("clean",) + CORRUPTIONS_SEVERITY_2
    )
    parser.add_argument(
        "--sonn-variant", default="obj_only", choices=("obj_only", "obj_bg", "hardest")
    )
    parser.add_argument(
        "--sim2real-type", default="so_obj_only_9", choices=SIM2REAL_TYPES
    )
    parser.add_argument(
        "--sim2real-corruption",
        default="clean",
        choices=("clean",) + CORRUPTIONS_SEVERITY_2,
    )
    parser.add_argument("--npoints", type=int, default=1024)

    parser.add_argument(
        "--attack",
        default="none",
        choices=("none", "fgsm", "fgm_l2", "pgd_linf", "pgd_l2"),
    )
    parser.add_argument("--attack-epsilon", type=float, default=0.05)
    parser.add_argument("--attack-steps", type=int, default=10)
    parser.add_argument("--attack-step-size", type=float, default=0.01)
    parser.add_argument("--no-attack-random-start", action="store_true")

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--source-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--check-data", action="store_true")

    parser.add_argument("--cache-type", default="hierarchical")
    parser.add_argument("--n-cluster", type=int, default=5)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--pc-model", default="eva_giant_patch14_560")
    parser.add_argument("--pretrained-pc", default="")
    parser.add_argument("--clip-model", default="EVA02-E-14-plus")
    parser.add_argument("--openclip-model", default="ViT-bigG-14")
    parser.add_argument(
        "--openshape-config",
        default=str(PROJECT_ROOT / "models" / "openshape" / "config.yaml"),
    )
    parser.add_argument("--pc-feat-dim", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--num-group", type=int, default=512)
    parser.add_argument("--pc-encoder-dim", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--patch-dropout", type=float, default=0.0)
    parser.add_argument("--pc-depth", type=int, default=12)
    parser.add_argument("--num-head", type=int, default=6)
    parser.add_argument("--encoder-dim", type=int, default=256)
    parser.add_argument("--ulip-version", choices=("ulip1", "ulip2"), default="ulip2")
    return parser


def _finalize_model_defaults(args) -> None:
    if args.backbone == "uni3d":
        args.pc_feat_dim = args.pc_feat_dim or 1408
        args.group_size = args.group_size or 64
        args.embed_dim = args.embed_dim or 1024
    elif args.backbone == "ulip":
        args.pc_feat_dim = args.pc_feat_dim or 768
        args.group_size = args.group_size or 32
        args.embed_dim = args.embed_dim or 512
    else:
        args.pc_feat_dim = args.pc_feat_dim or 1280
        args.group_size = args.group_size or 32
        args.embed_dim = args.embed_dim or 1280
    if args.benchmark == "sim2real" and not args.source_domain:
        args.source_domain = "shapenet_9"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _check_data(args, corruptions) -> None:
    summary = []
    for corruption in corruptions:
        target = build_target_dataset(args, corruption)
        source = build_source_dataset(args, target.classnames)
        summary.append(
            {
                "corruption": corruption,
                "target_samples": len(target),
                "source_samples": len(source),
                "classes": len(target.classnames),
            }
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _finalize_model_defaults(args)
    if args.all_corruptions and args.benchmark == "sim2real":
        parser.error("--all-corruptions applies only to ModelNet-C and SONN-C")
    if args.benchmark == "sim2real":
        corruptions = [args.sim2real_corruption]
    else:
        corruptions = (
            list(CORRUPTIONS_SEVERITY_2)
            if args.all_corruptions
            else [args.corruption]
        )
    if not args.check_data and not args.checkpoint:
        parser.error("--checkpoint is required unless --check-data is used")
    if (
        not args.check_data
        and args.backbone in ("openshape", "uni3d")
        and not args.text_checkpoint
    ):
        parser.error("--text-checkpoint is required for OpenShape and Uni3D")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=(
            logging.StreamHandler(),
            logging.FileHandler(str(output_dir / "pgmc.log"), encoding="utf-8"),
        ),
    )
    _seed_everything(args.seed)
    if args.check_data:
        _check_data(args, corruptions)
        return

    config = load_pgmc_config(args.config, args.benchmark)
    if args.epochs is not None:
        config["adapter"]["epochs"] = args.epochs
    backbone = load_backbone(args)
    results = []
    for corruption in corruptions:
        logging.getLogger("PGMC").info(
            "Starting PGMC benchmark=%s corruption=%s attack=%s",
            args.benchmark,
            corruption,
            args.attack,
        )
        results.append(run_experiment(args, config, backbone, corruption))

    summary = {
        "runs": results,
        "mean_zero_shot_accuracy": sum(
            item["zero_shot_accuracy"] for item in results
        )
        / len(results),
        "mean_pgmc_accuracy": sum(item["pgmc_accuracy"] for item in results)
        / len(results),
    }
    summary_path = output_dir / "pgmc_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
