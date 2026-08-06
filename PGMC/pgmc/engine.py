"""Training and evaluation engine for PGMC."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .adapter import PGMCFeatureAdapter
from .attacks import PointCloudAttack
from .cache import PGMCMemory, evidential_uncertainty
from .constants import TEXT_PROMPTS
from .data import PairedSourceDataset, build_source_dataset, build_target_dataset
from .fusion import fuse_pgmc_logits


LOGGER = logging.getLogger("PGMC")


def _features_from_points(points: torch.Tensor, device, dtype) -> torch.Tensor:
    points = points.to(device=device, dtype=dtype, non_blocking=True)
    rgb = torch.full_like(points, 0.4)
    return torch.cat([points, rgb], dim=-1)


def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probabilities = F.softmax(logits.float(), dim=1).clamp_min(1e-8)
    return -(probabilities * probabilities.log()).sum(dim=1).mean()


def _cache_scores(
    memory: PGMCMemory,
    global_query: torch.Tensor,
    reconstructed_query: torch.Tensor,
    local_query: torch.Tensor,
    config: Dict,
) -> Dict[str, torch.Tensor]:
    positive = config["cache"]["positive"]
    negative = config["cache"]["negative"]
    mask = tuple(negative["mask_threshold"])
    return {
        "global": memory.global_positive.logits(
            global_query, positive["alpha"], positive["beta"]
        )
        - memory.global_negative.logits(
            global_query, negative["alpha"], negative["beta"], mask
        ),
        "reconstructed": memory.reconstructed_positive.logits(
            reconstructed_query, positive["alpha"], positive["beta"]
        )
        - memory.reconstructed_negative.logits(
            reconstructed_query, negative["alpha"], negative["beta"], mask
        ),
        "local": memory.local_positive.logits(
            local_query, positive["alpha"], positive["beta"]
        ),
    }


def train_adapter(
    args,
    config: Dict,
    backbone,
    text_weights: torch.Tensor,
    source_loader: DataLoader,
) -> PGMCFeatureAdapter:
    embed_dim = int(text_weights.shape[0])
    adapter = PGMCFeatureAdapter(
        embed_dim,
        num_heads=int(config["adapter"]["num_heads"]),
        expansion=int(config["adapter"]["expansion"]),
        dropout=float(config["adapter"]["dropout"]),
    ).to(backbone.device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(config["adapter"]["learning_rate"]),
        weight_decay=float(config["adapter"]["weight_decay"]),
    )
    temperature = float(config["loss"]["temperature"])
    epochs = int(config["adapter"]["epochs"])
    logit_scale = float(config["logit_scale"])

    for epoch in range(epochs):
        adapter.train()
        total_loss = 0.0
        for clean_points, noisy_points, targets, _ in tqdm(
            source_loader, desc="PGMC source epoch {}/{}".format(epoch + 1, epochs)
        ):
            clean_features = _features_from_points(
                clean_points, backbone.device, backbone.dtype
            )
            noisy_features = _features_from_points(
                noisy_points, backbone.device, backbone.dtype
            )
            targets = targets.to(backbone.device, dtype=torch.long)
            with torch.no_grad():
                clean_global, _ = backbone.encode_raw(clean_features)
                noisy_global, _ = backbone.encode_raw(noisy_features)
                teacher_logits = logit_scale * (
                    F.normalize(clean_global, dim=-1) @ text_weights
                )

            reconstructed = adapter(noisy_global.float())
            student_logits = logit_scale * (
                F.normalize(reconstructed, dim=-1).to(text_weights.dtype)
                @ text_weights
            )
            reconstruction_loss = F.mse_loss(
                reconstructed, clean_global.float()
            )
            classification_loss = F.cross_entropy(student_logits.float(), targets)
            distillation_loss = F.kl_div(
                F.log_softmax(student_logits.float() / temperature, dim=1),
                F.softmax(teacher_logits.float() / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature ** 2)
            entropy_loss = _entropy_loss(student_logits)
            weights = config["loss"]
            loss = (
                float(weights["reconstruction"]) * reconstruction_loss
                + float(weights["classification"]) * classification_loss
                + float(weights["distillation"]) * distillation_loss
                + float(weights["entropy"]) * entropy_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        mean_loss = total_loss / max(1, len(source_loader))
        LOGGER.info("PGMC epoch %d/%d loss=%.6f", epoch + 1, epochs, mean_loss)
    return adapter.eval()


@torch.no_grad()
def build_source_memory(
    config: Dict,
    backbone,
    adapter: PGMCFeatureAdapter,
    text_weights: torch.Tensor,
    source_loader: DataLoader,
) -> PGMCMemory:
    cache_cfg = config["cache"]
    memory = PGMCMemory.create(
        text_weights.shape[1],
        int(cache_cfg["source_positive_capacity"]),
        int(cache_cfg["source_negative_capacity"]),
    )
    scale = float(config["logit_scale"])
    for clean_points, noisy_points, _, _ in tqdm(
        source_loader, desc="PGMC source memory"
    ):
        clean_input = _features_from_points(
            clean_points, backbone.device, backbone.dtype
        )
        noisy_input = _features_from_points(
            noisy_points, backbone.device, backbone.dtype
        )
        clean_raw, clean_local_raw = backbone.encode_raw(clean_input)
        noisy_raw, _ = backbone.encode_raw(noisy_input)
        clean_global = F.normalize(clean_raw, dim=-1)
        clean_local = F.normalize(clean_local_raw, dim=-1)
        reconstructed = F.normalize(adapter(noisy_raw.float()), dim=-1).to(
            text_weights.dtype
        )

        clean_logits = scale * (clean_global @ text_weights)
        reconstructed_logits = scale * (reconstructed @ text_weights)
        clean_probabilities = clean_logits.softmax(dim=1)
        reconstructed_probabilities = reconstructed_logits.softmax(dim=1)
        clean_predictions = clean_logits.argmax(dim=1)
        reconstructed_predictions = reconstructed_logits.argmax(dim=1)
        _, clean_uncertainty = evidential_uncertainty(clean_logits)
        _, reconstructed_uncertainty = evidential_uncertainty(
            reconstructed_logits
        )

        memory.global_positive.update(
            clean_predictions, clean_global, clean_uncertainty, clean_probabilities
        )
        memory.global_negative.update(
            clean_predictions, clean_global, clean_uncertainty, clean_probabilities
        )
        memory.reconstructed_positive.update(
            reconstructed_predictions,
            reconstructed,
            reconstructed_uncertainty,
            reconstructed_probabilities,
        )
        memory.reconstructed_negative.update(
            reconstructed_predictions,
            reconstructed,
            reconstructed_uncertainty,
            reconstructed_probabilities,
        )
        memory.local_positive.update(
            clean_predictions, clean_local, clean_uncertainty, clean_probabilities
        )
    return memory


def _apply_attack(
    attack: PointCloudAttack,
    features: torch.Tensor,
    targets: torch.Tensor,
    backbone,
    text_weights: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if attack.name == "none":
        return features
    with torch.enable_grad():
        return attack(
            features,
            targets,
            lambda attacked: backbone.logits(attacked, text_weights, scale),
        )


def build_target_memory(
    config: Dict,
    backbone,
    text_weights: torch.Tensor,
    target_loader: DataLoader,
    attack: PointCloudAttack,
) -> PGMCMemory:
    cache_cfg = config["cache"]
    initial_capacity = int(cache_cfg["initial_target_shots"])
    memory = PGMCMemory.create(
        text_weights.shape[1], initial_capacity, int(cache_cfg["target_negative_capacity"])
    )
    scale = float(config["logit_scale"])
    for points, targets, _ in tqdm(target_loader, desc="PGMC target memory"):
        features = _features_from_points(points, backbone.device, backbone.dtype)
        targets = targets.to(backbone.device, dtype=torch.long)
        features = _apply_attack(
            attack, features, targets, backbone, text_weights, scale
        )
        with torch.no_grad():
            global_features, local_features = backbone.encode(features)
            logits = scale * (global_features @ text_weights)
            probabilities = logits.softmax(dim=1)
            predictions = logits.argmax(dim=1)
            _, uncertainty = evidential_uncertainty(logits)
            memory.global_positive.update(
                predictions, global_features, uncertainty, probabilities
            )
            memory.local_positive.update(
                predictions, local_features, uncertainty, probabilities
            )

    expanded_capacity = int(cache_cfg["target_positive_capacity"])
    memory.global_positive.capacity = expanded_capacity
    memory.reconstructed_positive.capacity = expanded_capacity
    memory.local_positive.capacity = expanded_capacity
    return memory


def evaluate(
    args,
    config: Dict,
    backbone,
    adapter: PGMCFeatureAdapter,
    text_weights: torch.Tensor,
    source_memory: PGMCMemory,
    target_loader: DataLoader,
    attack: PointCloudAttack,
) -> Dict[str, float]:
    target_memory = build_target_memory(
        config, backbone, text_weights, target_loader, attack
    )
    scale = float(config["logit_scale"])
    correct_zero_shot = 0
    correct_pgmc = 0
    total = 0
    weights = config["fusion"]

    for step, (points, targets, _) in enumerate(
        tqdm(target_loader, desc="PGMC evaluation")
    ):
        features = _features_from_points(points, backbone.device, backbone.dtype)
        targets = targets.to(backbone.device, dtype=torch.long).reshape(-1)
        features = _apply_attack(
            attack, features, targets, backbone, text_weights, scale
        )
        with torch.no_grad():
            global_raw, local_raw = backbone.encode_raw(features)
            global_features = F.normalize(global_raw, dim=-1)
            local_features = F.normalize(local_raw, dim=-1)
            zero_shot = scale * (global_features @ text_weights)
            probabilities = zero_shot.softmax(dim=1)
            predictions = zero_shot.argmax(dim=1)
            _, uncertainty = evidential_uncertainty(zero_shot)

            reconstructed = F.normalize(adapter(global_raw.float()), dim=-1).to(
                text_weights.dtype
            )
            reconstructed_logits = scale * (reconstructed @ text_weights)
            reconstructed_probabilities = reconstructed_logits.softmax(dim=1)
            reconstructed_predictions = reconstructed_logits.argmax(dim=1)
            _, reconstructed_uncertainty = evidential_uncertainty(
                reconstructed_logits
            )

            target_memory.global_positive.update(
                predictions, global_features, uncertainty, probabilities
            )
            target_memory.global_negative.update(
                predictions, global_features, uncertainty, probabilities
            )
            target_memory.reconstructed_positive.update(
                reconstructed_predictions,
                reconstructed,
                reconstructed_uncertainty,
                reconstructed_probabilities,
            )
            target_memory.reconstructed_negative.update(
                reconstructed_predictions,
                reconstructed,
                reconstructed_uncertainty,
                reconstructed_probabilities,
            )
            target_memory.local_positive.update(
                predictions, local_features, uncertainty, probabilities
            )

            source_scores = _cache_scores(
                source_memory,
                global_features,
                reconstructed,
                local_features,
                config,
            )
            target_scores = _cache_scores(
                target_memory,
                global_features,
                reconstructed,
                local_features,
                config,
            )
            components = {
                "target_global": target_scores["global"],
                "target_reconstructed": target_scores["reconstructed"],
                "source_global": source_scores["global"],
                "source_reconstructed": source_scores["reconstructed"],
                "source_local": source_scores["local"],
                "target_local": target_scores["local"],
            }
            final_logits = fuse_pgmc_logits(zero_shot, components, weights)
            correct_zero_shot += int((predictions == targets).sum().item())
            correct_pgmc += int((final_logits.argmax(dim=1) == targets).sum().item())
            total += int(targets.numel())

        if args.print_freq > 0 and (step + 1) % args.print_freq == 0:
            LOGGER.info(
                "PGMC step=%d zero_shot=%.2f pgmc=%.2f",
                step + 1,
                100.0 * correct_zero_shot / max(1, total),
                100.0 * correct_pgmc / max(1, total),
            )

    return {
        "samples": total,
        "zero_shot_accuracy": 100.0 * correct_zero_shot / max(1, total),
        "pgmc_accuracy": 100.0 * correct_pgmc / max(1, total),
    }


def run_experiment(args, config: Dict, backbone, corruption: str) -> Dict:
    target_dataset = build_target_dataset(args, corruption)
    text_weights = backbone.text_classifier(
        target_dataset.classnames, TEXT_PROMPTS
    )
    source_dataset = build_source_dataset(args, target_dataset.classnames)
    if source_dataset is None:
        raise ValueError("--source-root is required for full PGMC evaluation")
    if args.benchmark == "sim2real" and not args.source_corruption:
        source_corruption = "jitter_2"
    elif args.source_corruption:
        source_corruption = args.source_corruption
    else:
        source_corruption = "dropout_global_2" if corruption == "clean" else corruption
    paired_source = PairedSourceDataset(
        source_dataset, source_corruption, seed=args.seed
    )
    source_loader = DataLoader(
        paired_source,
        batch_size=args.source_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    memory_loader = DataLoader(
        paired_source,
        batch_size=args.source_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    adapter = train_adapter(args, config, backbone, text_weights, source_loader)
    source_memory = build_source_memory(
        config, backbone, adapter, text_weights, memory_loader
    )
    attack = PointCloudAttack(
        args.attack,
        epsilon=args.attack_epsilon,
        steps=args.attack_steps,
        step_size=args.attack_step_size,
        random_start=not args.no_attack_random_start,
    )
    metrics = evaluate(
        args,
        config,
        backbone,
        adapter,
        text_weights,
        source_memory,
        target_loader,
        attack,
    )
    metrics.update(
        {
            "benchmark": args.benchmark,
            "corruption": corruption,
            "source_corruption": source_corruption,
            "backbone": args.backbone,
            "attack": args.attack,
        }
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = "{}_{}_{}_{}".format(
        args.benchmark, args.backbone, corruption, args.attack
    )
    torch.save(adapter.state_dict(), str(output_dir / "{}_adapter.pt".format(run_id)))
    (output_dir / "{}_metrics.json".format(run_id)).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics
