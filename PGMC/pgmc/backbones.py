"""Loading and uniform feature extraction for PGMC backbones."""

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn.functional as F


def _checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Backbone checkpoint not found: {}".format(checkpoint_path))
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "module", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value:
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported checkpoint structure in {}".format(checkpoint_path))
    return {key: value for key, value in checkpoint.items() if torch.is_tensor(value)}


def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str):
    return {
        (key[len(prefix) :] if key.startswith(prefix) else key): value
        for key, value in state.items()
    }


def _load_best_match(model: torch.nn.Module, state: Dict[str, torch.Tensor], label: str) -> None:
    target = model.state_dict()
    candidates = [state]
    for prefix in ("module.", "model.", "pc_encoder.", "backbone."):
        candidates.append(_strip_prefix(state, prefix))
    best = None
    best_count = -1
    for candidate in candidates:
        matched = {
            key: value
            for key, value in candidate.items()
            if key in target and target[key].shape == value.shape
        }
        if len(matched) > best_count:
            best, best_count = matched, len(matched)
    coverage = best_count / max(1, len(target))
    if coverage < 0.5:
        raise RuntimeError(
            "Only {:.1%} of {} parameters matched the checkpoint".format(coverage, label)
        )
    model.load_state_dict(best, strict=False)
    print("PGMC loaded {}: {}/{} tensors ({:.1%})".format(label, best_count, len(target), coverage))


class PGMCBackbone:
    def __init__(
        self,
        name: str,
        point_model: torch.nn.Module,
        text_model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.name = name
        self.point_model = point_model
        self.text_model = text_model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        for model in (self.point_model, self.text_model):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def encode_raw(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = features.to(device=self.device, dtype=self.dtype)
        xyz = features[..., :3].contiguous()
        if self.name == "ulip":
            output = self.point_model(xyz)
        elif self.name == "openshape":
            output = self.point_model(xyz, features)
        elif self.name == "uni3d":
            output = self.point_model.encode_pc(features)
        else:
            raise ValueError("Unknown PGMC backbone: {}".format(self.name))
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError(
                "PGMC requires hierarchical global/local output from {}".format(self.name)
            )
        return output[0], output[1]

    def encode(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        global_features, local_features = self.encode_raw(features)
        return F.normalize(global_features, dim=-1), F.normalize(local_features, dim=-1)

    def logits(self, features: torch.Tensor, text_weights: torch.Tensor, scale: float):
        global_features, _ = self.encode(features)
        return scale * (global_features @ text_weights)

    @torch.no_grad()
    def text_classifier(self, classnames: Iterable[str], prompts: Iterable[str]) -> torch.Tensor:
        weights = []
        for class_name in classnames:
            clean_name = class_name.replace("_", " ")
            texts = [prompt.format(clean_name) for prompt in prompts]
            tokens = self.tokenizer(texts).to(self.device)
            embeddings = self.text_model.encode_text(tokens)
            embeddings = F.normalize(embeddings, dim=-1)
            class_embedding = F.normalize(embeddings.mean(dim=0), dim=0)
            weights.append(class_embedding)
        return torch.stack(weights, dim=1).to(dtype=self.dtype)


def _load_openclip_text(model_name: str, checkpoint: str, device, dtype):
    import open_clip

    pretrained = checkpoint if checkpoint else None
    text_model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device="cpu"
    )
    text_model = text_model.to(device=device, dtype=dtype).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return text_model, tokenizer


def load_backbone(args) -> PGMCBackbone:
    args.cache_type = "hierarchical"
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("The bundled ULIP/OpenShape/Uni3D backbones require a CUDA device")
    dtype = torch.float16

    if args.backbone == "ulip":
        import clip
        from models import ulip

        point_model = ulip.create_ulip(args)
        point_state = _checkpoint_state(args.checkpoint)
        _load_best_match(point_model, point_state, "ULIP point encoder")

        text_model = ulip.create_clip_text_encoder(args)
        text_state = _checkpoint_state(args.text_checkpoint or args.checkpoint)
        _load_best_match(text_model, text_state, "ULIP text encoder")
        tokenizer = clip.tokenize
    elif args.backbone == "openshape":
        from omegaconf import OmegaConf
        from models import openshape

        config_path = Path(args.openshape_config).expanduser().resolve()
        config = OmegaConf.merge(OmegaConf.load(str(config_path)), vars(args))
        point_model = openshape.create_openshape(config)
        point_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(point_model)
        _load_best_match(
            point_model, _checkpoint_state(args.checkpoint), "OpenShape point encoder"
        )
        text_model, tokenizer = _load_openclip_text(
            args.openclip_model, args.text_checkpoint, device, dtype
        )
    elif args.backbone == "uni3d":
        from models import uni3d

        point_model = uni3d.create_uni3d(args)
        _load_best_match(
            point_model, _checkpoint_state(args.checkpoint), "Uni3D point encoder"
        )
        text_model, tokenizer = _load_openclip_text(
            args.clip_model, args.text_checkpoint, device, dtype
        )
    else:
        raise ValueError("Unsupported PGMC backbone: {}".format(args.backbone))

    point_model = point_model.to(device=device, dtype=dtype).eval()
    return PGMCBackbone(
        args.backbone, point_model, text_model, tokenizer, device, dtype
    )

