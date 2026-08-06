"""Uncertainty-managed PGMC global and local feature caches."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class CacheEntry:
    feature: torch.Tensor
    uncertainty: float
    probabilities: Optional[torch.Tensor]


class FeatureCache:
    def __init__(
        self,
        num_classes: int,
        capacity: int,
        kind: str = "global",
        negative: bool = False,
    ) -> None:
        if kind not in ("global", "local"):
            raise ValueError("Cache kind must be 'global' or 'local'")
        self.num_classes = num_classes
        self.capacity = capacity
        self.kind = kind
        self.negative = negative
        self.entries: Dict[int, List[CacheEntry]] = {}

    def __bool__(self) -> bool:
        return any(self.entries.values())

    def __len__(self) -> int:
        return sum(len(values) for values in self.entries.values())

    @torch.no_grad()
    def update(
        self,
        predictions: torch.Tensor,
        features: torch.Tensor,
        uncertainties: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None,
    ) -> None:
        predictions = predictions.reshape(-1)
        uncertainties = uncertainties.reshape(-1)
        if features.shape[0] != predictions.shape[0]:
            raise ValueError("Feature and prediction batch sizes differ")

        for index in range(predictions.shape[0]):
            class_index = int(predictions[index].item())
            feature = features[index].detach().clone()
            uncertainty = float(uncertainties[index].item())
            probability = None
            if probabilities is not None:
                probability = probabilities[index].detach().clone()
            item = CacheEntry(feature, uncertainty, probability)
            class_entries = self.entries.setdefault(class_index, [])
            class_entries.append(item)
            class_entries.sort(
                key=lambda entry: entry.uncertainty, reverse=self.negative
            )
            del class_entries[self.capacity :]

    def _materialize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        features = []
        values = []
        for class_index in sorted(self.entries):
            for item in self.entries[class_index]:
                feature = item.feature
                if self.kind == "local":
                    feature = feature.reshape(-1, feature.shape[-1])
                    repeat = feature.shape[0]
                else:
                    feature = feature.reshape(1, -1)
                    repeat = 1
                features.append(feature)
                if self.negative:
                    if item.probabilities is None:
                        raise ValueError("Negative cache entry is missing probabilities")
                    values.append(item.probabilities.reshape(1, -1).repeat(repeat, 1))
                else:
                    one_hot = F.one_hot(
                        torch.tensor(class_index, device=feature.device),
                        num_classes=self.num_classes,
                    ).to(feature.dtype)
                    values.append(one_hot.reshape(1, -1).repeat(repeat, 1))
        return torch.cat(features, dim=0), torch.cat(values, dim=0)

    @torch.no_grad()
    def logits(
        self,
        query: torch.Tensor,
        alpha: float,
        beta: float,
        mask_threshold: Tuple[float, float] = (0.03, 1.0),
    ) -> torch.Tensor:
        batch = query.shape[0]
        if not self:
            return torch.zeros(
                batch, self.num_classes, device=query.device, dtype=query.dtype
            )

        if query.ndim == 3:
            query = query.mean(dim=1)
        keys, values = self._materialize()
        keys = F.normalize(keys.to(device=query.device, dtype=query.dtype), dim=-1)
        values = values.to(device=query.device, dtype=query.dtype)
        if self.negative:
            lower, upper = mask_threshold
            values = ((values > lower) & (values < upper)).to(query.dtype)

        affinity = F.normalize(query, dim=-1) @ keys.transpose(0, 1)
        if self.kind == "global":
            weights = torch.exp(beta * affinity)
        else:
            weights = torch.exp(beta * (affinity - 1.0))
        return alpha * (weights @ values)


@dataclass
class PGMCMemory:
    global_positive: FeatureCache
    global_negative: FeatureCache
    reconstructed_positive: FeatureCache
    reconstructed_negative: FeatureCache
    local_positive: FeatureCache

    @classmethod
    def create(
        cls,
        num_classes: int,
        positive_capacity: int,
        negative_capacity: int,
    ) -> "PGMCMemory":
        return cls(
            global_positive=FeatureCache(num_classes, positive_capacity),
            global_negative=FeatureCache(
                num_classes, negative_capacity, negative=True
            ),
            reconstructed_positive=FeatureCache(num_classes, positive_capacity),
            reconstructed_negative=FeatureCache(
                num_classes, negative_capacity, negative=True
            ),
            local_positive=FeatureCache(
                num_classes, positive_capacity, kind="local"
            ),
        )


def evidential_uncertainty(logits: torch.Tensor):
    evidence = F.softplus(logits.float())
    concentration = (evidence + 1.0).sum(dim=1, keepdim=True)
    belief = evidence / concentration
    uncertainty = logits.shape[1] / concentration
    return belief, uncertainty.squeeze(1)

