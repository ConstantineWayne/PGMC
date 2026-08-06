"""Differentiable adversarial attacks for point clouds (XYZ only)."""

from typing import Callable

import torch
import torch.nn.functional as F


class PointCloudAttack:
    def __init__(
        self,
        name: str = "none",
        epsilon: float = 0.05,
        steps: int = 10,
        step_size: float = 0.01,
        random_start: bool = True,
    ) -> None:
        aliases = {"fgm": "fgm_l2", "pgd": "pgd_linf"}
        self.name = aliases.get(name, name)
        if self.name not in ("none", "fgsm", "fgm_l2", "pgd_linf", "pgd_l2"):
            raise ValueError("Unsupported adversarial attack: {}".format(name))
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.random_start = bool(random_start)

    @staticmethod
    def _l2_normalize(gradient: torch.Tensor) -> torch.Tensor:
        flat = gradient.reshape(gradient.shape[0], -1)
        norm = flat.norm(p=2, dim=1).clamp_min(1e-12)
        return gradient / norm.view(-1, 1, 1)

    def _project_l2(self, delta: torch.Tensor) -> torch.Tensor:
        flat = delta.reshape(delta.shape[0], -1)
        norm = flat.norm(p=2, dim=1).clamp_min(1e-12)
        scale = torch.minimum(torch.ones_like(norm), self.epsilon / norm)
        return delta * scale.view(-1, 1, 1)

    def __call__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        forward_logits: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if self.name == "none":
            return features.detach()

        original = features.detach()
        original_xyz = original[..., :3]
        rgb = original[..., 3:].detach()
        targets = targets.to(original.device, dtype=torch.long).reshape(-1)

        if self.name in ("fgsm", "fgm_l2"):
            xyz = original_xyz.clone().requires_grad_(True)
            logits = forward_logits(torch.cat([xyz, rgb], dim=-1))
            loss = F.cross_entropy(logits.float(), targets)
            gradient = torch.autograd.grad(loss, xyz, only_inputs=True)[0]
            if self.name == "fgsm":
                delta = self.epsilon * gradient.sign()
            else:
                delta = self.epsilon * self._l2_normalize(gradient)
            return torch.cat([(original_xyz + delta).detach(), rgb], dim=-1)

        if self.random_start:
            if self.name == "pgd_linf":
                delta = torch.empty_like(original_xyz).uniform_(
                    -self.epsilon, self.epsilon
                )
            else:
                delta = self._project_l2(torch.randn_like(original_xyz))
                radius = torch.rand(
                    original_xyz.shape[0], 1, 1, device=original_xyz.device
                )
                delta = delta * radius
        else:
            delta = torch.zeros_like(original_xyz)

        for _ in range(self.steps):
            xyz = (original_xyz + delta).detach().requires_grad_(True)
            logits = forward_logits(torch.cat([xyz, rgb], dim=-1))
            loss = F.cross_entropy(logits.float(), targets)
            gradient = torch.autograd.grad(loss, xyz, only_inputs=True)[0]
            if self.name == "pgd_linf":
                delta = delta + self.step_size * gradient.sign()
                delta = delta.clamp(-self.epsilon, self.epsilon)
            else:
                delta = delta + self.step_size * self._l2_normalize(gradient)
                delta = self._project_l2(delta)
        return torch.cat([(original_xyz + delta).detach(), rgb], dim=-1)

