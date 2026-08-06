"""The seven atomic PointCloud-C corruptions used by PGMC source pairs."""

import math
import re
from typing import Callable, Dict, Tuple

import numpy as np

from .constants import CORRUPTION_ALIASES, CORRUPTION_BASES, DEFAULT_SEVERITY


def normalize_unit_sphere(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=1).max(initial=0.0)
    if radius < 1e-8:
        return np.zeros_like(centered, dtype=np.float32)
    return (centered / radius).astype(np.float32)


def resample_points(points: np.ndarray, npoints: int, rng=None) -> np.ndarray:
    """Return exactly ``npoints`` while retaining every point when possible."""
    rng = np.random if rng is None else rng
    points = np.asarray(points, dtype=np.float32)
    count = points.shape[0]
    if count == 0:
        raise ValueError("Cannot resample an empty point cloud")
    if count == npoints:
        return points.copy()
    if count > npoints:
        indices = rng.choice(count, npoints, replace=False)
    else:
        extra = rng.choice(count, npoints - count, replace=True)
        indices = np.concatenate([np.arange(count), extra])
    return points[indices].astype(np.float32, copy=False)


def parse_corruption(name: str, default_severity: int = DEFAULT_SEVERITY) -> Tuple[str, int]:
    value = name.strip().lower().replace("-", "_")
    if value == "clean":
        return "clean", default_severity
    match = re.fullmatch(r"(.+)_([0-4])", value)
    if match:
        base, severity_text = match.groups()
        severity = int(severity_text)
    else:
        base, severity = value, default_severity
    base = CORRUPTION_ALIASES.get(base, base)
    if base not in CORRUPTION_BASES:
        raise ValueError(
            "Unknown corruption '{}'. Expected one of: {}".format(
                name, ", ".join(CORRUPTION_BASES)
            )
        )
    return base, severity


def canonical_corruption(name: str, severity: int = DEFAULT_SEVERITY) -> str:
    base, parsed_severity = parse_corruption(name, severity)
    if base == "clean":
        return "clean"
    return "{}_{}".format(base, parsed_severity)


def _shuffle(points: np.ndarray, rng) -> np.ndarray:
    return points[rng.permutation(points.shape[0])]


def _cluster_sizes(num_clusters: int, total_size: int, rng) -> np.ndarray:
    assignments = rng.randint(0, num_clusters, size=total_size)
    return np.bincount(assignments, minlength=num_clusters)


def _sample_unit_sphere(count: int, rng) -> np.ndarray:
    radius = np.power(rng.uniform(0.0, 1.0, size=(count, 1)), 1.0 / 3.0)
    cos_theta = rng.uniform(-1.0, 1.0, size=(count, 1))
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(count, 1))
    xyz = np.concatenate(
        [
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta),
        ],
        axis=1,
    )
    return xyz.astype(np.float32)


def corrupt_scale(points: np.ndarray, level: int, rng) -> np.ndarray:
    limit = (1.6, 1.7, 1.8, 1.9, 2.0)[level]
    scale = rng.uniform(1.0 / limit, limit, size=3)
    return normalize_unit_sphere(points * scale)


def corrupt_jitter(points: np.ndarray, level: int, rng) -> np.ndarray:
    sigma = 0.01 * (level + 1)
    return (points + sigma * rng.randn(*points.shape)).astype(np.float32)


def corrupt_rotate(points: np.ndarray, level: int, rng) -> np.ndarray:
    angle_limit = (math.pi / 6.0) * (level + 1) / 5.0
    x_angle, y_angle, z_angle = rng.uniform(-angle_limit, angle_limit, size=3)
    rx = np.array(
        [[1, 0, 0], [0, np.cos(x_angle), -np.sin(x_angle)], [0, np.sin(x_angle), np.cos(x_angle)]]
    )
    ry = np.array(
        [[np.cos(y_angle), 0, np.sin(y_angle)], [0, 1, 0], [-np.sin(y_angle), 0, np.cos(y_angle)]]
    )
    rz = np.array(
        [[np.cos(z_angle), -np.sin(z_angle), 0], [np.sin(z_angle), np.cos(z_angle), 0], [0, 0, 1]]
    )
    return np.dot(points, np.dot(rz, np.dot(ry, rx))).astype(np.float32)


def corrupt_dropout_global(points: np.ndarray, level: int, rng) -> np.ndarray:
    drop_rate = (0.25, 0.375, 0.5, 0.625, 0.75)[level]
    shuffled = _shuffle(points, rng)
    keep = max(1, int(points.shape[0] * (1.0 - drop_rate)))
    return shuffled[:keep].astype(np.float32)


def corrupt_dropout_local(points: np.ndarray, level: int, rng) -> np.ndarray:
    result = points.copy()
    total_to_drop = min(100 * (level + 1), max(1, result.shape[0] - 1))
    num_clusters = int(rng.randint(1, 8))
    for cluster_size in _cluster_sizes(num_clusters, total_to_drop, rng):
        if cluster_size <= 0 or result.shape[0] <= 1:
            continue
        cluster_size = min(int(cluster_size), result.shape[0] - 1)
        result = _shuffle(result, rng)
        distance = np.sum((result - result[:1]) ** 2, axis=1)
        result = result[np.argsort(distance)[::-1]][: result.shape[0] - cluster_size]
    return result.astype(np.float32)


def corrupt_add_global(points: np.ndarray, level: int, rng) -> np.ndarray:
    added = _sample_unit_sphere(10 * (level + 1), rng)
    return np.concatenate([points, added], axis=0).astype(np.float32)


def corrupt_add_local(points: np.ndarray, level: int, rng) -> np.ndarray:
    total_to_add = 100 * (level + 1)
    num_clusters = int(rng.randint(1, 8))
    shuffled = _shuffle(points, rng)
    clusters = []
    for index, cluster_size in enumerate(_cluster_sizes(num_clusters, total_to_add, rng)):
        if cluster_size <= 0:
            continue
        center = shuffled[index % shuffled.shape[0] : index % shuffled.shape[0] + 1]
        sigma = rng.uniform(0.075, 0.125)
        clusters.append(center + sigma * rng.randn(int(cluster_size), 3))
    added = np.concatenate(clusters, axis=0).astype(np.float32)
    norm = np.linalg.norm(added, axis=1, keepdims=True)
    added = np.where(norm > 1.0, added / np.maximum(norm, 1e-8), added)
    return np.concatenate([points, added], axis=0).astype(np.float32)


CORRUPTION_FUNCTIONS: Dict[str, Callable] = {
    "scale": corrupt_scale,
    "jitter": corrupt_jitter,
    "rotate": corrupt_rotate,
    "dropout_global": corrupt_dropout_global,
    "dropout_local": corrupt_dropout_local,
    "add_global": corrupt_add_global,
    "add_local": corrupt_add_local,
}


def apply_corruption(points: np.ndarray, name: str, rng=None, npoints=None) -> np.ndarray:
    """Apply a canonical corruption, or preserve the point cloud for ``clean``."""
    rng = np.random if rng is None else rng
    base, severity = parse_corruption(name)
    source = np.asarray(points, dtype=np.float32).copy()
    corrupted = source if base == "clean" else CORRUPTION_FUNCTIONS[base](source, severity, rng)
    if npoints is not None:
        corrupted = resample_points(corrupted, npoints, rng)
    return corrupted.astype(np.float32, copy=False)
