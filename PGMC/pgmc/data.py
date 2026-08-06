"""Dataset loaders for ModelNet-C, SONN-C, and PGMC Sim2Real."""

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import DEFAULT_SEVERITY, TEXT_PROMPTS
from .corruptions import apply_corruption, canonical_corruption, normalize_unit_sphere, resample_points


def _read_classnames(root: Path) -> List[str]:
    for filename in ("shape_names.txt", "classnames.txt"):
        path = root / filename
        if path.is_file():
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return []


def _prepare_points(points: np.ndarray, npoints: int, backbone: str, normalize: bool) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)[:, :3]
    points = resample_points(points, npoints)
    if normalize:
        points = normalize_unit_sphere(points)
    if backbone == "openshape":
        points = points.copy()
        points[:, [1, 2]] = points[:, [2, 1]]
    return points.astype(np.float32, copy=False)


class H5CorruptionDataset(Dataset):
    """Read ``clean.h5`` or a pre-generated ``{corruption_type}_2.h5`` file."""

    def __init__(
        self,
        root: str,
        corruption: str,
        npoints: int,
        backbone: str,
        variant: Optional[str] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.corruption = canonical_corruption(corruption, DEFAULT_SEVERITY)
        self.npoints = npoints
        self.backbone = backbone
        self.template = TEXT_PROMPTS

        data_root = self.root / variant if variant else self.root
        data_file = data_root / "{}.h5".format(self.corruption)
        if not data_file.is_file():
            raise FileNotFoundError(
                "PGMC expected clean/severity-2 benchmark file: {}".format(data_file)
            )
        classnames = _read_classnames(self.root) or _read_classnames(data_root)
        if not classnames:
            raise FileNotFoundError(
                "Missing shape_names.txt/classnames.txt under {}".format(self.root)
            )

        with h5py.File(str(data_file), "r") as handle:
            self.points = handle["data"][:].astype(np.float32)
            self.labels = handle["label"][:].reshape(-1).astype(np.int64)
        if self.points.shape[0] != self.labels.shape[0]:
            raise ValueError("Point and label counts differ in {}".format(data_file))
        self.classnames = classnames

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        label = int(self.labels[index])
        points = _prepare_points(
            self.points[index], self.npoints, self.backbone, normalize=False
        )
        return torch.from_numpy(points), label, self.classnames[label]


class FolderPointCloudDataset(Dataset):
    """Class-folder dataset supporting ``class/{train,test}/*.npy`` layouts."""

    def __init__(
        self,
        root: Path,
        npoints: int,
        backbone: str,
        splits: Sequence[str],
        classnames: Optional[Sequence[str]] = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.npoints = npoints
        self.backbone = backbone
        self.template = TEXT_PROMPTS
        available = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        self.classnames = list(classnames) if classnames else available
        class_to_index = {name: index for index, name in enumerate(self.classnames)}
        self.samples: List[Tuple[Path, int]] = []
        for class_name in self.classnames:
            class_root = self.root / class_name
            if not class_root.is_dir():
                continue
            found_split = False
            for split in splits:
                split_root = class_root / split
                if split_root.is_dir():
                    found_split = True
                    self.samples.extend(
                        (path, class_to_index[class_name]) for path in sorted(split_root.glob("*.npy"))
                    )
            if not found_split:
                self.samples.extend(
                    (path, class_to_index[class_name]) for path in sorted(class_root.glob("*.npy"))
                )
        if not self.samples:
            raise FileNotFoundError("No .npy point clouds found under {}".format(self.root))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        points = np.load(str(path)).astype(np.float32)
        points = _prepare_points(points, self.npoints, self.backbone, normalize=True)
        return torch.from_numpy(points), label, self.classnames[label]


class Sim2RealDataset(Dataset):
    """Sim2Real loader that auto-detects H5 or class-folder layouts."""

    def __init__(
        self,
        root: str,
        sim2real_type: str,
        corruption: str,
        npoints: int,
        backbone: str,
    ) -> None:
        base = Path(root).expanduser().resolve()
        typed_root = base / sim2real_type
        data_root = typed_root if typed_root.is_dir() else base
        canonical = "clean" if corruption == "clean" else canonical_corruption(corruption)
        h5_file = data_root / "{}.h5".format(canonical)

        self.npoints = npoints
        self.backbone = backbone
        self.template = TEXT_PROMPTS
        self._folder: Optional[FolderPointCloudDataset] = None
        self.points = None
        self.labels = None

        if h5_file.is_file():
            classnames = _read_classnames(data_root) or _read_classnames(base)
            if not classnames:
                raise FileNotFoundError(
                    "Sim2Real H5 data requires shape_names.txt or classnames.txt under {}"
                    .format(data_root)
                )
            with h5py.File(str(h5_file), "r") as handle:
                self.points = handle["data"][:].astype(np.float32)
                self.labels = handle["label"][:].reshape(-1).astype(np.int64)
            self.classnames = classnames
        else:
            self._folder = FolderPointCloudDataset(
                data_root, npoints, backbone, splits=("test",)
            )
            self.classnames = self._folder.classnames

    def __len__(self) -> int:
        if self._folder is not None:
            return len(self._folder)
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        if self._folder is not None:
            return self._folder[index]
        label = int(self.labels[index])
        points = _prepare_points(
            self.points[index], self.npoints, self.backbone, normalize=True
        )
        return torch.from_numpy(points), label, self.classnames[label]


def _preferred_h5_files(root: Path) -> List[Path]:
    preferred = (
        "ply_data_trainminusval.h5",
        "ply_data_valid.h5",
        "trainminusval.h5",
        "valid.h5",
        "train.h5",
    )
    result = []
    for name in preferred:
        result.extend(sorted(root.rglob(name)))
    if result:
        return list(dict.fromkeys(result))
    candidates = [
        path
        for path in sorted(root.rglob("*.h5"))
        if not any(path.stem.endswith("_{}".format(level)) for level in range(5))
    ]
    return candidates


class SourceDataset(Dataset):
    """Clean source data used to learn the PGMC feature reconstruction adapter."""

    def __init__(
        self,
        root: str,
        target_classnames: Sequence[str],
        npoints: int,
        backbone: str,
        variant: Optional[str] = None,
        folder_splits: Sequence[str] = ("train", "test"),
    ) -> None:
        base = Path(root).expanduser().resolve()
        data_root = base / variant if variant and (base / variant).is_dir() else base
        self.npoints = npoints
        self.backbone = backbone
        self.classnames = list(target_classnames)
        self.template = TEXT_PROMPTS
        self._folder = None

        h5_files = _preferred_h5_files(data_root)
        if h5_files:
            source_names = _read_classnames(base) or _read_classnames(data_root)
            point_parts, label_parts = [], []
            for path in h5_files:
                with h5py.File(str(path), "r") as handle:
                    point_parts.append(handle["data"][:].astype(np.float32))
                    label_parts.append(handle["label"][:].reshape(-1).astype(np.int64))
            self.points = np.concatenate(point_parts, axis=0)
            raw_labels = np.concatenate(label_parts, axis=0)
            if source_names and source_names != self.classnames:
                target_map = {name: index for index, name in enumerate(self.classnames)}
                remap = {index: target_map[name] for index, name in enumerate(source_names) if name in target_map}
                keep = np.array([int(label) in remap for label in raw_labels], dtype=bool)
                self.points = self.points[keep]
                self.labels = np.array([remap[int(label)] for label in raw_labels[keep]], dtype=np.int64)
            else:
                self.labels = raw_labels
        else:
            self._folder = FolderPointCloudDataset(
                data_root,
                npoints,
                backbone,
                splits=folder_splits,
                classnames=self.classnames,
            )
            self.points = None
            self.labels = None

    def __len__(self) -> int:
        if self._folder is not None:
            return len(self._folder)
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        if self._folder is not None:
            return self._folder[index]
        label = int(self.labels[index])
        points = _prepare_points(
            self.points[index], self.npoints, self.backbone, normalize=True
        )
        return torch.from_numpy(points), label, self.classnames[label]


class PairedSourceDataset(Dataset):
    """Return clean/corrupted source pairs with a fixed severity-2 corruption."""

    def __init__(self, source: SourceDataset, corruption: str, seed: int = 1) -> None:
        self.source = source
        self.corruption = canonical_corruption(corruption, DEFAULT_SEVERITY)
        self.seed = seed
        self.classnames = source.classnames
        self.template = source.template

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        clean, label, class_name = self.source[index]
        rng = np.random.RandomState(self.seed + index)
        noisy = apply_corruption(
            clean.numpy(), self.corruption, rng=rng, npoints=clean.shape[0]
        )
        return clean, torch.from_numpy(noisy), label, class_name


def build_target_dataset(args, corruption: str):
    if args.benchmark == "modelnet_c":
        return H5CorruptionDataset(
            args.data_root, corruption, args.npoints, args.backbone
        )
    if args.benchmark == "sonn_c":
        return H5CorruptionDataset(
            args.data_root,
            corruption,
            args.npoints,
            args.backbone,
            variant=args.sonn_variant,
        )
    if args.benchmark == "sim2real":
        return Sim2RealDataset(
            args.data_root,
            args.sim2real_type,
            args.sim2real_corruption,
            args.npoints,
            args.backbone,
        )
    raise ValueError("Unsupported PGMC benchmark: {}".format(args.benchmark))


def build_source_dataset(args, target_classnames: Sequence[str]):
    if not args.source_root:
        return None
    variant = args.sonn_variant if args.benchmark == "sonn_c" else None
    if args.benchmark == "sim2real" and args.source_domain:
        variant = args.source_domain
    return SourceDataset(
        args.source_root,
        target_classnames,
        args.npoints,
        args.backbone,
        variant=variant,
    )
