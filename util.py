from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WindowDataset(Dataset):
    def __init__(self, X: Sequence[Sequence[Sequence[float]]], Y: Sequence[Sequence[float]]):
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        return x, y


def make_folds(exp_ids: List[int], n_splits: int = 5, seed: int = 42) -> List[List[int]]:
    uniq = sorted(set(exp_ids))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for i, e in enumerate(uniq):
        folds[i % n_splits].append(e)
    return folds


def split_by_exp(sample_exp_ids: List[int], folds: List[List[int]], fold_index: int) -> Tuple[List[int], List[int]]:
    test_exps = set(folds[fold_index])
    train_idx: List[int] = []
    test_idx: List[int] = []
    for i, e in enumerate(sample_exp_ids):
        (test_idx if e in test_exps else train_idx).append(i)
    return train_idx, test_idx


def binary_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Tuple[float, float]:
    # y in {0,1}
    tp = tn = fp = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0

    def f1(p, r):
        return 2 * p * r / (p + r) if (p + r) else 0.0

    # class 1
    p1 = tp / (tp + fp) if (tp + fp) else 0.0
    r1 = tp / (tp + fn) if (tp + fn) else 0.0
    f1_pos = f1(p1, r1)
    # class 0
    p0 = tn / (tn + fn) if (tn + fn) else 0.0
    r0 = tn / (tn + fp) if (tn + fp) else 0.0
    f1_neg = f1(p0, r0)
    macro_f1 = (f1_pos + f1_neg) / 2.0
    return acc, macro_f1


def mae(pred: torch.Tensor, true: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - true)).item()


def default_excel_path(root: str | Path | None = None) -> Path | None:
    root = Path(root) if root else Path.cwd()
    # try env
    envp = os.environ.get('FLASHOVER_XLSX')
    if envp:
        p = Path(envp)
        if p.exists():
            return p
    # search known subdir
    candidates: List[Path] = []
    for p in root.rglob('*.xlsx'):
        if 'ThreeCompartments' in p.name or 'Three' in p.name or 'P-Flash' in p.name:
            candidates.append(p)
    if candidates:
        return sorted(candidates, key=lambda x: len(str(x)))[0]
    return None


def add_main_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--data_path',
        type=str,
        default='DataForFlashover/three-compartments/P-Flash_ThreeCompartments_all data 2020-6-26.xlsx',
        help='Path to Excel data file (.xlsx) relative to project root',
    )
    parser.add_argument('--history', type=int, default=256)
    parser.add_argument('--horizon', type=int, default=24)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--channels', type=str, default='HD1,HD2,HD3', help='Comma-separated input channels')
    parser.add_argument('--target_channel', type=str, default='HD1')
    parser.add_argument('--trunc_temp', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--d_state', type=int, default=64)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=600.0, help='Flashover temperature threshold')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--enable_fp32_high_precision', action='store_true',
                        help='Enable torch.set_float32_matmul_precision("high"). Warn if request fails.')


__all__ = [
    'set_seed',
    'get_device',
    'WindowDataset',
    'make_folds',
    'split_by_exp',
    'binary_metrics',
    'mae',
    'default_excel_path',
    'add_main_args',
]
