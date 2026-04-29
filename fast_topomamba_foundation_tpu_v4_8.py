#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast TopoMamba Foundation Training Pipeline, TPU v4-8 edition
==============================================================

This is a TPU/XLA rewrite of the original Fast TopoMamba training script.
Important TPU changes:
  - Uses torch_xla xmp.spawn for multi-core TPU execution.
  - Uses DistributedSampler + MpDeviceLoader.
  - Uses xm.optimizer_step and XLA all-reduce metric aggregation.
  - Defaults to the pure PyTorch scan backend because fused mamba_ssm CUDA
    kernels are not TPU/XLA kernels.
  - Removes per-step .item() synchronizations from the hot path.
  - Replaces Python/dictionary region masking with vectorized tensor masking.

Example TPU v4-8 command:

PJRT_DEVICE=TPU XLA_USE_BF16=1 python fast_topomamba_foundation_tpu_v4_8.py \
  --device tpu \
  --data_source npz \
  --npz_path /path/to/pathmnist_224.npz \
  --output_dir ./runs/fast_topomamba_tpu_v4_8 \
  --image_size 224 \
  --patch_size 16 \
  --patch_encoder resnet18 \
  --pretrained true \
  --patches_per_region 16 \
  --epochs 100 \
  --batch_size 16 \
  --eval_batch_size 32 \
  --scan_backend torch \
  --analysis_every 10 \
  --best_metric knn

Note: --batch_size is per TPU worker/core, not global batch size.
"""

from __future__ import annotations
import argparse, csv, json, math, os, random, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x=None, *args, **kwargs):
        return x if x is not None else range(0)

try:
    import torchvision
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
except Exception as exc:
    raise ImportError("torchvision is required") from exc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from mamba_ssm import Mamba
    HAS_MAMBA_SSM = True
except Exception:
    Mamba = None
    HAS_MAMBA_SSM = False


try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    HAS_XLA = True
except Exception:
    xm = None
    xmp = None
    pl = None
    HAS_XLA = False


def bool_arg(v):
    if isinstance(v, bool): return v
    v = str(v).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}: return True
    if v in {"0", "false", "f", "no", "n"}: return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def ensure_dir(path):
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))


def append_csv(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists(); keys = list(row.keys())
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not exists: writer.writeheader()
        writer.writerow(row)


def is_finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x.detach()).all().item())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


# =============================================================================
# Dataset loading
# =============================================================================

def build_image_transform(image_size: int, force_rgb: bool, n_channels: int, normalize_imagenet: bool):
    ops = []
    if force_rgb:
        ops.append(T.Lambda(lambda im: im.convert("RGB"))); n_channels = 3
    else:
        if n_channels == 1: ops.append(T.Grayscale(num_output_channels=1))
        else: ops.append(T.Lambda(lambda im: im.convert("RGB"))); n_channels = 3
    ops += [T.Resize((image_size, image_size)), T.ToTensor()]
    if n_channels == 3 and normalize_imagenet:
        ops.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    elif n_channels == 3:
        ops.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    else:
        ops.append(T.Normalize([0.5], [0.5]))
    return T.Compose(ops)


class NPZSplitDataset(Dataset):
    def __init__(self, npz_path: str, split: str, image_size: int, force_rgb: bool, normalize_imagenet: bool):
        obj = np.load(npz_path)
        ik, lk = f"{split}_images", f"{split}_labels"
        if ik not in obj or lk not in obj:
            raise KeyError(f"Missing {ik}/{lk}; keys={list(obj.keys())}")
        self.images = obj[ik]
        self.labels = obj[lk].reshape(-1).astype(np.int64)
        self.n_channels = 3 if force_rgb else self._infer_channels(self.images)
        self.transform = build_image_transform(image_size, force_rgb, self.n_channels, normalize_imagenet)

    @staticmethod
    def _infer_channels(arr):
        if arr.ndim == 3: return 1
        if arr.ndim == 4:
            if arr.shape[-1] in {1, 3}: return int(arr.shape[-1])
            if arr.shape[1] in {1, 3}: return int(arr.shape[1])
        return 3

    def __len__(self): return len(self.images)

    def _to_pil(self, arr):
        from PIL import Image
        arr = np.asarray(arr)
        if arr.ndim == 2: return Image.fromarray(arr.astype(np.uint8), mode="L")
        if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return Image.fromarray(arr[..., 0].astype(np.uint8), mode="L")
        return Image.fromarray(arr.astype(np.uint8))

    def __getitem__(self, idx):
        x = self.transform(self._to_pil(self.images[idx]))
        return {"image": x, "label": torch.tensor(int(self.labels[idx]), dtype=torch.long), "index": torch.tensor(idx)}


class MedMNISTDataset(Dataset):
    def __init__(self, dataset, split, size, image_size, data_dir, download, force_rgb, normalize_imagenet):
        try:
            import medmnist
            from medmnist import INFO
        except Exception as exc:
            raise ImportError("Install medmnist for --data_source medmnist") from exc
        if dataset not in INFO: raise ValueError(f"Unknown MedMNIST dataset {dataset}")
        info = INFO[dataset]; cls = getattr(medmnist, info["python_class"])
        original_ch = int(info.get("n_channels", 3))
        self.n_channels = 3 if force_rgb else original_ch
        self.n_classes = len(info["label"])
        transform = build_image_transform(image_size, force_rgb, self.n_channels, normalize_imagenet)
        ensure_dir(data_dir)
        try:
            self.base = cls(size=size, split=split, root=data_dir, download=download, transform=transform)
        except TypeError:
            self.base = cls(split=split, root=data_dir, download=download, transform=transform)

    def __len__(self): return len(self.base)
    def _label(self, y): return int(y.reshape(-1)[0].item()) if torch.is_tensor(y) else int(np.asarray(y).reshape(-1)[0])
    def __getitem__(self, idx):
        x, y = self.base[idx]
        return {"image": x, "label": torch.tensor(self._label(y), dtype=torch.long), "index": torch.tensor(idx)}


class WrappedImageFolder(Dataset):
    def __init__(self, root, image_size, force_rgb, normalize_imagenet):
        self.base = ImageFolder(str(root), transform=build_image_transform(image_size, True, 3, normalize_imagenet))
        self.n_channels = 3; self.n_classes = len(self.base.classes)
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        return {"image": x, "label": torch.tensor(int(y), dtype=torch.long), "index": torch.tensor(idx)}


def labels_from_dataset(ds):
    if isinstance(ds, Subset):
        base = labels_from_dataset(ds.dataset); return base[np.asarray(ds.indices)]
    if hasattr(ds, "labels"): return np.asarray(ds.labels).reshape(-1)
    if hasattr(ds, "base") and hasattr(ds.base, "targets"): return np.asarray(ds.base.targets).reshape(-1)
    return np.asarray([int(ds[i]["label"]) for i in range(len(ds))])


def split_dataset(ds, val_fraction, seed):
    n = len(ds); idx = np.arange(n); rng = np.random.default_rng(seed); rng.shuffle(idx)
    n_val = max(1, int(round(n * val_fraction)))
    return Subset(ds, idx[n_val:].tolist()), Subset(ds, idx[:n_val].tolist())


def build_datasets(args):
    if args.data_source == "npz":
        obj = np.load(args.npz_path); keys = set(obj.keys())
        train_full = NPZSplitDataset(args.npz_path, "train", args.image_size, args.force_rgb, args.normalize_imagenet)
        if "val_images" in keys and "val_labels" in keys and not args.force_split:
            train_ds = train_full
            val_ds = NPZSplitDataset(args.npz_path, "val", args.image_size, args.force_rgb, args.normalize_imagenet)
        elif "test_images" in keys and "test_labels" in keys and not args.force_split:
            train_ds = train_full
            val_ds = NPZSplitDataset(args.npz_path, "test", args.image_size, args.force_rgb, args.normalize_imagenet)
        else:
            train_ds, val_ds = split_dataset(train_full, args.val_fraction, args.seed)
        labels = np.concatenate([labels_from_dataset(train_ds), labels_from_dataset(val_ds)])
        return train_ds, val_ds, train_full.n_channels, int(labels.max()) + 1

    if args.data_source == "medmnist":
        full = MedMNISTDataset(args.dataset, "train", args.size, args.image_size, args.data_dir, args.download, args.force_rgb, args.normalize_imagenet)
        try:
            val_ds = MedMNISTDataset(args.dataset, args.val_split, args.size, args.image_size, args.data_dir, args.download, args.force_rgb, args.normalize_imagenet)
            train_ds = full
        except Exception:
            train_ds, val_ds = split_dataset(full, args.val_fraction, args.seed)
        return train_ds, val_ds, full.n_channels, full.n_classes

    if args.data_source == "folder":
        root = Path(args.folder_root)
        if (root / "train").exists() and ((root / "val").exists() or (root / "test").exists()):
            train_ds = WrappedImageFolder(root / "train", args.image_size, args.force_rgb, args.normalize_imagenet)
            val_ds = WrappedImageFolder(root / ("val" if (root / "val").exists() else "test"), args.image_size, args.force_rgb, args.normalize_imagenet)
            return train_ds, val_ds, train_ds.n_channels, train_ds.n_classes
        full = WrappedImageFolder(root, args.image_size, args.force_rgb, args.normalize_imagenet)
        train_ds, val_ds = split_dataset(full, args.val_fraction, args.seed)
        return train_ds, val_ds, full.n_channels, full.n_classes

    raise ValueError(args.data_source)


# =============================================================================
# Geometry and Laplacian PE
# =============================================================================

def _rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1: x, y = n - 1 - x, n - 1 - y
        x, y = y, x
    return x, y


def _xy2d(n, x, y):
    d = 0; s = n // 2
    while s > 0:
        rx = 1 if (x & s) else 0; ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _rot(s, x, y, rx, ry); s //= 2
    return d


def hilbert_order(coords_hw):
    rows = coords_hw[:, 0].cpu().numpy().astype(np.int64)
    cols = coords_hw[:, 1].cpu().numpy().astype(np.int64)
    side = 1; max_side = int(max(rows.max(initial=0) + 1, cols.max(initial=0) + 1))
    while side < max_side: side *= 2
    keys = np.asarray([_xy2d(side, int(c), int(r)) for r, c in zip(rows, cols)])
    return torch.from_numpy(np.argsort(keys)).long()


def factor_pair(P, patch_grid):
    factors = []
    for h in range(1, P + 1):
        if P % h == 0:
            w = P // h
            if h <= patch_grid and w <= patch_grid: factors.append((h, w))
    if not factors: raise ValueError(f"Cannot factor patches_per_region={P}")
    factors.sort(key=lambda x: (abs(x[0] - x[1]), -x[0]))
    return factors[0]


def auto_stride(h, w): return max(1, h // 2), max(1, w // 2)


def build_geometry(args):
    if args.image_size % args.patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size")
    pg_h = pg_w = args.image_size // args.patch_size
    if args.region_patch_h > 0 and args.region_patch_w > 0:
        rph, rpw = args.region_patch_h, args.region_patch_w
    else:
        rph, rpw = factor_pair(args.patches_per_region, pg_h)
    if args.region_stride_h > 0 and args.region_stride_w > 0:
        rsh, rsw = args.region_stride_h, args.region_stride_w
    else:
        rsh, rsw = auto_stride(rph, rpw)
    if rph > pg_h or rpw > pg_w: raise ValueError("Region larger than patch grid")

    starts_h = list(range(0, pg_h - rph + 1, rsh)); starts_w = list(range(0, pg_w - rpw + 1, rsw))
    indices, coords_hw, centers = [], [], []
    for rh, top in enumerate(starts_h):
        for rw, left in enumerate(starts_w):
            idxs, xs, ys = [], [], []
            for dy in range(rph):
                for dx in range(rpw):
                    py, px = top + dy, left + dx
                    idxs.append(py * pg_w + px); xs.append((px + 0.5) * args.patch_size); ys.append((py + 0.5) * args.patch_size)
            indices.append(idxs); coords_hw.append([rh, rw]); centers.append([float(np.mean(xs)), float(np.mean(ys))])
    region_indices = torch.tensor(indices, dtype=torch.long)
    region_mask = torch.ones_like(region_indices, dtype=torch.bool)
    coords_hw = torch.tensor(coords_hw, dtype=torch.long); centers = torch.tensor(centers, dtype=torch.float32)
    order = hilbert_order(coords_hw)
    return {
        "image_size": args.image_size, "patch_size": args.patch_size,
        "patch_grid_h": pg_h, "patch_grid_w": pg_w,
        "region_patch_h": rph, "region_patch_w": rpw,
        "region_stride_h": rsh, "region_stride_w": rsw,
        "region_grid_h": len(starts_h), "region_grid_w": len(starts_w),
        "num_regions": int(region_indices.shape[0]), "patches_per_region": int(region_indices.shape[1]),
        "region_indices": region_indices[order].contiguous(),
        "region_patch_mask": region_mask[order].contiguous(),
        "region_coords_hw": coords_hw[order].contiguous(),
        "region_centers_xy": centers[order].contiguous(),
    }


def laplacian_pe(coords_xy, pe_dim, knn_k):
    n = int(coords_xy.shape[0])
    if n <= 1: return torch.zeros(n, pe_dim)
    dist = torch.cdist(coords_xy.float(), coords_xy.float())
    k = min(knn_k, n - 1); knn = torch.topk(dist, k=k + 1, largest=False).indices[:, 1:]
    pos = dist[dist > 0]; sigma = max(float(torch.median(pos).item()) if pos.numel() else 1.0, 1e-6)
    A = torch.zeros(n, n)
    for i in range(n):
        for j in knn[i].tolist():
            A[i, j] = max(A[i, j], math.exp(-(float(dist[i, j]) ** 2) / (2 * sigma * sigma)))
    A = torch.maximum(A, A.t()); deg = A.sum(1)
    D_inv = torch.diag(torch.pow(deg + 1e-8, -0.5)); L = torch.eye(n) - D_inv @ A @ D_inv
    _, evecs = torch.linalg.eigh(L)
    usable = min(pe_dim, max(0, n - 1)); pe = evecs[:, 1:1 + usable].float()
    if pe.shape[1] < pe_dim: pe = F.pad(pe, (0, pe_dim - pe.shape[1]))
    for j in range(pe.shape[1]):
        col = pe[:, j]
        if col.abs().sum() > 0 and col[torch.argmax(col.abs())] < 0: pe[:, j] = -col
    return pe.contiguous()


# =============================================================================
# Encoders and VMamba
# =============================================================================

class PixelMLPPatchEncoder(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, hidden_dim):
        super().__init__(); self.patch_size = patch_size
        self.net = nn.Sequential(nn.LayerNorm(in_channels * patch_size * patch_size), nn.Linear(in_channels * patch_size * patch_size, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim), nn.LayerNorm(embed_dim))
    def forward(self, x):
        p = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2).contiguous()
        z = torch.nan_to_num(self.net(p), nan=0.0, posinf=1.0, neginf=-1.0)
        return F.normalize(z, dim=-1, eps=1e-6)


class DenseTorchvisionPatchEncoder(nn.Module):
    def __init__(self, name, pretrained, embed_dim, patch_grid_hw):
        super().__init__(); name = name.lower(); self.patch_grid_hw = patch_grid_hw
        if name == "resnet18":
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None; m = torchvision.models.resnet18(weights=w); out_ch = 512; self.features = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        elif name == "resnet34":
            w = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None; m = torchvision.models.resnet34(weights=w); out_ch = 512; self.features = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        elif name == "resnet50":
            w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None; m = torchvision.models.resnet50(weights=w); out_ch = 2048; self.features = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        elif name == "convnext_tiny":
            w = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None; m = torchvision.models.convnext_tiny(weights=w); out_ch = 768; self.features = m.features
        elif name == "efficientnet_b0":
            w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None; m = torchvision.models.efficientnet_b0(weights=w); out_ch = 1280; self.features = m.features
        else: raise ValueError(name)
        self.proj = nn.Conv2d(out_ch, embed_dim, 1, bias=False); self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        f = self.features(x); f = F.interpolate(f, size=self.patch_grid_hw, mode="bilinear", align_corners=False); f = self.proj(f)
        t = self.norm(f.flatten(2).transpose(1, 2).contiguous())
        return F.normalize(torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0), dim=-1, eps=1e-6)


def build_patch_encoder(args, n_channels, patch_grid_hw):
    if args.patch_encoder == "pixel_mlp": return PixelMLPPatchEncoder(n_channels, args.patch_size, args.patch_embed_dim, args.patch_mlp_hidden)
    if n_channels != 3: raise ValueError("Pretrained encoders require RGB; use --force_rgb true")
    return DenseTorchvisionPatchEncoder(args.patch_encoder, args.pretrained, args.patch_embed_dim, patch_grid_hw)


class SlowSelectiveScan1D(nn.Module):
    def __init__(self, dim, state_dim=48, dt_rank=24):
        super().__init__(); self.state_dim = state_dim; self.dt_rank = dt_rank
        self.in_proj = nn.Linear(dim, 2 * dim, bias=False); self.x_proj = nn.Linear(dim, dt_rank + 2 * state_dim, bias=False); self.dt_proj = nn.Linear(dt_rank, dim)
        nn.init.uniform_(self.dt_proj.bias, -4, -2)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1).float().unsqueeze(0).repeat(dim, 1))); self.D = nn.Parameter(torch.ones(dim)); self.out_proj = nn.Linear(dim, dim, bias=False); self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        B, L, D = x.shape
        N = self.state_dim
        xz = self.in_proj(x)
        u, z = xz.chunk(2, -1)                                     # u,z: [B,L,D]
        params = self.x_proj(u)
        dt_raw, Bm, Cm = params.split([self.dt_rank, N, N], -1)
        delta = F.softplus(self.dt_proj(dt_raw)).clamp(max=10)     # [B,L,D]
        A = -torch.exp(self.A_log.float()).clamp(max=1e4)           # [D,N]

        # --- Vectorized scan (no Python for-loop → XLA-friendly static graph) ---
        # dA[b,t,d,n] = exp(delta[b,t,d] * A[d,n])
        dA = torch.exp(
            (delta.unsqueeze(-1) * A[None, None]).clamp(-30, 30)
        )  # [B,L,D,N]
        # dB*u: outer product over N and D dims
        dBu = (delta.unsqueeze(-1) * Bm.unsqueeze(2)) * u.unsqueeze(-1)  # [B,L,D,N]

        # Linear recurrence h[t] = dA[t]*h[t-1] + dBu[t], h[-1]=0
        # Solved via log-space prefix-product cumsum (fully vectorised):
        #   cumA[t] = prod_{s=0}^{t} dA[s]
        #   h[t]    = cumA[t] * cumsum(dBu / cumA, dim=1)[t]
        log_dA   = torch.log(dA.clamp(min=1e-38))          # [B,L,D,N]
        log_cumA = torch.cumsum(log_dA, dim=1)              # [B,L,D,N]
        cumA     = torch.exp(log_cumA)                      # [B,L,D,N]
        h = cumA * torch.cumsum(dBu * torch.exp(-log_cumA), dim=1)  # [B,L,D,N]

        # Output: y[t] = C[t] @ h[t] + D * u[t]
        y_seq = (Cm.unsqueeze(2) * h).sum(-1) + self.D * u  # [B,L,D]
        y = self.out_proj(y_seq * F.silu(z))
        return self.norm(x + torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0))


class FastMambaScan1D(nn.Module):
    def __init__(self, dim, state_dim, conv_kernel, expand, dropout):
        super().__init__()
        if not HAS_MAMBA_SSM: raise ImportError("mamba_ssm unavailable; use --scan_backend torch")
        self.mamba = Mamba(d_model=dim, d_state=state_dim, d_conv=conv_kernel, expand=expand); self.norm = nn.LayerNorm(dim); self.drop = nn.Dropout(dropout)
    def forward(self, x):
        y = torch.nan_to_num(self.mamba(x), nan=0.0, posinf=1.0, neginf=-1.0)
        return self.norm(x + self.drop(y))


def build_scan(args, dim):
    if args.scan_backend == "mamba": return FastMambaScan1D(dim, args.vmamba_state_dim, args.vmamba_conv_kernel, args.mamba_expand, args.dropout)
    return SlowSelectiveScan1D(dim, args.vmamba_state_dim, args.vmamba_dt_rank)


class VMambaCrossScanBlock(nn.Module):
    def __init__(self, args, dim):
        super().__init__(); self.norm = nn.LayerNorm(dim); self.in_proj = nn.Linear(dim, dim); self.dwconv = nn.Conv2d(dim, dim, args.vmamba_conv_kernel, padding=args.vmamba_conv_kernel // 2, groups=dim)
        self.lr_scan = build_scan(args, dim); self.rl_scan = build_scan(args, dim); self.tb_scan = build_scan(args, dim); self.bt_scan = build_scan(args, dim)
        self.out_proj = nn.Linear(dim, dim); self.drop = nn.Dropout(args.dropout); self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(args.dropout), nn.Linear(4 * dim, dim), nn.Dropout(args.dropout))
    def forward(self, grid, valid):
        B, H, W, D = grid.shape; x = self.dwconv(self.in_proj(self.norm(grid)).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        rows = x.view(B * H, W, D) * valid.view(B * H, W).unsqueeze(-1).to(x.dtype)
        lr = self.lr_scan(rows).view(B, H, W, D); rl = torch.flip(self.rl_scan(torch.flip(rows, [1])), [1]).view(B, H, W, D)
        cols = x.permute(0, 2, 1, 3).contiguous().view(B * W, H, D) * valid.permute(0, 2, 1).contiguous().view(B * W, H).unsqueeze(-1).to(x.dtype)
        tb = self.tb_scan(cols).view(B, W, H, D).permute(0, 2, 1, 3).contiguous(); bt = torch.flip(self.bt_scan(torch.flip(cols, [1])), [1]).view(B, W, H, D).permute(0, 2, 1, 3).contiguous()
        out = grid + self.drop(self.out_proj((lr + rl + tb + bt) / 4)); out = out + self.ffn(out)
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0) * valid.unsqueeze(-1).to(out.dtype)


class VMamba2DEncoder(nn.Module):
    def __init__(self, args, dim): super().__init__(); self.blocks = nn.ModuleList([VMambaCrossScanBlock(args, dim) for _ in range(args.vmamba_depth)]); self.norm = nn.LayerNorm(dim)
    def forward(self, grid, valid):
        x = grid
        for b in self.blocks: x = b(x, valid)
        return self.norm(x)


class GatedRegionAggregator(nn.Module):
    def __init__(self, patch_dim, region_dim, hidden_dim, patches_per_region):
        super().__init__(); self.local_pos = nn.Parameter(torch.randn(1, 1, patches_per_region, patch_dim) * 0.02)
        self.gate = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.proj = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, region_dim), nn.GELU(), nn.Linear(region_dim, region_dim), nn.LayerNorm(region_dim))
    def forward(self, patch_tokens, patch_mask):
        B, R, P, D = patch_tokens.shape; x = patch_tokens + self.local_pos[:, :, :P, :].to(patch_tokens.device, patch_tokens.dtype); logits = self.gate(x).squeeze(-1); mask = patch_mask.bool(); logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min); attn = torch.softmax(logits, -1) * mask.to(logits.dtype); attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-6); pooled = self.proj((attn.unsqueeze(-1) * x).sum(-2)); return F.normalize(torch.nan_to_num(pooled, nan=0.0, posinf=1.0, neginf=-1.0), dim=-1, eps=1e-6)


class PositionalMLP(nn.Module):
    def __init__(self, in_dim, out_dim): super().__init__(); self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
    def forward(self, x): return self.net(x)


def scatter_to_grid(seq, coords_hw, grid_h=None, grid_w=None):
    """Scatter region sequence to a 2D grid without in-place advanced indexing.

    The one-hot route is more compiler-friendly for TPU/XLA than grid[b, r, c] = seq.
    Region counts here are small enough that this is not the bottleneck.
    """
    B, R, D = seq.shape
    device = seq.device
    coords = coords_hw.to(device)
    rows, cols = coords[:, 0].long(), coords[:, 1].long()
    H = int(grid_h) if grid_h is not None else int(rows.detach().cpu().max()) + 1
    W = int(grid_w) if grid_w is not None else int(cols.detach().cpu().max()) + 1
    lin = rows * W + cols
    oh = F.one_hot(lin, num_classes=H * W).to(seq.dtype)  # [R, H*W]
    grid = torch.einsum("brd,rp->bpd", seq, oh).view(B, H, W, D)
    valid_2d = oh.sum(0).view(H, W).bool()
    valid = valid_2d.unsqueeze(0).expand(B, H, W)
    rr = rows.unsqueeze(0).expand(B, R)
    cc = cols.unsqueeze(0).expand(B, R)
    return grid, valid, torch.stack([rr, cc], -1)


def gather_from_grid(grid, pos):
    B, H, W, D = grid.shape
    idx = (pos[:, :, 0].long() * W + pos[:, :, 1].long()).unsqueeze(-1).expand(-1, -1, D)
    return grid.view(B, H * W, D).gather(1, idx)


def nt_xent(z1, z2, temperature):
    z1 = F.normalize(z1.float(), dim=-1, eps=1e-6); z2 = F.normalize(z2.float(), dim=-1, eps=1e-6); N = z1.shape[0]
    if N < 2: return z1.sum() * 0.0
    z = torch.cat([z1, z2], 0); logits = (z @ z.t() / max(float(temperature), 1e-4)).clamp(-80, 80); eye = torch.eye(2 * N, dtype=torch.bool, device=logits.device); logits = logits.masked_fill(eye, -torch.finfo(logits.dtype).max / 2); targets = torch.cat([torch.arange(N, 2 * N, device=logits.device), torch.arange(0, N, device=logits.device)]); return F.cross_entropy(logits, targets)



def build_topology_masks(coords_xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Static near/far graph masks computed once on CPU.

    This replaces torch.quantile inside the forward pass to avoid repeated
    TPU/XLA synchronization and recompilation pressure.
    """
    coords = coords_xy.float().cpu()
    R = int(coords.shape[0])
    if R <= 1:
        z = torch.zeros(R, R, dtype=torch.bool)
        return z, z
    dist = torch.cdist(coords, coords)
    pos = dist[dist > 0]
    if pos.numel() < 4:
        z = torch.zeros(R, R, dtype=torch.bool)
        return z, z
    q1 = torch.quantile(pos, 0.25)
    q3 = torch.quantile(pos, 0.75)
    near = (dist > 0) & (dist <= q1)
    far = dist >= q3
    return near.bool(), far.bool()


def topology_loss_static(
    region_tokens: torch.Tensor,
    near_mask: torch.Tensor,
    far_mask: torch.Tensor,
    has_near: bool = True,
    has_far: bool = True,
) -> torch.Tensor:
    """Topology regularization using precomputed graph neighborhoods.

    `has_near` and `has_far` are Python booleans computed once in __init__, so
    the TPU forward pass does not perform bool(.cpu()) synchronizations.
    """
    rt = F.normalize(region_tokens.float(), dim=-1, eps=1e-6)
    sim = torch.bmm(rt, rt.transpose(1, 2))
    zero = region_tokens.sum() * 0.0
    near = near_mask.to(region_tokens.device, dtype=sim.dtype).unsqueeze(0)
    far = far_mask.to(region_tokens.device, dtype=sim.dtype).unsqueeze(0)
    near_loss = (((1.0 - sim) * near).sum() / near.sum().clamp_min(1.0)) if has_near else zero
    far_loss = ((F.relu(sim - 0.2) * far).sum() / far.sum().clamp_min(1.0)) if has_far else zero
    return near_loss + far_loss


class FastTopoMambaFoundation(nn.Module):
    def __init__(self, args, n_channels, geometry):
        super().__init__()
        self.args = args
        self.geometry = geometry
        self.region_grid_h = int(geometry["region_grid_h"])
        self.region_grid_w = int(geometry["region_grid_w"])
        self.num_regions = int(geometry["num_regions"])

        self.patch_encoder = build_patch_encoder(args, n_channels, (geometry["patch_grid_h"], geometry["patch_grid_w"]))
        self.register_buffer("region_indices", geometry["region_indices"].long(), persistent=True)
        self.register_buffer("region_patch_mask", geometry["region_patch_mask"].bool(), persistent=True)
        self.register_buffer("region_coords_hw", geometry["region_coords_hw"].long(), persistent=True)
        self.register_buffer("region_centers_xy", geometry["region_centers_xy"].float(), persistent=True)
        self.register_buffer("graph_pe", laplacian_pe(geometry["region_centers_xy"], args.graph_pe_dim, args.knn_k).float(), persistent=True)
        near, far = build_topology_masks(geometry["region_centers_xy"])
        self.has_topo_near = bool(near.any().item())
        self.has_topo_far = bool(far.any().item())
        self.register_buffer("topo_near_mask", near, persistent=False)
        self.register_buffer("topo_far_mask", far, persistent=False)

        self.region_agg = GatedRegionAggregator(args.patch_embed_dim, args.region_embed_dim, args.agg_hidden_dim, int(geometry["patches_per_region"]))
        self.token_proj = nn.Linear(args.region_embed_dim, args.model_dim)
        self.graph_pe_proj = PositionalMLP(args.graph_pe_dim, args.model_dim)
        self.encoder = VMamba2DEncoder(args, args.model_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, args.model_dim) * 0.02)
        self.recon_head = nn.Linear(args.model_dim, args.region_embed_dim)
        self.proj_head = nn.Sequential(
            nn.LayerNorm(args.model_dim),
            nn.Linear(args.model_dim, args.model_dim),
            nn.GELU(),
            nn.Linear(args.model_dim, args.embedding_dim),
        )

    def image_to_region_tokens(self, x):
        pt = self.patch_encoder(x)
        idx = self.region_indices.to(pt.device)
        B, R, P = pt.shape[0], idx.shape[0], idx.shape[1]
        rp = pt[:, idx.reshape(-1), :].view(B, R, P, -1)
        rm = self.region_patch_mask.to(pt.device).unsqueeze(0).expand(B, -1, -1)
        return self.region_agg(rp, rm), pt

    def add_pe(self, rt):
        B, R, _ = rt.shape
        pe = self.graph_pe[:R].unsqueeze(0).expand(B, R, -1).to(rt.device, rt.dtype)
        return torch.nan_to_num(self.token_proj(rt) + self.graph_pe_proj(pe), nan=0.0, posinf=1.0, neginf=-1.0)

    def spatial_encode(self, seq):
        grid, valid, pos = scatter_to_grid(
            seq,
            self.region_coords_hw[:seq.shape[1]],
            self.region_grid_h,
            self.region_grid_w,
        )
        return gather_from_grid(self.encoder(grid, valid), pos)

    def encode_image(self, x):
        rt, pt = self.image_to_region_tokens(x)
        enc = self.spatial_encode(self.add_pe(rt))
        emb = F.normalize(self.proj_head(enc.mean(1)), dim=-1, eps=1e-6)
        return {
            "embedding": torch.nan_to_num(emb, nan=0.0, posinf=1.0, neginf=-1.0),
            "region_tokens": rt,
            "encoded_regions": enc,
            "patch_tokens": pt,
        }

    def make_region_mask(self, B, device):
        """XLA-friendly block-ish region masking.

        The original code used Python dictionaries, .tolist(), random.choice(),
        and per-sample loops on device tensors. This version builds block-like
        scores with vectorized tensor ops and keeps exactly target masked regions
        per sample.
        """
        R = self.num_regions
        target = max(1, min(R, int(round(R * float(self.args.mask_ratio)))))
        coords = self.region_coords_hw.to(device)
        rows = coords[:, 0].view(1, 1, R)
        cols = coords[:, 1].view(1, 1, R)

        block_area = max(1, int(self.args.mask_block_max) * int(self.args.mask_block_max))
        num_blocks = max(1, int(math.ceil(target / block_area)))
        min_b = max(1, int(self.args.mask_block_min))
        max_b = max(min_b, int(self.args.mask_block_max))

        h = torch.randint(min_b, max_b + 1, (B, num_blocks, 1), device=device)
        w = torch.randint(min_b, max_b + 1, (B, num_blocks, 1), device=device)
        r0 = torch.randint(0, max(1, self.region_grid_h), (B, num_blocks, 1), device=device)
        c0 = torch.randint(0, max(1, self.region_grid_w), (B, num_blocks, 1), device=device)

        in_block = (rows >= r0) & (rows < r0 + h) & (cols >= c0) & (cols < c0 + w)
        block_mask = in_block.any(dim=1)
        scores = torch.rand(B, R, device=device) + block_mask.to(torch.float32) * 2.0
        idx = torch.topk(scores, k=target, dim=1).indices
        mask = torch.zeros(B, R, dtype=torch.bool, device=device)
        mask.scatter_(1, idx, True)
        return mask

    def forward_ssl(self, x1, x2):
        o1 = self.encode_image(x1)
        target = o1["region_tokens"].detach()
        seq = self.add_pe(o1["region_tokens"])
        mask = self.make_region_mask(x1.shape[0], x1.device)
        seq_m = torch.where(mask.unsqueeze(-1), self.mask_token.to(seq.dtype), seq)

        recon = torch.nan_to_num(self.recon_head(self.spatial_encode(seq_m)), nan=0.0, posinf=1.0, neginf=-1.0)
        recon_mse = F.mse_loss(recon[mask].float(), target[mask].float())
        recon_cos = (1.0 - F.cosine_similarity(recon[mask].float(), target[mask].float(), dim=-1, eps=1e-6)).mean()

        o2 = self.encode_image(x2)
        contrastive = nt_xent(o1["embedding"], o2["embedding"], self.args.temperature)
        topo = topology_loss_static(
            o1["region_tokens"],
            self.topo_near_mask,
            self.topo_far_mask,
            self.has_topo_near,
            self.has_topo_far,
        )
        total = torch.nan_to_num(
            self.args.w_recon * recon_mse + self.args.w_contrastive * contrastive + self.args.w_topology * topo,
            nan=0.0,
            posinf=1e4,
            neginf=0.0,
        )

        with torch.no_grad():
            z1, z2 = o1["embedding"].float(), o2["embedding"].float()
            pos_cos = F.cosine_similarity(z1, z2, dim=-1, eps=1e-6).mean()
            zn = F.normalize(z1, dim=-1, eps=1e-6)
            sim = zn @ zn.t()
            neg_cos = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)].mean() if sim.shape[0] > 1 else sim.sum() * 0
            emb_var = z1.var(0).mean()

        return {
            "total": total,
            "recon_mse": recon_mse,
            "recon_cos": recon_cos,
            "contrastive": contrastive,
            "topology": topo,
            "emb_var": emb_var,
            "pos_cos": pos_cos,
            "neg_cos": neg_cos,
            "embedding": o1["embedding"].detach(),
            "region_original": target.detach(),
            "region_reconstructed": recon.detach(),
            "region_mask": mask.detach(),
        }


# =============================================================================
# TPU / train / eval helpers
# =============================================================================

def is_xla_device(device: torch.device) -> bool:
    return HAS_XLA and str(device).startswith("xla")


def xla_world_size() -> int:
    if not HAS_XLA:
        return 1
    try:
        return int(xm.xrt_world_size())
    except Exception:
        return 1


def xla_rank() -> int:
    if not HAS_XLA:
        return 0
    try:
        return int(xm.get_ordinal())
    except Exception:
        return 0


def is_master_process() -> bool:
    if not HAS_XLA:
        return True
    try:
        return bool(xm.is_master_ordinal())
    except Exception:
        return xla_rank() == 0


def master_print(*args, **kwargs) -> None:
    if HAS_XLA:
        try:
            xm.master_print(*args, **kwargs)
            return
        except Exception:
            pass
    if is_master_process():
        print(*args, **kwargs)


def make_views(x, args):
    x1 = x
    x2 = x
    if args.contrastive_flip:
        flip = (torch.rand(x.shape[0], device=x.device) < 0.5).view(-1, 1, 1, 1)
        x2 = torch.where(flip, torch.flip(x2, [-1]), x2)
    if args.contrastive_vflip:
        flip = (torch.rand(x.shape[0], device=x.device) < 0.5).view(-1, 1, 1, 1)
        x2 = torch.where(flip, torch.flip(x2, [-2]), x2)
    if args.view_noise_std > 0:
        x2 = x2 + float(args.view_noise_std) * torch.randn_like(x2)
    return x1, x2


@torch.no_grad()
def extract_embeddings(model, loader, device, args, max_samples=None):
    model.eval()
    zs, ys, seen = [], [], 0
    use_xla = is_xla_device(device)
    iterator = loader if use_xla else tqdm(loader, desc="Extract embeddings", disable=not args.show_progress, leave=False)
    for batch in iterator:
        x = batch["image"] if use_xla else batch["image"].to(device, non_blocking=True)
        y = batch["label"].long()
        if not use_xla:
            y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp and device.type == "cuda")):
            z = model.encode_image(x)["embedding"].float()
        if use_xla:
            y = y.to(device)
            z = xm.all_gather(z, dim=0)
            y = xm.all_gather(y, dim=0)
            xm.mark_step()
        zs.append(z.cpu())
        ys.append(y.cpu())
        seen += int(z.shape[0])
        if max_samples and seen >= max_samples:
            break
    z, y = torch.cat(zs, 0), torch.cat(ys, 0).view(-1)
    return (z[:max_samples], y[:max_samples]) if max_samples else (z, y)


def knn_accuracy(train_z, train_y, val_z, val_y, k=5):
    train_z = F.normalize(train_z.float(), dim=-1, eps=1e-6)
    val_z = F.normalize(val_z.float(), dim=-1, eps=1e-6)
    idx = (val_z @ train_z.t()).topk(k=min(k, train_z.shape[0]), dim=1).indices
    neigh = train_y[idx]
    preds = []
    for i in range(neigh.shape[0]):
        vals, counts = torch.unique(neigh[i], return_counts=True)
        preds.append(vals[torch.argmax(counts)])
    pred = torch.stack(preds)
    return float((pred == val_y).float().mean() * 100.0)


def save_pca_plot(z, y, path, title):
    if not HAS_MPL:
        return
    try:
        zc = z.float() - z.float().mean(0, keepdim=True)
        _, _, vh = torch.linalg.svd(zc, full_matrices=False)
        xy = (zc @ vh[:2].t()).numpy()
        yy = y.numpy()
        plt.figure(figsize=(7, 6))
        sc = plt.scatter(xy[:, 0], xy[:, 1], c=yy, s=8, cmap="tab10", alpha=0.75)
        plt.title(title)
        plt.colorbar(sc, fraction=0.046)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
    except Exception as e:
        master_print(f"[WARN] PCA failed: {e}")


def save_tsne_plot(z, y, path, title, max_samples):
    if not HAS_MPL:
        return
    try:
        from sklearn.manifold import TSNE
        xy = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30).fit_transform(z[:max_samples].float().numpy())
        yy = y[:max_samples].numpy()
        plt.figure(figsize=(7, 6))
        sc = plt.scatter(xy[:, 0], xy[:, 1], c=yy, s=8, cmap="tab10", alpha=0.75)
        plt.title(title)
        plt.colorbar(sc, fraction=0.046)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
    except Exception as e:
        master_print(f"[WARN] t-SNE skipped: {e}")


def save_curves(history, path):
    if not HAS_MPL or not history:
        return
    keys = ["train_total", "val_total", "train_recon_mse", "val_recon_mse", "train_contrastive", "val_contrastive", "val_knn_acc"]
    epochs = [r["epoch"] for r in history]
    plt.figure(figsize=(10, 6))
    for k in keys:
        plt.plot(epochs, [r.get(k, np.nan) for r in history], label=k)
    plt.legend()
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def ckpt_payload(model, optimizer, scheduler, scaler, epoch, best_metric, args, geometry, history):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "args": vars(args),
        "geometry": {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in geometry.items()},
        "history": history,
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_metric, args, geometry, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ckpt_payload(model, optimizer, scheduler, scaler, epoch, best_metric, args, geometry, history)
    dev = next(model.parameters()).device
    if is_xla_device(dev):
        xm.save(payload, str(path), master_only=True)
    elif is_master_process():
        torch.save(payload, path)


def move_optimizer_state_to_device(optimizer, device):
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    c = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(c["model"], strict=True)
    if optimizer and c.get("optimizer"):
        optimizer.load_state_dict(c["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
    if scheduler and c.get("scheduler"):
        scheduler.load_state_dict(c["scheduler"])
    if scaler and c.get("scaler"):
        scaler.load_state_dict(c["scaler"])
    return c


def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio):
    def fn(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        t = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1 + math.cos(math.pi * t)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def run_epoch(model, loader, optimizer, scheduler, scaler, device, args, train, epoch):
    model.train(train)
    keys = ["total", "recon_mse", "recon_cos", "contrastive", "topology", "emb_var", "pos_cos", "neg_cos"]
    use_xla = is_xla_device(device)
    meter = torch.zeros(len(keys) + 1, device=device, dtype=torch.float32)
    phase = "Train" if train else "Val"
    iterator = loader if use_xla else tqdm(loader, desc=f"{phase} {epoch:03d}", disable=not args.show_progress, leave=train)
    if train:
        optimizer.zero_grad(set_to_none=True)

    # Stash for deferred CPU save (avoids mid-graph .cpu() sync on TPU)
    _save_example_tensors = None

    master_print(f"[{phase.upper()} {epoch:03d}] Starting epoch...")

    for step, batch in enumerate(iterator, 1):
        x = batch["image"] if use_xla else batch["image"].to(device, non_blocking=True)
        x1, x2 = make_views(x, args)

        if train:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp and device.type == "cuda")):
                out = model.forward_ssl(x1, x2)
                loss = out["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if use_xla:
                xm.optimizer_step(optimizer, barrier=False)
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                xm.mark_step()
                if step == 1:
                    master_print(f"[{phase.upper()} {epoch:03d}] Step 1 mark_step done — XLA graph compiled.")
            else:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        else:
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp and device.type == "cuda")):
                    out = model.forward_ssl(x1, x2)
            if use_xla:
                xm.mark_step()

        if use_xla and step % 10 == 0:
            master_print(f"[{phase.upper()} {epoch:03d}] Step {step} done...")

        vals = torch.stack([torch.nan_to_num(out[k].detach().float(), nan=0.0, posinf=1e4, neginf=-1e4) for k in keys])
        meter[:-1] += vals
        meter[-1] += 1.0

        # Stash tensors for deferred save — do NOT call .cpu() inside the XLA graph
        if train and step == 1 and args.save_examples_every > 0 and epoch % args.save_examples_every == 0 and is_master_process():
            _save_example_tensors = {
                "epoch": epoch,
                "region_original": out["region_original"][:8].detach(),
                "region_reconstructed": out["region_reconstructed"][:8].detach(),
                "region_mask": out["region_mask"][:8].detach(),
            }

        if not use_xla and hasattr(iterator, "set_postfix"):
            denom = max(1.0, float(meter[-1].detach().cpu()))
            iterator.set_postfix(loss=f"{float(meter[0].detach().cpu())/denom:.4f}", mse=f"{float(meter[1].detach().cpu())/denom:.4f}")

    # All steps done — reduce and sync before .cpu()
    if use_xla:
        meter = xm.all_reduce(xm.REDUCE_SUM, meter)
        xm.mark_step()
        master_print(f"[{phase.upper()} {epoch:03d}] All-reduce done.")

    # Now safe to move tensors to CPU
    if _save_example_tensors is not None:
        ex_path = Path(args.output_dir) / "examples" / f"region_recon_epoch_{epoch:03d}.pt"
        ex_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in _save_example_tensors.items()}, ex_path)

    arr = meter.detach().cpu().numpy()
    denom = max(1.0, float(arr[-1]))
    out = {k: float(arr[i] / denom) for i, k in enumerate(keys)}
    out["skipped_batches"] = 0.0
    return out


def make_loader(dataset, batch_size, shuffle, drop_last, args, sampler=None, device=None):
    pin = bool(device is not None and device.type == "cuda")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=drop_last,
        persistent_workers=bool(args.num_workers > 0),
    )


def _worker(rank, args):
    use_xla = args.device == "tpu"
    if use_xla:
        device = xm.xla_device()
        world = xla_world_size()
        ordinal = xla_rank()
        args.amp = False
        # Fused mamba_ssm kernels are CUDA-oriented. On TPU, use the pure torch scan.
        if args.scan_backend == "mamba":
            args.scan_backend = "torch"
        # DataLoader workers forked from xmp.spawn conflict with PJRT → deadlock.
        # Force single-threaded data loading on TPU.
        if args.num_workers > 0:
            master_print(f"[INIT] TPU detected: overriding num_workers {args.num_workers} → 0 (fork/PJRT deadlock prevention)")
            args.num_workers = 0
    else:
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available() and not args.cpu):
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        world = 1
        ordinal = 0

    master_print(f"[INIT] rank={rank} ordinal={ordinal} world_size={world} device={device}")
    seed_everything(int(args.seed) + ordinal)
    outdir = ensure_dir(args.output_dir) if is_master_process() else Path(args.output_dir)
    if is_master_process():
        for d in ["checkpoints", "plots", "embeddings", "examples"]:
            ensure_dir(outdir / d)
    if use_xla:
        xm.rendezvous("dirs_ready")
        master_print("[INIT] rendezvous 'dirs_ready' passed.")

    if args.scan_backend == "mamba" and not HAS_MAMBA_SSM:
        raise ImportError("mamba_ssm unavailable. Use --scan_backend torch or install mamba-ssm.")

    master_print("[INIT] Building datasets...")
    train_ds, val_ds, n_channels, n_classes = build_datasets(args)
    args.n_channels = int(n_channels)
    args.n_classes = int(n_classes)
    master_print(f"[INIT] Dataset ready: train={len(train_ds)} val={len(val_ds)} classes={n_classes} channels={n_channels}")
    master_print("[INIT] Building geometry...")
    geometry = build_geometry(args)
    master_print(f"[INIT] Geometry: num_regions={geometry['num_regions']} patches_per_region={geometry['patches_per_region']}")

    if is_master_process():
        save_json(outdir / "config.json", {
            "args": vars(args),
            "geometry": {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in geometry.items()},
            "has_mamba_ssm": HAS_MAMBA_SSM,
            "has_xla": HAS_XLA,
        })
        master_print("[CONFIG]")
        master_print(json.dumps({
            "data_source": args.data_source,
            "npz_path": args.npz_path,
            "dataset": args.dataset,
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "patch_encoder": args.patch_encoder,
            "pretrained": args.pretrained,
            "scan_backend": args.scan_backend,
            "has_mamba_ssm": HAS_MAMBA_SSM,
            "has_xla": HAS_XLA,
            "world_size": world,
            "patch_grid": [geometry["patch_grid_h"], geometry["patch_grid_w"]],
            "region_patch": [geometry["region_patch_h"], geometry["region_patch_w"]],
            "region_stride": [geometry["region_stride_h"], geometry["region_stride_w"]],
            "region_grid": [geometry["region_grid_h"], geometry["region_grid_w"]],
            "num_regions": geometry["num_regions"],
            "patches_per_region": geometry["patches_per_region"],
            "n_classes": n_classes,
            "device": str(device),
            "output_dir": str(outdir),
        }, indent=2))

    if use_xla:
        train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=ordinal, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=ordinal, shuffle=False, drop_last=False)
        eval_train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=ordinal, shuffle=False, drop_last=False)
        eval_val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=ordinal, shuffle=False, drop_last=False)
    else:
        train_sampler = val_sampler = eval_train_sampler = eval_val_sampler = None

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, drop_last=True, args=args, sampler=train_sampler, device=device)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False, drop_last=False, args=args, sampler=val_sampler, device=device)
    eval_train_loader = make_loader(train_ds, args.eval_batch_size, shuffle=False, drop_last=False, args=args, sampler=eval_train_sampler, device=device)
    eval_val_loader = make_loader(val_ds, args.eval_batch_size, shuffle=False, drop_last=False, args=args, sampler=eval_val_sampler, device=device)

    master_print("[INIT] Building model...")
    model = FastTopoMambaFoundation(args, n_channels, geometry).to(device)
    master_print(f"[INIT] Trainable parameters: {count_parameters(model):.2f}M")
    if use_xla:
        # Warm the device allocation so graph compilation starts clean
        xm.mark_step()
        master_print("[INIT] Model moved to TPU, mark_step done.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = cosine_warmup_scheduler(
        optimizer,
        max(1, len(train_loader) * args.warmup_epochs),
        max(1, len(train_loader) * args.epochs),
        args.min_lr_ratio,
    )
    scaler = None  # XLA does not use CUDA GradScaler; CUDA path uses normal fp32 here for stability.
    master_print(f"[INIT] Optimizer=AdamW lr={args.lr} wd={args.weight_decay} | batch_size={args.batch_size} world={world} global_batch={args.batch_size * world}")
    master_print(f"[INIT] Train steps/epoch={len(train_loader)} | warmup_epochs={args.warmup_epochs} total_epochs={args.epochs}")
    master_print("[INIT] *** XLA graph will compile on the first forward+backward pass. This may take 5-15 min. Please wait... ***")

    history = []
    best_metric = -float("inf") if args.best_metric == "knn" else float("inf")
    start_epoch = 1
    if args.resume != "none":
        ckpt_path = outdir / "checkpoints" / ("last.pt" if args.resume == "last" else "best.pt") if args.resume in {"last", "best"} else Path(args.resume)
        if ckpt_path.exists():
            c = load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, device)
            start_epoch = int(c["epoch"]) + 1
            best_metric = float(c.get("best_metric", best_metric))
            history = c.get("history", [])
            master_print(f"[RESUME] {ckpt_path}, start_epoch={start_epoch}")
        else:
            master_print(f"[WARN] Resume checkpoint not found: {ckpt_path}")

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        t0 = time.time()

        tr_loader = pl.MpDeviceLoader(train_loader, device) if use_xla else train_loader
        va_loader = pl.MpDeviceLoader(val_loader, device) if use_xla else val_loader
        tr = run_epoch(model, tr_loader, optimizer, scheduler, scaler, device, args, True, epoch)
        va = run_epoch(model, va_loader, None, None, None, device, args, False, epoch)

        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "time_sec": time.time() - t0}
        for k, v in tr.items():
            row[f"train_{k}"] = v
        for k, v in va.items():
            row[f"val_{k}"] = v

        if epoch == 1 or (args.analysis_every > 0 and epoch % args.analysis_every == 0):
            et_loader = pl.MpDeviceLoader(eval_train_loader, device) if use_xla else eval_train_loader
            ev_loader = pl.MpDeviceLoader(eval_val_loader, device) if use_xla else eval_val_loader
            tz, ty = extract_embeddings(model, et_loader, device, args, args.max_eval_samples)
            vz, vy = extract_embeddings(model, ev_loader, device, args, args.max_eval_samples)
            knn = knn_accuracy(tz, ty, vz, vy, args.knn_k_eval)
            row["val_knn_acc"] = knn
            if is_master_process():
                torch.save({"epoch": epoch, "train_z": tz, "train_y": ty, "val_z": vz, "val_y": vy}, outdir / "embeddings" / f"embeddings_epoch_{epoch:03d}.pt")
                save_pca_plot(vz, vy, outdir / "plots" / f"pca_epoch_{epoch:03d}.png", f"PCA epoch {epoch}, kNN={knn:.2f}")
                if args.tsne:
                    save_tsne_plot(vz, vy, outdir / "plots" / f"tsne_epoch_{epoch:03d}.png", f"t-SNE epoch {epoch}, kNN={knn:.2f}", args.tsne_max_samples)

        row.setdefault("val_knn_acc", float("nan"))
        history.append(row)
        current = float(row.get("val_knn_acc", -float("inf"))) if args.best_metric == "knn" else float(row["val_total"])
        is_best = current > best_metric if args.best_metric == "knn" else current < best_metric
        if is_best:
            best_metric = current
            save_checkpoint(outdir / "checkpoints" / "best.pt", model, optimizer, scheduler, scaler, epoch, best_metric, args, geometry, history)
            master_print(f"[INFO] Saved best.pt at epoch {epoch}, {args.best_metric}={best_metric:.4f}")
        save_checkpoint(outdir / "checkpoints" / "last.pt", model, optimizer, scheduler, scaler, epoch, best_metric, args, geometry, history)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(outdir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, scaler, epoch, best_metric, args, geometry, history)

        if is_master_process():
            append_csv(outdir / "metrics.csv", row)
            save_json(outdir / "metrics.json", history)
            save_curves(history, outdir / "plots" / "loss_curves.png")
            master_print(
                f"[EPOCH {epoch:03d}] train_total={row['train_total']:.4f} val_total={row['val_total']:.4f} "
                f"train_mse={row['train_recon_mse']:.4f} val_mse={row['val_recon_mse']:.4f} "
                f"train_ctr={row['train_contrastive']:.4f} val_ctr={row['val_contrastive']:.4f} "
                f"val_var={row['val_emb_var']:.6f} pos={row['val_pos_cos']:.3f} neg={row['val_neg_cos']:.3f} "
                + (f"knn={row['val_knn_acc']:.2f}" if "val_knn_acc" in row else "")
            )
        if use_xla:
            xm.rendezvous(f"epoch_{epoch}_done")

    master_print(f"[DONE] Outputs saved to {outdir}")


def main():
    args = build_parser().parse_args()
    if args.cpu:
        args.device = "cpu"
    if args.device == "tpu":
        if not HAS_XLA:
            raise ImportError("torch_xla is required for --device tpu")
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        if args.xla_bf16:
            os.environ["XLA_USE_BF16"] = "1"
        if args.num_cores > 0:
            xmp.spawn(_worker, args=(args,), nprocs=args.num_cores, start_method=args.xla_spawn_method)
        else:
            xmp.spawn(_worker, args=(args,), start_method=args.xla_spawn_method)
    else:
        _worker(0, args)


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser("Fast TopoMamba foundation training pipeline, TPU v4-8 friendly")
    p.add_argument("--data_source", type=str, default="npz", choices=["npz", "medmnist", "folder"])
    p.add_argument("--npz_path", type=str, default="")
    p.add_argument("--dataset", type=str, default="pathmnist")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--download", type=bool_arg, default=True)
    p.add_argument("--folder_root", type=str, default="")
    p.add_argument("--val_split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--force_split", type=bool_arg, default=False)
    p.add_argument("--val_fraction", type=float, default=0.15)

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--force_rgb", type=bool_arg, default=True)
    p.add_argument("--normalize_imagenet", type=bool_arg, default=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="tpu", choices=["auto", "cpu", "cuda", "tpu"])
    p.add_argument("--cpu", type=bool_arg, default=False)
    p.add_argument("--amp", type=bool_arg, default=False)
    p.add_argument("--show_progress", type=bool_arg, default=False)
    p.add_argument("--num_cores", type=int, default=0, help="0 means let torch_xla choose all visible TPU devices")
    p.add_argument("--xla_bf16", type=bool_arg, default=True)
    p.add_argument("--xla_spawn_method", type=str, default="fork", choices=["fork", "spawn"])

    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--patch_encoder", type=str, default="resnet18", choices=["pixel_mlp", "resnet18", "resnet34", "resnet50", "convnext_tiny", "efficientnet_b0"])
    p.add_argument("--pretrained", type=bool_arg, default=True)
    p.add_argument("--patch_embed_dim", type=int, default=384)
    p.add_argument("--patch_mlp_hidden", type=int, default=768)

    p.add_argument("--patches_per_region", type=int, default=16)
    p.add_argument("--region_patch_h", type=int, default=0)
    p.add_argument("--region_patch_w", type=int, default=0)
    p.add_argument("--region_stride_h", type=int, default=0)
    p.add_argument("--region_stride_w", type=int, default=0)

    p.add_argument("--graph_pe_dim", type=int, default=16)
    p.add_argument("--knn_k", type=int, default=8)
    p.add_argument("--region_embed_dim", type=int, default=384)
    p.add_argument("--model_dim", type=int, default=384)
    p.add_argument("--embedding_dim", type=int, default=384)
    p.add_argument("--agg_hidden_dim", type=int, default=128)
    p.add_argument("--vmamba_depth", type=int, default=4)
    p.add_argument("--vmamba_state_dim", type=int, default=48)
    p.add_argument("--vmamba_dt_rank", type=int, default=24)
    p.add_argument("--vmamba_conv_kernel", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--scan_backend", type=str, default="torch", choices=["mamba", "torch"])
    p.add_argument("--mamba_expand", type=int, default=2)

    p.add_argument("--mask_ratio", type=float, default=0.50)
    p.add_argument("--mask_block_min", type=int, default=1)
    p.add_argument("--mask_block_max", type=int, default=3)
    p.add_argument("--w_recon", type=float, default=1.0)
    p.add_argument("--w_contrastive", type=float, default=0.2)
    p.add_argument("--w_topology", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.10)
    p.add_argument("--contrastive_flip", type=bool_arg, default=True)
    p.add_argument("--contrastive_vflip", type=bool_arg, default=False)
    p.add_argument("--view_noise_std", type=float, default=0.0)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16, help="Per-TPU-process batch size, not global batch size")
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--analysis_every", type=int, default=10)
    p.add_argument("--max_eval_samples", type=int, default=5000)
    p.add_argument("--knn_k_eval", type=int, default=5)
    p.add_argument("--tsne", type=bool_arg, default=False)
    p.add_argument("--tsne_max_samples", type=int, default=1500)
    p.add_argument("--save_examples_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--best_metric", type=str, default="knn", choices=["knn", "loss"])
    p.add_argument("--resume", type=str, default="none")
    return p


if __name__ == "__main__":
    main()
