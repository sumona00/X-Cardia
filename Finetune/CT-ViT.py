from __future__ import annotations

import os
import argparse
import random
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from sklearn.metrics import roc_auc_score

# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Distributed helpers
# -------------------------
def setup_distributed() -> Tuple[torch.device, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        if device.type == "cuda":
            torch.cuda.set_device(0)
    return device, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def ddp_allgather_np(arr: np.ndarray) -> np.ndarray:
    if not dist.is_initialized():
        return arr
    obj_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(obj_list, arr)
    pieces = [x for x in obj_list if isinstance(x, np.ndarray) and x.size > 0]
    if not pieces:
        return np.array([], dtype=arr.dtype)
    return np.concatenate(pieces, axis=0)


def ddp_average_scalar(x: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return float(x)
    t = torch.tensor([float(x)], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / dist.get_world_size()
    return float(t.item())


# -------------------------
# Utilities
# -------------------------
def resize_3d_to_cube(vol: torch.Tensor, out_size: int = 160) -> torch.Tensor:
    """
    vol: (B, C, D, H, W) float tensor
    returns: (B, C, out_size, out_size, out_size)
    """
    if vol.shape[-3:] == (out_size, out_size, out_size):
        return vol
    return F.interpolate(vol, size=(out_size, out_size, out_size), mode="trilinear", align_corners=False)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ============================================================
# CTViT Encoder (expects transformer.py in repo)
# ============================================================
from einops import rearrange
from einops.layers.torch import Rearrange as EinopsRearrange

# must exist in your repo (or vendor minimal implementation)
from transformer import pair, ContinuousPositionBias, Transformer, exists


class CTViT_Encoder(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        codebook_size: int,
        image_size: int,
        patch_size: int,
        temporal_patch_size: int,
        spatial_depth: int,
        temporal_depth: int,
        dim_head: int = 64,
        heads: int = 8,
        channels: int = 1,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_h, patch_w = self.patch_size
        self.temporal_patch_size = temporal_patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)

        H, W = self.image_size
        assert (H % patch_h) == 0 and (W % patch_w) == 0

        self.to_patch_emb = nn.Sequential(
            EinopsRearrange(
                "b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)",
                p1=patch_h,
                p2=patch_w,
                pt=temporal_patch_size,
            ),
            nn.LayerNorm(channels * patch_w * patch_h * temporal_patch_size),
            nn.Linear(channels * patch_w * patch_h * temporal_patch_size, dim),
            nn.LayerNorm(dim),
        )

        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=True,
        )

        self.enc_spatial_transformer = Transformer(depth=spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth=temporal_depth, **transformer_kwargs)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    @property
    def patch_height_width(self) -> Tuple[int, int]:
        return (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        h, w = self.patch_height_width
        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, "b t h w d -> (b t) (h w) d")
        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)
        tokens = self.enc_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=video_shape)
        tokens = rearrange(tokens, "(b t) (h w) d -> b t h w d", b=b, h=h, w=w)

        tokens = rearrange(tokens, "b t h w d -> (b h w) t d")
        tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape)
        tokens = rearrange(tokens, "(b h w) t d -> b t h w d", b=b, h=h, w=w)

        return tokens

    def forward(self, video: torch.Tensor, mask=None) -> torch.Tensor:
        assert video.ndim == 5, f"Expected (b,c,f,h,w), got {video.shape}"
        b, c, f, h, w = video.shape

        assert not exists(mask) or mask.shape[-1] == f
        assert (f % self.temporal_patch_size) == 0, (
            f"f={f} must be divisible by temporal_patch_size={self.temporal_patch_size}"
        )

        tokens = self.to_patch_emb(video)
        tokens = self.encode(tokens)

        tokens = rearrange(tokens, "b t h w d -> b (t h w) d")
        pooled = tokens.mean(dim=1)
        return self.mlp_head(pooled)


def extract_state_dict(weight_file: str) -> Dict[str, torch.Tensor]:
    """
    Loads a PyTorch checkpoint and returns a flat state_dict.
    Supports common wrappers like {'state_dict': ...}, {'model': ...}, etc.
    """
    ckpt = torch.load(weight_file, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "encoder", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    if not isinstance(ckpt, dict):
        raise ValueError("Unsupported checkpoint format. Expected dict-like state_dict.")
    out = {}
    for name, param in ckpt.items():
        if torch.is_tensor(param):
            out[name] = param.detach().cpu()
    return out


# ============================================================
# Head-only multitask model
# ============================================================
DEFAULT_TARGET_COLS = [
    "task1_label",
    "task2_label",
    "task3_label",
]

SEVERITY_MAP = {"none": 0, "mild": 0, "moderate": 1, "severe": 1}


class HeadOnlyMultiTask(nn.Module):
    """
    Frozen encoder -> embedding -> trainable linear heads
    """
    def __init__(self, encoder: CTViT_Encoder, emb_dim: int, n_tasks: int):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList([nn.Linear(emb_dim, 1) for _ in range(n_tasks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)  # (B, emb_dim) after encoder.mlp_head = Identity
        logits = torch.cat([h(z) for h in self.heads], dim=1)  # (B, T)
        return logits


def freeze_encoder(encoder: nn.Module):
    for p in encoder.parameters():
        p.requires_grad = False


def only_head_params(model: HeadOnlyMultiTask):
    for p in model.encoder.parameters():
        p.requires_grad = False
    return list(model.heads.parameters())


# ============================================================
# Data
# ============================================================
def prepare_dataframe(csv_path: str, target_cols: List[str], volume_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if volume_col not in df.columns:
        raise ValueError(f"CSV must contain a '{volume_col}' column.")

    for col in target_cols:
        if col not in df.columns:
            raise ValueError(f"CSV is missing required label column: {col}")

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(-1).astype(float)
            df[col + "_label"] = df[col].astype(int)
        else:
            s = df[col].fillna("N/A").astype(str).str.lower().str.strip()
            df[col + "_label"] = s.map(lambda x: SEVERITY_MAP.get(x, -1)).astype(int)

    return df


def drop_all_missing(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    mask = np.zeros(len(df), dtype=bool)
    for col in target_cols:
        mask |= (df[f"{col}_label"] != -1).values
    return df[mask].reset_index(drop=True)


class VolumeLoader:
    def __init__(
        self,
        backend: str,
        *,
        azure_account_url: Optional[str] = None,
        azure_container: Optional[str] = None,
        azure_managed_identity_client_id: Optional[str] = None,
    ):
        self.backend = backend.lower()
        if self.backend not in {"local", "azure_blob"}:
            raise ValueError("backend must be one of: {'local', 'azure_blob'}")

        self.azure_container = azure_container
        self._blob_service = None

        if self.backend == "azure_blob":
            if not azure_account_url or not azure_container:
                raise ValueError("--azure_account_url and --azure_container are required for azure_blob backend.")

            # Optional dependency only if user selects azure backend
            from azure.identity import DefaultAzureCredential
            from azure.storage.blob import BlobServiceClient

            cred_kwargs = {}
            if azure_managed_identity_client_id:
                cred_kwargs["managed_identity_client_id"] = azure_managed_identity_client_id

            credential = DefaultAzureCredential(**cred_kwargs)
            self._blob_service = BlobServiceClient(account_url=azure_account_url, credential=credential)

    def load(self, key_or_path: str) -> np.ndarray:
        if self.backend == "local":
            # Expect npy file containing (D,H,W) float32/float16/float64
            arr = np.load(key_or_path)
            return arr.astype(np.float32)

        # azure_blob
        blob_client = self._blob_service.get_blob_client(self.azure_container, key_or_path)
        raw = blob_client.download_blob().readall()

        # Prefer npy format if possible
        try:
            import io
            arr = np.load(io.BytesIO(raw))
            return arr.astype(np.float32)
        except Exception:
            # Fallback: treat as raw float32/float64 array (requires known cube size elsewhere)
            arr = np.frombuffer(raw, dtype=np.float32)
            return arr


class MultiTaskCTDataset(Dataset):
    """
    Returns:
      vol: (1, D, H, W) float32
      labels: (T,) int64 in {-1,0,1}
    """
    def __init__(
        self,
        df: pd.DataFrame,
        loader: VolumeLoader,
        target_cols: List[str],
        volume_col: str,
        expected_cube: Optional[int] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.loader = loader
        self.target_cols = target_cols
        self.volume_col = volume_col
        self.expected_cube = expected_cube

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        key = str(row[self.volume_col])

        arr = self.loader.load(key)

        # If fallback/raw buffer was used, reshape requires expected_cube
        if arr.ndim == 1:
            if not self.expected_cube:
                raise ValueError("Raw buffer volume detected. Provide --expected_cube to reshape.")
            n = self.expected_cube
            if arr.size != n * n * n:
                raise ValueError(f"Raw buffer size {arr.size} does not match expected_cube^3 = {n**3}.")
            arr = arr.reshape(n, n, n)

        if arr.ndim != 3:
            raise ValueError(f"Expected a 3D volume (D,H,W). Got shape: {arr.shape}")

        vol = torch.from_numpy(arr).unsqueeze(0)  # (1, D, H, W)

        labels = torch.tensor([int(row[f"{col}_label"]) for col in self.target_cols], dtype=torch.int64)
        return vol, labels


# ============================================================
# Loss + metrics
# ============================================================
def estimate_pos_weight(train_df: pd.DataFrame, target_cols: List[str], device: torch.device) -> torch.Tensor:
    pw = []
    for col in target_cols:
        v = train_df[f"{col}_label"].values
        pos = (v == 1).sum()
        neg = (v == 0).sum()
        w = (neg / max(1, pos)) if pos > 0 else 1.0
        pw.append(w)
    return torch.tensor(pw, dtype=torch.float32, device=device)


def masked_bce_loss(logits: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, T)
    y:      (B, T) with {-1,0,1}
    """
    y_float = (y == 1).float()
    valid = (y != -1).float()

    crit = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    loss_all = crit(logits, y_float)
    loss = (loss_all * valid).sum() / max(1.0, valid.sum())
    return loss


@torch.no_grad()
def evaluate_auc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_cols: List[str],
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()

    preds_per_task: List[List[np.ndarray]] = [[] for _ in range(len(target_cols))]
    trues_per_task: List[List[np.ndarray]] = [[] for _ in range(len(target_cols))]

    for vol, y in loader:
        vol = vol.to(device, non_blocking=True)
        vol = resize_3d_to_cube(vol, out_size=160)
        y = y.to(device, non_blocking=True)

        logits = model(vol)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        for t in range(len(target_cols)):
            mask = (y_np[:, t] != -1)
            if mask.any():
                preds_per_task[t].append(probs[mask, t].astype(np.float32))
                trues_per_task[t].append((y_np[mask, t] == 1).astype(np.int64))

    local_preds = [np.concatenate(p) if p else np.array([], np.float32) for p in preds_per_task]
    local_trues = [np.concatenate(t) if t else np.array([], np.int64) for t in trues_per_task]

    per_task_auc: Dict[str, float] = {}
    macro_list: List[float] = []

    for t, name in enumerate(target_cols):
        gp = ddp_allgather_np(local_preds[t])
        gt = ddp_allgather_np(local_trues[t])

        if gp.size > 0 and np.unique(gt).size == 2:
            auc = float(roc_auc_score(gt, gp))
            if is_main():
                per_task_auc[name] = auc
                macro_list.append(auc)

    if is_main():
        macro_mean = float(np.mean(macro_list)) if macro_list else 0.0
        macro_std = float(np.std(macro_list, ddof=1)) if len(macro_list) > 1 else 0.0
    else:
        macro_mean, macro_std = 0.0, 0.0

    buf = torch.tensor([macro_mean, macro_std], device=device, dtype=torch.float32)
    if dist.is_initialized():
        dist.broadcast(buf, src=0)

    return float(buf[0].item()), float(buf[1].item()), per_task_auc


# ============================================================
# Main: train head only + test
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()

    # CSVs
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True)

    # Columns
    p.add_argument("--volume_col", type=str, default="path", help="Column in CSV that points to volumes.")
    p.add_argument(
        "--target_cols",
        type=str,
        default=",".join(DEFAULT_TARGET_COLS),
        help="Comma-separated list of label columns.",
    )

    # Checkpoint + output
    p.add_argument("--pretrained_encoder_ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    # Training
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min_delta", type=float, default=1e-4)

    # CTViT params (must match pretrained)
    p.add_argument("--image_size", type=int, default=160)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--temporal_patch_size", type=int, default=2)
    p.add_argument("--spatial_depth", type=int, default=4)
    p.add_argument("--temporal_depth", type=int, default=4)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--dim_head", type=int, default=32)
    p.add_argument("--heads", type=int, default=8)

    # Data backend
    p.add_argument("--data_backend", type=str, default="local", choices=["local", "azure_blob"])
    p.add_argument("--expected_cube", type=int, default=None, help="If raw buffer volumes are used, set cube size (e.g., 164).")

    # Azure options (only used if data_backend=azure_blob)
    p.add_argument("--azure_account_url", type=str, default=None)
    p.add_argument("--azure_container", type=str, default=None)
    p.add_argument("--azure_managed_identity_client_id", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    target_cols = [c.strip() for c in args.target_cols.split(",") if c.strip()]
    if len(target_cols) == 0:
        raise ValueError("--target_cols must include at least one label column")

    # setup
    ensure_dir(args.output_dir)
    set_seed(args.seed)
    device, _ = setup_distributed()

    if is_main():
        ws = dist.get_world_size() if dist.is_initialized() else 1
        print(f"[DEVICE] {device} | DDP={dist.is_initialized()} | world_size={ws}")
        print(f"[DATA] backend={args.data_backend} | volume_col={args.volume_col}")
        print(f"[TASKS] {len(target_cols)} tasks: {target_cols}")

    # dataframes
    train_df = drop_all_missing(prepare_dataframe(args.train_csv, target_cols, args.volume_col), target_cols)
    val_df = drop_all_missing(prepare_dataframe(args.val_csv, target_cols, args.volume_col), target_cols)
    test_df = drop_all_missing(prepare_dataframe(args.test_csv, target_cols, args.volume_col), target_cols)

    # loader backend
    loader_backend = VolumeLoader(
        args.data_backend,
        azure_account_url=args.azure_account_url,
        azure_container=args.azure_container,
        azure_managed_identity_client_id=args.azure_managed_identity_client_id,
    )

    train_set = MultiTaskCTDataset(train_df, loader_backend, target_cols, args.volume_col, expected_cube=args.expected_cube)
    val_set = MultiTaskCTDataset(val_df, loader_backend, target_cols, args.volume_col, expected_cube=args.expected_cube)
    test_set = MultiTaskCTDataset(test_df, loader_backend, target_cols, args.volume_col, expected_cube=args.expected_cube)

    # samplers
    if dist.is_initialized():
        ws, rk = dist.get_world_size(), dist.get_rank()
        train_sampler = DistributedSampler(train_set, num_replicas=ws, rank=rk, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_set, num_replicas=ws, rank=rk, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(test_set, num_replicas=ws, rank=rk, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    # build encoder + load pretrained weights
    encoder = CTViT_Encoder(
        dim=args.dim,
        codebook_size=8192,
        image_size=args.image_size,
        patch_size=args.patch_size,
        temporal_patch_size=args.temporal_patch_size,
        spatial_depth=args.spatial_depth,
        temporal_depth=args.temporal_depth,
        dim_head=args.dim_head,
        heads=args.heads,
        channels=1,
    )

    sd = extract_state_dict(args.pretrained_encoder_ckpt)
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if is_main():
        print(f"[CKPT] Loaded encoder from: {args.pretrained_encoder_ckpt}")
        print(f"[CKPT] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    # make encoder output embeddings
    encoder.mlp_head = nn.Identity()

    # freeze encoder
    freeze_encoder(encoder)

    # build model
    model = HeadOnlyMultiTask(encoder=encoder, emb_dim=args.dim, n_tasks=len(target_cols)).to(device)

    # optimizer ONLY for heads
    optimizer = torch.optim.AdamW(only_head_params(model), lr=args.lr, weight_decay=args.weight_decay)

    # pos_weight from training data
    pos_weight = estimate_pos_weight(train_df, target_cols, device=device)

    # train loop
    best_val = -1.0
    epochs_no_improve = 0
    best_path = os.path.join(args.output_dir, "best_head_only.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        total_loss = 0.0
        n_batches = 0

        for vol, y in train_loader:
            vol = vol.to(device, non_blocking=True)
            vol = resize_3d_to_cube(vol, out_size=160)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(vol)
            loss = masked_bce_loss(logits, y, pos_weight)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().item())
            n_batches += 1

        local_loss = total_loss / max(1, n_batches)
        avg_loss = ddp_average_scalar(local_loss, device=device)

        if is_main():
            print(f"[Epoch {epoch}] train_loss={avg_loss:.6f}")

        if dist.is_initialized():
            dist.barrier()

        val_macro, val_std, _ = evaluate_auc(model, val_loader, device=device, target_cols=target_cols)

        if is_main():
            print(f"[Epoch {epoch}] val_macro_AUROC={val_macro:.4f} ± {val_std:.4f}")

        if is_main():
            if val_macro > best_val + args.min_delta:
                best_val = val_macro
                epochs_no_improve = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "best_val_macro_auc": best_val,
                        "args": vars(args),
                        "target_cols": target_cols,
                    },
                    best_path,
                )
                print(f"[Epoch {epoch}] New best val={best_val:.4f} -> saved {best_path}")
            else:
                epochs_no_improve += 1

        stop_flag = torch.tensor([0], device=device, dtype=torch.int32)
        if is_main() and (epochs_no_improve >= args.patience):
            stop_flag[0] = 1
        if dist.is_initialized():
            dist.broadcast(stop_flag, src=0)
        if int(stop_flag.item()) == 1:
            if is_main():
                print(f"[EARLY STOP] best_val_macro_AUROC={best_val:.4f}")
            break

    # test
    if is_main():
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"[TEST] Loaded best checkpoint from {best_path}")

    if dist.is_initialized():
        dist.barrier()

    test_macro, test_std, per_task = evaluate_auc(model, test_loader, device=device, target_cols=target_cols)

    if is_main():
        print("\n" + "=" * 100)
        print(f"[FINAL TEST] macro AUROC mean = {test_macro:.4f} | std(across tasks) = {test_std:.4f}")
        print("Per-task AUROC:")
        for k in sorted(per_task.keys()):
            print(f"  {k}: {per_task[k]:.4f}")
        print("=" * 100 + "\n")

    cleanup_distributed()


if __name__ == "__main__":
    main()
