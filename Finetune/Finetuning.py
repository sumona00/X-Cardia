from __future__ import annotations
import os, argparse, gc, random, math
from typing import Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler

from sklearn.metrics import roc_auc_score

try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    mlflow = None
    _HAS_MLFLOW = False

try:
    import deepspeed
except Exception as e:
    raise ImportError(
        "DeepSpeed is required for this script. Install it (e.g., pip install deepspeed)."
    ) from e

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# -------------------- Reproducibility --------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------- Distributed helpers --------------------
def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

    if world_size > 1:
        deepspeed.init_distributed(dist_backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(0)
    return device, local_rank

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0

def ddp_average_scalar(value: float) -> float:
    if not dist.is_initialized():
        return float(value)
    dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    t = torch.tensor([float(value)], device=dev, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / dist.get_world_size()
    return float(t.item())

def ddp_allgather_np(arr: np.ndarray) -> np.ndarray:
    if not dist.is_initialized():
        return arr
    obj_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(obj_list, arr)
    pieces = [x for x in obj_list if isinstance(x, np.ndarray) and x.size > 0]
    if not pieces:
        return np.array([], dtype=arr.dtype if isinstance(arr, np.ndarray) else np.float32)
    return np.concatenate(pieces, axis=0)

# -------------------- Args --------------------
def parse_args():
    p = argparse.ArgumentParser()

    # CSVs
    p.add_argument("--train_csv", type=str, default="train.csv")
    p.add_argument("--val_csv",   type=str, default="val.csv")
    p.add_argument("--test_csv",  type=str, default="test.csv")

    # IO
    p.add_argument("--input_dir", type=str, default="input_dir", help="Where checkpoints live (if any).")
    p.add_argument("--output_dir", type=str, default="outputs", help="Where run artifacts/checkpoints are saved.")

    # Data loading
    p.add_argument("--dtype", type=str, default="float64", choices=["float16", "float32", "float64"],
                   help="Raw buffer dtype on disk/blob (will be converted to float32).")
    p.add_argument("--volume_shape", type=str, default="164,164,164",
                   help="Preferred D,H,W (comma-separated). If reshape fails, falls back to cube root.")
    p.add_argument("--path_mode", type=str, default="local", choices=["local", "azure"],
                   help="Where to read `Path` from: local filesystem or Azure Blob.")
    p.add_argument("--local_path_prefix", type=str, default="",
                   help="Optional prefix prepended to CSV Path for local mode (e.g., /data/ct_volumes).")
    p.add_argument("--azure_account_url", type=str, default=os.environ.get("AZURE_STORAGE_ACCOUNT_URL", ""),
                   help="e.g. https://<account>.blob.core.windows.net")
    p.add_argument("--azure_container", type=str, default=os.environ.get("AZURE_STORAGE_CONTAINER", ""),
                   help="Blob container name.")
    p.add_argument("--azure_credential_mode", type=str, default=os.environ.get("AZURE_CREDENTIAL_MODE", "default"),
                   choices=["default", "managed_identity"],
                   help="Credential type for Azure SDK. 'default' uses DefaultAzureCredential.")
    p.add_argument("--azure_managed_identity_client_id", type=str,
                   default=os.environ.get("AZURE_MANAGED_IDENTITY_CLIENT_ID", ""),
                   help="Only if using managed identity and you need to specify client id.")

    # Train
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)

    # DeepSpeed
    p.add_argument("--grad_accum_steps", type=int, default=2)

    # Early stop
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--min_epochs", type=int, default=6)

    # init / ckpt
    p.add_argument("--pretrained_ckpt", type=str, default="best_NW_Head_05.pth")
    p.add_argument("--init", choices=["pretrained", "scratch"], default="pretrained")

    # sweep controls
    p.add_argument("--fractions", type=str, default="1.0",
                   help="Comma-separated train fractions, e.g. '0.1,0.3,0.7'")
    p.add_argument("--seeds", type=str, default="123,456,789",
                   help="Comma-separated seeds, e.g. '123,456,789'")

    # MLflow
    p.add_argument("--use_mlflow", action="store_true",
                   help="Enable MLflow logging if mlflow is installed/configured.")
    p.add_argument("--mlflow_experiment", type=str, default="fraction_sweep",
                   help="MLflow experiment name (optional).")

    return p.parse_args()

args = parse_args()

# env/setup
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
os.makedirs(args.output_dir, exist_ok=True)

device, local_rank = setup_distributed()
torch.autograd.set_detect_anomaly(False)

# -------------------- Storage backend --------------------
class StorageBackend:
    def read_bytes(self, key: str) -> bytes:
        raise NotImplementedError

class LocalFileStorage(StorageBackend):
    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    def read_bytes(self, key: str) -> bytes:
        path = os.path.join(self.prefix, key) if self.prefix else key
        with open(path, "rb") as f:
            return f.read()

class AzureBlobStorage(StorageBackend):
    def __init__(self, account_url: str, container: str, credential_mode: str, managed_identity_client_id: str):
        if not account_url or not container:
            raise ValueError("For Azure mode, set --azure_account_url and --azure_container (or env vars).")

        # Import only if needed
        from azure.storage.blob import BlobServiceClient
        from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

        if credential_mode == "managed_identity":
            if managed_identity_client_id:
                cred = ManagedIdentityCredential(client_id=managed_identity_client_id)
            else:
                cred = ManagedIdentityCredential()
        else:
            cred = DefaultAzureCredential()

        self.container = container
        self.blob_service = BlobServiceClient(account_url=account_url, credential=cred)

    def read_bytes(self, key: str) -> bytes:
        bc = self.blob_service.get_blob_client(self.container, key)
        return bc.download_blob().readall()

storage: StorageBackend
if args.path_mode == "azure":
    storage = AzureBlobStorage(
        account_url=args.azure_account_url,
        container=args.azure_container,
        credential_mode=args.azure_credential_mode,
        managed_identity_client_id=args.azure_managed_identity_client_id,
    )
else:
    storage = LocalFileStorage(prefix=args.local_path_prefix)

# -------------------- Targets --------------------
target_cols = [
    "lvef_lte_45_flag",
    "lvwt_gte_13_flag",
    "aortic_stenosis_moderate_severe_flag",
    "aortic_regurgitation_moderate_severe_flag",
    "mitral_regurgitation_moderate_severe_flag",
    "tricuspid_regurgitation_moderate_severe_flag",
    "pulmonary_regurgitation_moderate_severe_flag",
    "pasp_gte_45_flag",
    "tr_max_gte_32_flag",
    "shd_flag",
]
sev_map = {"none": 0, "presumed none": -1, "mild": 0, "moderate": 1, "severe": 1}

# -------------------- Data --------------------
def _parse_shape(s: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("--volume_shape must be 'D,H,W', e.g. '164,164,164'")
    return parts[0], parts[1], parts[2]

VOL_D, VOL_H, VOL_W = _parse_shape(args.volume_shape)

class EchoCTDataset(Dataset):
    """
    Returns (volume, labels_vec)
    labels_vec: int64 shape (T,) with {-1,0,1}
    """
    def __init__(self, df: pd.DataFrame, storage_backend: StorageBackend):
        self.df = df.reset_index(drop=True)
        self.storage = storage_backend
        self.raw_dtype = np.dtype(args.dtype)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        key = row["Path"]

        raw = self.storage.read_bytes(key)
        arr = np.frombuffer(raw, dtype=self.raw_dtype).astype(np.float32)

        # Preferred fixed shape; if that fails, fallback to cube
        try:
            vol = arr.reshape(VOL_D, VOL_H, VOL_W)
        except Exception:
            d = int(round(arr.size ** (1 / 3)))
            vol = arr.reshape(d, d, d)

        vol = torch.from_numpy(vol).unsqueeze(0)  # (1, D, H, W)

        labels = [int(row[f"{col}_label"]) for col in target_cols]
        labels = torch.tensor(labels, dtype=torch.int64)
        return vol, labels

def prepare_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in target_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(-1).astype(float)
            df[col + "_label"] = df[col].astype(int)
        else:
            df[col] = df[col].fillna("N/A").astype(str).str.lower()
            df[col + "_label"] = df[col].map(lambda x: sev_map.get(x, -1)).astype(int)
    assert "Path" in df.columns, "CSV must contain a 'Path' column."
    return df

def drop_all_missing(df: pd.DataFrame) -> pd.DataFrame:
    mask = np.zeros(len(df), dtype=bool)
    for col in target_cols:
        mask |= (df[f"{col}_label"] != -1).values
    return df[mask].reset_index(drop=True)

train_df = drop_all_missing(prepare_dataframe(args.train_csv))
val_df   = drop_all_missing(prepare_dataframe(args.val_csv))
test_df  = drop_all_missing(prepare_dataframe(args.test_csv))

val_set  = EchoCTDataset(val_df, storage)
test_set = EchoCTDataset(test_df, storage)

# -------------------- Model --------------------
from resnet import generate_model  

class MultiTaskModel(nn.Module):
    def __init__(self, backbone: nn.Module, n_tasks: int):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = 128

        if hasattr(self.backbone, "fc") and isinstance(self.backbone.fc, nn.Linear):
            in_f = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_f, self.feature_dim)

        self.heads = nn.ModuleList([nn.Linear(self.feature_dim, 1) for _ in range(n_tasks)])

    def forward(self, x):
        z = self.backbone(x)  # (B,128)
        logits = torch.cat([h(z) for h in self.heads], dim=1)  # (B,T)
        return logits

def load_backbone_with_ckpt_or_fail(backbone: nn.Module, ckpt_path: str) -> None:
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    state_dict = torch.load(ckpt_path, map_location="cpu")
    new_sd = {}
    for k, v in state_dict.items():
        nk = (k.replace("module.", "")
               .replace("backbone.", "")
               .replace("model.", ""))
        new_sd[nk] = v

    missing, unexpected = backbone.load_state_dict(new_sd, strict=False)
    if is_main():
        print(f"[CKPT LOAD] Loaded from {ckpt_path}")
        print(f"[CKPT LOAD] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

def build_model_for_finetune(ckpt_path: Optional[str], init_mode: str) -> nn.Module:
    backbone = generate_model(model_depth=50, n_input_channels=1, n_classes=128)

    if init_mode == "pretrained":
        assert ckpt_path is not None and os.path.isfile(ckpt_path), f"Missing ckpt: {ckpt_path}"
        load_backbone_with_ckpt_or_fail(backbone, ckpt_path)
    else:
        if is_main():
            print("[INIT] Training from scratch (no pre-trained weights).")
    if hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Linear):
        in_f = backbone.fc.in_features
        backbone.fc = nn.Linear(in_f, 128)

    # Freezing policy
    if init_mode == "pretrained":
        for name, p in backbone.named_parameters():
            p.requires_grad = ("layer4" in name) or ("fc" in name)
    else:
        for p in backbone.parameters():
            p.requires_grad = True

    return MultiTaskModel(backbone, n_tasks=len(target_cols))

# -------------------- Subset broadcast --------------------
def broadcast_subset_indices(n_total: int, frac: float, seed: int) -> np.ndarray:
    """Rank0 samples indices; broadcast to all ranks (CUDA tensor for NCCL)."""
    k = max(1, int(math.floor(frac * n_total)))

    if not dist.is_initialized():
        rng = np.random.RandomState(seed + int(round(frac * 1000)))
        idx = rng.choice(n_total, size=k, replace=False)
        return np.sort(idx).astype(np.int64)

    rank = dist.get_rank()
    dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")

    if rank == 0:
        rng = np.random.RandomState(seed + int(round(frac * 1000)))
        idx = np.sort(rng.choice(n_total, size=k, replace=False)).astype(np.int64)
        idx_t = torch.from_numpy(idx).to(device=dev)
        len_buf = torch.tensor([idx_t.numel()], dtype=torch.int64, device=dev)
    else:
        len_buf = torch.empty(1, dtype=torch.int64, device=dev)

    dist.broadcast(len_buf, src=0)
    k = int(len_buf.item())

    if rank != 0:
        idx_t = torch.empty(k, dtype=torch.int64, device=dev)

    dist.broadcast(idx_t, src=0)
    return idx_t.cpu().numpy()

# -------------------- Train/Eval for one (seed, fraction) --------------------
def train_eval_one_fraction_seed(
    train_df: pd.DataFrame,
    frac: float,
    base_output: str,
    seed: int,
) -> float:
    """Returns test_macro_auc_mean (float), broadcasted to all ranks."""
    set_seed(seed)

    full_train_set = EchoCTDataset(train_df, storage)
    subset_indices = broadcast_subset_indices(len(full_train_set), frac, seed)
    sub_train_set = Subset(full_train_set, subset_indices.tolist())

    # samplers/loaders
    if dist.is_initialized():
        ws = dist.get_world_size()
        rk = dist.get_rank()
        train_sampler = DistributedSampler(sub_train_set, num_replicas=ws, rank=rk, shuffle=True, drop_last=False)
        val_sampler   = DistributedSampler(val_set,       num_replicas=ws, rank=rk, shuffle=False, drop_last=False)
        test_sampler  = DistributedSampler(test_set,      num_replicas=ws, rank=rk, shuffle=False, drop_last=False)
    else:
        train_sampler = RandomSampler(sub_train_set)
        val_sampler   = SequentialSampler(val_set)
        test_sampler  = SequentialSampler(test_set)

    train_loader = DataLoader(
        sub_train_set, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=False
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=False
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, sampler=test_sampler,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=False
    )

    ckpt_path = os.path.join(args.input_dir, args.pretrained_ckpt) if args.init == "pretrained" else None
    model = build_model_for_finetune(ckpt_path, args.init)

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "zero_optimization": {"stage": 2},
        "bf16": {"enabled": False},
        "fp16": {"enabled": False},
        "gradient_clipping": 1.0,
    }

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    model_engine, optimizer_engine, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.n_epochs
    base_lr = args.lr if args.lr else 5e-4
    max_lr = max(base_lr, 1e-3)

    scheduler = OneCycleLR(
        optimizer_engine.optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=1.0,
        final_div_factor=10.0,
    )

    def estimate_pos_weight(df):
        pw = []
        for col in target_cols:
            v = df[f"{col}_label"].values
            pos = (v == 1).sum()
            neg = (v == 0).sum()
            w = (neg / max(1, pos)) if pos > 0 else 1.0
            pw.append(w)
        return torch.tensor(pw, dtype=torch.float32, device=model_engine.device)

    pos_weight = estimate_pos_weight(train_df)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    def forward_with_loss(batch):
        x, y = batch
        x = x.to(model_engine.device, non_blocking=True)
        y = y.to(model_engine.device, non_blocking=True)

        logits = model_engine(x)
        y_float = (y == 1).float()
        valid_mask = (y != -1).float()

        loss_all = criterion(logits, y_float)
        loss_masked = (loss_all * valid_mask).sum()
        n_valid = int(valid_mask.sum().item())
        if n_valid == 0:
            return torch.zeros((), device=logits.device, dtype=loss_masked.dtype), 0
        return loss_masked / n_valid, n_valid

    def evaluate(loader: DataLoader, split_name: str) -> Tuple[float, float]:
        """Returns (macro_auc_mean, macro_auc_std_across_tasks) on rank0; broadcast to all ranks."""
        model_engine.eval()

        preds_per_task = [[] for _ in range(len(target_cols))]
        trues_per_task = [[] for _ in range(len(target_cols))]

        with torch.no_grad():
            for x, y in loader:
                x = x.to(model_engine.device, non_blocking=True)
                y = y.to(model_engine.device, non_blocking=True)
                probs = torch.sigmoid(model_engine(x)).detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()

                for t in range(len(target_cols)):
                    mask = (y_np[:, t] != -1)
                    if mask.any():
                        preds_per_task[t].append(probs[mask, t])
                        trues_per_task[t].append((y_np[mask, t] == 1).astype(np.int64))

        local_preds = [np.concatenate(p) if p else np.array([], np.float32) for p in preds_per_task]
        local_trues = [np.concatenate(t) if t else np.array([], np.int64) for t in trues_per_task]

        macro_list = []
        per_task_auc = {}

        for t, name in enumerate(target_cols):
            gp = ddp_allgather_np(local_preds[t])
            gt = ddp_allgather_np(local_trues[t])

            auc = None
            if gp.size > 0 and np.unique(gt).size == 2:
                try:
                    auc = float(roc_auc_score(gt, gp))
                except Exception:
                    auc = None

            if is_main() and (auc is not None):
                per_task_auc[name] = auc
                macro_list.append(auc)

        if is_main():
            macro_mean = float(np.mean(macro_list)) if macro_list else 0.0
            macro_std  = float(np.std(macro_list, ddof=1)) if len(macro_list) > 1 else 0.0

            tag = f"p{int(round(frac*100))}_s{seed}"
            if args.use_mlflow and _HAS_MLFLOW:
                for k, v in per_task_auc.items():
                    mlflow.log_metric(f"{split_name}_auc_{k}_{tag}", v)
                mlflow.log_metric(f"{split_name}_macro_auc_mean_{tag}", macro_mean)
                mlflow.log_metric(f"{split_name}_macro_auc_std_{tag}", macro_std)

            print(f"[{split_name.upper()} frac={frac:.2f} seed={seed}] macro AUC = {macro_mean:.4f} ± {macro_std:.4f}")
        else:
            macro_mean, macro_std = 0.0, 0.0

        buf = torch.tensor([macro_mean, macro_std], device=model_engine.device, dtype=torch.float32)
        if dist.is_initialized():
            dist.broadcast(buf, src=0)
        return float(buf[0].item()), float(buf[1].item())

    # ---- MLflow child run for this (seed, frac) ----
    run_name = f"train_frac_{int(round(frac*100))}_seed_{seed}"
    out_dir = os.path.join(base_output, run_name)
    os.makedirs(out_dir, exist_ok=True)

    if is_main() and args.use_mlflow and _HAS_MLFLOW:
        mlflow.start_run(run_name=run_name, nested=True)
        mlflow.log_param("train_frac", frac)
        mlflow.log_param("seed", seed)
        mlflow.log_param("train_subset_size", len(subset_indices))
        mlflow.log_param("init", args.init)

    # ---- Train loop ----
    best_macro_val_auc = -1.0
    epochs_no_improve = 0

    for epoch in range(1, args.n_epochs + 1):
        model_engine.train()
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        total_loss_val, total_valid = 0.0, 0

        for batch in train_loader:
            model_engine.zero_grad()
            loss, n_valid = forward_with_loss(batch)
            if n_valid == 0:
                continue
            model_engine.backward(loss)
            model_engine.step()
            scheduler.step()

            total_loss_val += float(loss.detach().item()) * n_valid
            total_valid += n_valid

            del loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        local_avg = (total_loss_val / max(1, total_valid)) if total_valid > 0 else 0.0
        avg_loss = ddp_average_scalar(local_avg)

        if is_main():
            tag = f"p{int(round(frac*100))}_s{seed}"
            lr_now = scheduler.get_last_lr()[0]
            if args.use_mlflow and _HAS_MLFLOW:
                mlflow.log_metric(f"train_loss_{tag}", avg_loss, step=epoch)
                mlflow.log_metric(f"lr_{tag}", lr_now, step=epoch)
            print(f"[{run_name} | Epoch {epoch}] train_loss(avg) = {avg_loss:.6f} | lr={lr_now:.3e}")
            print(f"[{run_name} | Epoch {epoch}] running validation...")

        val_macro_mean, _ = evaluate(val_loader, "val")

        improved = False
        if is_main():
            if (epoch >= 1) and (val_macro_mean > best_macro_val_auc + args.min_delta):
                best_macro_val_auc = val_macro_mean
                epochs_no_improve = 0
                ckpt_out = os.path.join(out_dir, "best_model.pt")
                model_state = model_engine.module.state_dict()
                torch.save({"model": model_state, "epoch": epoch, "macro_val_auc": best_macro_val_auc}, ckpt_out)
                print(f"[{run_name} | Epoch {epoch}] New best val macro AUC {best_macro_val_auc:.4f} → saved {ckpt_out}")
                improved = True
            else:
                epochs_no_improve += 1

        stop_flag = torch.tensor([0], device=model_engine.device, dtype=torch.int32)
        if is_main():
            should_consider = (epoch >= args.min_epochs)
            if (not improved) and should_consider and (epochs_no_improve >= args.patience):
                stop_flag[0] = 1

        if dist.is_initialized():
            dist.broadcast(stop_flag, src=0)
        if int(stop_flag.item()) == 1:
            if is_main():
                print(f"[{run_name}] Early stopping at epoch {epoch}; best val macro AUC={best_macro_val_auc:.4f}.")
            break

    # ---- Final TEST (load best checkpoint on rank0) ----
    if is_main():
        best_path = os.path.join(out_dir, "best_model.pt")
        if os.path.isfile(best_path):
            blob = torch.load(best_path, map_location="cpu")
            model_engine.module.load_state_dict(blob["model"], strict=False)
            print(f"[{run_name}] Loaded best checkpoint: {best_path}")
        else:
            print(f"[{run_name}] Best checkpoint not found; using current weights.")

    if dist.is_initialized():
        dist.barrier()

    test_macro_mean, test_macro_std_tasks = evaluate(test_loader, "test")

    if is_main() and args.use_mlflow and _HAS_MLFLOW:
        mlflow.log_metric("test_macro_auc_mean", test_macro_mean)
        mlflow.log_metric("test_macro_auc_std_across_tasks", test_macro_std_tasks)
        mlflow.end_run()

    return test_macro_mean

# -------------------- MAIN: loop over fractions + seeds --------------------
fractions = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]
seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

if is_main():
    print(f"[CONFIG] path_mode={args.path_mode} | init={args.init} | fractions={fractions} | seeds={seeds}")
    if args.use_mlflow and not _HAS_MLFLOW:
        print("[WARN] --use_mlflow was set but mlflow import failed; continuing without MLflow.")

if is_main() and args.use_mlflow and _HAS_MLFLOW:
    mlflow.set_experiment(args.mlflow_experiment)
    mlflow.start_run(run_name="fraction_sweep_seeded")
    mlflow.log_param("fractions", fractions)
    mlflow.log_param("seeds", seeds)
    mlflow.log_param("path_mode", args.path_mode)
    mlflow.log_param("init", args.init)

for frac in fractions:
    test_scores = []
    for seed in seeds:
        if is_main():
            print("\n" + "=" * 90)
            print(f"[RUN] frac={frac:.2f} | seed={seed}")
            print("=" * 90)

        score = train_eval_one_fraction_seed(
            train_df=train_df,
            frac=frac,
            base_output=args.output_dir,
            seed=seed,
        )
        if is_main():
            test_scores.append(score)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if is_main():
        test_scores_np = np.array(test_scores, dtype=np.float64)
        mean_ = float(test_scores_np.mean()) if test_scores_np.size else 0.0
        std_  = float(test_scores_np.std(ddof=1)) if test_scores_np.size > 1 else 0.0

        print("\n" + "#" * 90)
        print(f"[SUMMARY] frac={frac:.2f} | TEST macro-AUROC over seeds {seeds}")
        print(f"[SUMMARY] mean = {mean_:.4f} | std = {std_:.4f}")
        print(f"[SUMMARY] per-seed = {', '.join([f'{s:.4f}' for s in test_scores])}")
        print("#" * 90 + "\n")

        if args.use_mlflow and _HAS_MLFLOW:
            mlflow.log_metric(f"test_macro_auc_mean_over_seeds_p{int(round(frac*100))}", mean_)
            mlflow.log_metric(f"test_macro_auc_std_over_seeds_p{int(round(frac*100))}", std_)

if is_main() and args.use_mlflow and _HAS_MLFLOW:
    mlflow.end_run()

cleanup_distributed()
