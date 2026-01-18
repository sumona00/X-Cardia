from __future__ import annotations
import os
import argparse
import random
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
import json
from resnet import generate_model


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================== Phenotypes ==================
range_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "ivs_measurement": (0.6, 1.1),
    "lvpw_measurement": (0.6, 1.1),
    "lv_d_measurement": (3.5, 5.6),
    "lv_s_measurement": (2.0, 4.0),
}
NUM_CONCEPTS = len(range_map)


# ================== Data ==================
class CBMCustomDataset(Dataset):
    """
    Dataset:
    - expects a CSV with structured cols + concept source cols
    - optionally a "Path" column pointing to a local .npy volume (3D or [1,D,H,W])
    """

    def __init__(
        self,
        df: pd.DataFrame,
        rotate: bool = False,
        volume_shape: Tuple[int, int, int] = (164, 164, 164),
    ):
        self.df = df.reset_index(drop=True)
        self.rotate = rotate
        self.volume_shape = tuple(volume_shape)
        self.structured_cols = [
            "ECHO_lvef_value", "vad_flag", "Patients_Age", "Patients_Sex", "race_1_x",
            "e_wave_velocity", "a_wave_velocity", "e_a_ratio", "valve_prosthetic_flag",
            "aortic_valve_prosthetic_flag", "mitral_valve_prosthetic_flag",
            "tricuspid_valve_prosthetic_flag", "pulmonary_valve_prosthetic_flag",
            "ventricular_rate", "atrial_rate", "pr_interval", "qrs_duration",
            "qt_inverval", "qt_corrected", "ventricular_pacing_flag",
            "ECHO_tr_max_velocity_value", "ECHO_pericardial_effusion_value",
            "ECHO_pasp_value", "ECHO_pasp_less_rap_value", "ECHO_rap_value",
            "ECHO_lvot_area_value", "ECHO_lvot_diameter_value",
            "ECHO_lvot_vmax_value", "ECHO_lvot_vmean_value",
            "ECHO_lvot_peak_gradient_value", "ECHO_lvot_mean_gradient_value",
            "ECHO_lvot_vti_value", "ECHO_mr_vti", "ECHO_mr_peak_velocity",
            "ECHO_mv_dti_e_prime", "ECHO_mv_dti_e_prime_lateral",
            "ECHO_mv_dti_e_prime_medial", "ECHO_mv_dti_e_prime_avg",
            "ECHO_mv_dti_a_prime_lateral", "ECHO_mv_dti_a_prime_medial",
            "ECHO_mv_dti_a_prime_avg", "ECHO_e_a_ratio_calc_flag",
            "ECHO_av_mean_gradient", "ECHO_av_peak_gradient",
            "ECHO_av_peak_velocity", "ECHO_av_vti",
        ]
        for c in self.structured_cols:
            if c not in self.df.columns:
                self.df[c] = np.nan
                
        for c in self.structured_cols:
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce")

        # Impute structured
        struct_df = self.df[self.structured_cols].copy()
        imputer = KNNImputer(n_neighbors=5)
        imp = imputer.fit_transform(struct_df)
        self.df.loc[:, self.structured_cols] = imp
        self.df[self.structured_cols] = self.df[self.structured_cols].astype("float32")

        # Binarize phenotype labels from numeric ranges
        for col, (low, high) in range_map.items():
            if col not in self.df.columns:
                self.df[col] = 0
                continue
            s = pd.to_numeric(self.df[col], errors="coerce")
            inside = True
            if low is not None:
                inside = inside & (s >= low)
            if high is not None:
                inside = inside & (s <= high)
            self.df[col] = (~inside).fillna(True).astype("int8")

    def __len__(self):
        return len(self.df)

    def _load_volume(self, row: pd.Series) -> torch.Tensor:
        path = row.get("Path", None)
        if isinstance(path, str) and os.path.isfile(path):
            try:
                arr = np.load(path)
                if arr.ndim == 3:
                    vol = torch.from_numpy(arr).unsqueeze(0).float()  # [1,D,H,W]
                elif arr.ndim == 4 and arr.shape[0] == 1:
                    vol = torch.from_numpy(arr).float()
                else:
                    raise ValueError(f"Unexpected volume shape: {arr.shape}")
                return vol
            except Exception:
                pass

        d, h, w = self.volume_shape
        return torch.zeros((1, d, h, w), dtype=torch.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        vol = self._load_volume(row)
        struct = torch.tensor([float(row[c]) for c in self.structured_cols], dtype=torch.float32)
        concepts = torch.tensor([int(row[c]) for c in range_map.keys()], dtype=torch.float32)
        return vol, struct, concepts


# ================== Losses ==================
class CLIPLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = float(temperature)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, o0: torch.Tensor, o1: torch.Tensor) -> torch.Tensor:
        o0 = F.normalize(o0, dim=1)
        o1 = F.normalize(o1, dim=1)
        logits = (o0 @ o1.t()) / self.temp
        labels = torch.arange(o0.size(0), device=o0.device)
        return 0.5 * (self.ce(logits, labels) + self.ce(logits.t(), labels))


# ================== NW Head ==================
class NWHead(nn.Module):
    """Nadaraya–Watson head for multi-label concepts."""
    def __init__(self, dim: int, num_concepts: int, tau: float = 1.0, normalize: bool = True):
        super().__init__()
        self.tau = float(tau)
        self.normalize = bool(normalize)
        self.num_concepts = int(num_concepts)
        self.register_buffer("support_feats", torch.empty(0, dim))
        self.register_buffer("support_labels", torch.empty(0, num_concepts))

    @torch.no_grad()
    def set_support(self, feats: torch.Tensor, labels_multi: torch.Tensor):
        if self.normalize:
            feats = F.normalize(feats, dim=1)
        self.support_feats = feats
        self.support_labels = labels_multi

    def forward(self, query_feats: torch.Tensor):
        if self.support_feats.numel() == 0:
            probs = torch.full((query_feats.size(0), self.num_concepts), 0.5, device=query_feats.device)
            weights = torch.full((query_feats.size(0), 1), 1.0, device=query_feats.device)
            return probs, weights
        q = F.normalize(query_feats, dim=1) if self.normalize else query_feats
        s = self.support_feats
        logits = (q @ s.T) / self.tau
        weights = F.softmax(logits, dim=1)
        probs = weights @ self.support_labels
        return probs, weights


# ================== Tabular Encoder ==================
class TabularTransformerEncoder(nn.Module):
    def __init__(self, num_features: int, embed_dim: int = 256, output_dim: int = 256,
                 num_heads: int = 8, num_layers: int = 3, ff_dim: int = 1024):
        super().__init__()
        self.num_features = int(num_features)

        self.feature_weights = nn.Parameter(torch.empty(self.num_features, embed_dim))
        self.feature_bias = nn.Parameter(torch.zeros(self.num_features, embed_dim))

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True,
        )
        self.trans = nn.TransformerEncoder(enc, num_layers=num_layers)

        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, output_dim))
        self.output_dim = int(output_dim)

        nn.init.xavier_uniform_(self.feature_weights)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        x = x.unsqueeze(-1) * self.feature_weights + self.feature_bias  # [B, F, E]
        x = self.trans(x)
        x = x.mean(dim=1)
        return self.head(x)


# ================== MultiModal Model ==================
class MultiModalNW(nn.Module):
    """
    3D ResNet-50 but returns features BEFORE its final fc
    """

    def __init__(self, res3d: nn.Module, tab_enc: nn.Module, num_concepts: int,
                 embed_dim: int = 128, proj_dim: int = 256):
        super().__init__()
        self.res3d = res3d
        self.img_feat_dim = 2048
        self.fc_img = nn.Linear(self.img_feat_dim, embed_dim)
        self.tab_enc = tab_enc
        self.proj_img = nn.Sequential(nn.Linear(embed_dim, proj_dim), nn.ReLU(inplace=True), nn.Linear(proj_dim, proj_dim))
        self.proj_tab = nn.Sequential(nn.Linear(tab_enc.output_dim, proj_dim), nn.ReLU(inplace=True), nn.Linear(proj_dim, proj_dim))

        self.nw_head = NWHead(dim=proj_dim, num_concepts=num_concepts, tau=1.0, normalize=True)

    def forward(self, v3d: torch.Tensor, xtab: torch.Tensor):
        x = self.res3d.conv1(v3d)
        x = self.res3d.bn1(x)
        x = self.res3d.relu(x)
        if not getattr(self.res3d, "no_max_pool", False):
            x = self.res3d.maxpool(x)

        x = self.res3d.layer1(x)
        x = self.res3d.layer2(x)
        x = self.res3d.layer3(x)
        x = self.res3d.layer4(x)

        x = self.res3d.avgpool(x)        # [B, 2048, 1, 1, 1]
        x = x.view(x.size(0), -1)        # [B, 2048]

        f_img = self.fc_img(x)           # [B, embed_dim]
        emb_img = self.proj_img(f_img)   # [B, proj_dim]

        f_tab = self.tab_enc(xtab)       # [B, tab_dim]
        emb_tab = self.proj_tab(f_tab)   # [B, proj_dim]

        z = F.normalize(emb_img + emb_tab, dim=1)
        return emb_img, emb_tab, z


# ================== Support Bank ==================
@torch.no_grad()
def build_support_bank_multilabel(
    loader: DataLoader,
    model: MultiModalNW,
    per_concept: int = 5,
    device: torch.device = torch.device("cuda"),
    gather: bool = True,
):
    model.eval()
    feats, labs = [], []
    for vol, struct, concepts in loader:
        vol = vol.to(device, non_blocking=True)
        struct = struct.to(device, non_blocking=True)
        concepts = concepts.to(device, non_blocking=True)
        _eimg, _etab, z = model(vol, struct)
        feats.append(z.detach().cpu())
        labs.append(concepts.detach().cpu())

    X = torch.cat(feats, 0)
    Y = torch.cat(labs, 0)

    if gather and dist.is_initialized() and dist.get_world_size() > 1:
        world = dist.get_world_size()
        Xs = [torch.empty_like(X.to(device)) for _ in range(world)]
        Ys = [torch.empty_like(Y.to(device)) for _ in range(world)]
        dist.all_gather(Xs, X.to(device))
        dist.all_gather(Ys, Y.to(device))
        X = torch.cat([t.cpu() for t in Xs], 0)
        Y = torch.cat([t.cpu() for t in Ys], 0)

    N, D = X.shape
    C = Y.shape[1]
    selected_idx = set()
    Xn = X.numpy()
    Yn = Y.numpy()

    for c in range(C):
        pos_idx = np.where(Yn[:, c] > 0.5)[0]
        if len(pos_idx) == 0:
            continue
        k = min(per_concept, len(pos_idx))
        if k <= 1:
            picks = pos_idx[:k]
        else:
            Xc = Xn[pos_idx]
            km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(Xc)
            centers = km.cluster_centers_
            picks_local = []
            for cen in centers:
                j = np.argmin(((Xc - cen) ** 2).sum(1))
                picks_local.append(pos_idx[j])
            picks = np.array(picks_local, dtype=int)
        selected_idx.update(int(i) for i in np.atleast_1d(picks).tolist())

    if len(selected_idx) == 0:
        take = min(8, N)
        idx = np.random.choice(np.arange(N), size=take, replace=False)
    else:
        idx = np.array(sorted(list(selected_idx)), dtype=int)

    return X[idx].to(device).float(), Y[idx].to(device).float()


# ================== Metrics ==================
@torch.no_grad()
def evaluate_nw_metrics(model: MultiModalNW, loader: DataLoader, device: torch.device):
    model.eval()
    all_probs, all_targets = [], []
    total_bce, n_elems = 0.0, 0

    for vol, struct, concepts in loader:
        vol = vol.to(device, non_blocking=True)
        struct = struct.to(device, non_blocking=True)
        concepts = concepts.to(device, non_blocking=True)

        _eimg, _etab, z = model(vol, struct)
        probs, _ = model.nw_head(z)

        bce = F.binary_cross_entropy(probs, concepts, reduction="sum")
        total_bce += float(bce.item())
        n_elems += int(concepts.numel())

        all_probs.append(probs.detach().cpu())
        all_targets.append(concepts.detach().cpu())

    if len(all_probs) == 0:
        return None

    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    bce_mean = total_bce / max(n_elems, 1)

    y_true = targets.numpy().astype(int)
    y_score = probs.numpy()
    C = y_true.shape[1]

    aps = []
    for c in range(C):
        y_c = y_true[:, c]
        y_s = y_score[:, c]
        if y_c.max() == y_c.min():
            continue
        try:
            aps.append(average_precision_score(y_c, y_s))
        except Exception:
            continue
    mAP = float(np.mean(aps)) if len(aps) > 0 else float("nan")

    aucs = []
    for c in range(C):
        y_c = y_true[:, c]
        y_s = y_score[:, c]
        if y_c.max() == y_c.min():
            continue
        try:
            aucs.append(roc_auc_score(y_c, y_s))
        except Exception:
            continue
    auroc_macro = float(np.mean(aucs)) if len(aucs) > 0 else float("nan")

    try:
        auroc_micro = float(roc_auc_score(y_true, y_score, average="micro"))
    except Exception:
        auroc_micro = float("nan")

    y_pred_bin = (y_score >= 0.5).astype(int)
    try:
        f1_micro = float(f1_score(y_true, y_pred_bin, average="micro", zero_division=0))
    except Exception:
        f1_micro = float("nan")
    try:
        f1_macro = float(f1_score(y_true, y_pred_bin, average="macro", zero_division=0))
    except Exception:
        f1_macro = float("nan")

    return {
        "bce": bce_mean,
        "mAP": mAP,
        "auroc_micro": auroc_micro,
        "auroc_macro": auroc_macro,
        "f1_micro@0.5": f1_micro,
        "f1_macro@0.5": f1_macro,
    }


# ================== DDP setup ==================
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return True, device, local_rank
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, device, 0


# ================== Main ==================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="./runs")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-5)

    parser.add_argument("--use_clip", action="store_true", default=False)
    parser.add_argument("--clip_temp", type=float, default=0.07)
    parser.add_argument("--lambda_nw", type=float, default=0.5)

    parser.add_argument("--support_k_per_concept", type=int, default=5)
    parser.add_argument("--support_update_every", type=int, default=5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--volume_dhw", type=int, nargs=3, default=[164, 164, 164])
    parser.add_argument("--resnet_conv1_t", type=int, default=7)
    parser.add_argument("--resnet_conv1_t_stride", type=int, default=1)
    parser.add_argument("--resnet_no_max_pool", action="store_true", default=False)

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    is_ddp, device, local_rank = setup_ddp()

    # Data
    tr_path = os.path.join(args.input_dir, args.train_csv)
    vl_path = os.path.join(args.input_dir, args.val_csv)
    df_tr = pd.read_csv(tr_path)
    df_vl = pd.read_csv(vl_path)

    train_ds = CBMCustomDataset(df_tr, volume_shape=tuple(args.volume_dhw))
    val_ds = CBMCustomDataset(df_vl, volume_shape=tuple(args.volume_dhw))

    tr_samp = DistributedSampler(train_ds) if is_ddp else None
    vl_samp = DistributedSampler(val_ds, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(tr_samp is None),
        sampler=tr_samp,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=vl_samp,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    num_features = len(train_ds.structured_cols)

    # ResNet50 from resnet.py
    embed_dim = 128
    res3d = generate_model(
        model_depth=50,
        n_input_channels=1,
        n_classes=embed_dim,            
        conv1_t_size=args.resnet_conv1_t,
        conv1_t_stride=args.resnet_conv1_t_stride,
        no_max_pool=args.resnet_no_max_pool,
        shortcut_type="B",
        widen_factor=1.0,
    )

    tab_enc = TabularTransformerEncoder(num_features=num_features)

    model = MultiModalNW(
        res3d=res3d,
        tab_enc=tab_enc,
        num_concepts=NUM_CONCEPTS,
        embed_dim=embed_dim,
        proj_dim=256,
    ).to(device)

    base = model.module if hasattr(model, "module") else model
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    clip_loss = CLIPLoss(temperature=args.clip_temp) if args.use_clip else None
    optim_ = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_, T_0=10, T_mult=2, eta_min=1e-6)

    # Init support bank on rank0
    if (not is_ddp) or dist.get_rank() == 0:
        S_feat, S_lab = build_support_bank_multilabel(
            train_loader, base, per_concept=args.support_k_per_concept, device=device, gather=is_ddp
        )
        base.nw_head.set_support(S_feat, S_lab)

    if is_ddp:
        dist.broadcast(base.nw_head.support_feats, src=0)
        dist.broadcast(base.nw_head.support_labels, src=0)

    best_score = -float("inf")

    for epoch in range(args.epochs):
        if is_ddp:
            tr_samp.set_epoch(epoch)

        model.train()
        for vol, struct, concepts in train_loader:
            vol = vol.to(device, non_blocking=True)
            struct = struct.to(device, non_blocking=True)
            concepts = concepts.to(device, non_blocking=True)

            emb_img, emb_tab, z = model(vol, struct)

            loss_total = torch.tensor(0.0, device=device)
            if clip_loss is not None:
                loss_total = loss_total + clip_loss(emb_img, emb_tab)

            probs, _ = base.nw_head(z)
            loss_nw = F.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6), concepts)
            loss_total = loss_total + args.lambda_nw * loss_nw

            optim_.zero_grad(set_to_none=True)
            loss_total.backward()
            optim_.step()

        sched.step(epoch)

        # Refresh supports
        if ((epoch + 1) % args.support_update_every == 0) and ((not is_ddp) or dist.get_rank() == 0):
            S_feat, S_lab = build_support_bank_multilabel(
                train_loader, base, per_concept=args.support_k_per_concept, device=device, gather=is_ddp
            )
            base.nw_head.set_support(S_feat, S_lab)
            if is_ddp:
                dist.broadcast(base.nw_head.support_feats, src=0)
                dist.broadcast(base.nw_head.support_labels, src=0)

        # Validation
        metrics = evaluate_nw_metrics(base, val_loader, device)

        if (not is_ddp) or dist.get_rank() == 0:
            if metrics is None:
                print(f"Epoch {epoch+1:03d} | (no validation batches)")
                continue

            print(
                f"Epoch {epoch+1:03d} | "
                f"BCE: {metrics['bce']:.6f} | mAP: {metrics['mAP']:.4f} | "
                f"AUROC(micro/macro): {metrics['auroc_micro']:.4f}/{metrics['auroc_macro']:.4f} | "
                f"F1@0.5(micro/macro): {metrics['f1_micro@0.5']:.4f}/{metrics['f1_macro@0.5']:.4f}"
            )

            with open(os.path.join(args.output_dir, "val_metrics_latest.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            mAP_val = metrics.get("mAP", float("nan"))
            val_score = mAP_val if (mAP_val == mAP_val) else -metrics["bce"]

            torch.save(base.state_dict(), os.path.join(args.output_dir, "latest.pth"))
            if val_score > best_score:
                best_score = val_score
                torch.save(base.state_dict(), os.path.join(args.output_dir, "best.pth"))

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```
