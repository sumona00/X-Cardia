from __future__ import annotations

import os
import json
import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------- Plot style: paper-ready ----------
matplotlib.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.autolayout": True,
})
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# -------------------------
# IO helpers
# -------------------------
def _parse_embedding_spec(spec: str) -> Tuple[str, str, str]:
    """
    spec formats:
      - npy:/abs/or/rel/path.npy
      - npz:/abs/or/rel/path.npz:key
      - csv:/abs/or/rel/path.csv:col_prefix   (expects columns col_prefix0, col_prefix1, ... OR a single column of lists not supported)

    Returns: (kind, path, key)
    """
    if ":" not in spec:
        raise ValueError("Embedding spec must start with npy:, npz:, or csv:")
    kind, rest = spec.split(":", 1)
    kind = kind.lower().strip()

    if kind == "npy":
        return kind, rest, ""
    if kind == "npz":
        # npz:path:key
        if rest.count(":") < 1:
            raise ValueError("npz spec must be: npz:/path/file.npz:key")
        path, key = rest.split(":", 1)
        return kind, path, key
    if kind == "csv":
        if rest.count(":") < 1:
            raise ValueError("csv spec must be: csv:/path/file.csv:col_prefix")
        path, key = rest.split(":", 1)
        return kind, path, key

    raise ValueError(f"Unknown embedding spec kind: {kind}")


def load_embeddings(spec: str) -> np.ndarray:
    kind, path, key = _parse_embedding_spec(spec)
    if kind == "npy":
        X = np.load(path)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array from {path}, got shape {X.shape}")
        return X.astype(np.float32)

    if kind == "npz":
        z = np.load(path, allow_pickle=True)
        if key not in z:
            raise KeyError(f"Key '{key}' not found in {path}. Available keys: {list(z.keys())}")
        X = z[key]
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for key '{key}' in {path}, got shape {X.shape}")
        return X.astype(np.float32)

    # csv
    df = pd.read_csv(path)
    # columns like key0, key1, ...
    cols = [c for c in df.columns if c.startswith(key)]
    if len(cols) == 0:
        raise ValueError(f"No columns starting with prefix '{key}' in {path}")
    cols = sorted(cols, key=lambda c: int(c.replace(key, "")) if c.replace(key, "").isdigit() else 10**9)
    X = df[cols].to_numpy(dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D features from CSV, got {X.shape}")
    return X


def savefig(path_png: str):
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    plt.savefig(path_png, bbox_inches="tight")
    base, _ = os.path.splitext(path_png)
    plt.savefig(base + ".svg", bbox_inches="tight")
    plt.close()


# -------------------------
# Plotting
# -------------------------
def scatter_pca(
    coords: np.ndarray,
    y: np.ndarray,
    out_path: str,
    title: str,
    explained: Tuple[float, float],
    labels: Tuple[str, str, Optional[str]] = ("A", "B", None),
    max_points_per_group: int = 4000,
    seed: int = 0,
):
    """
    coords: (N,2)
    y:      (N,) group labels in {0,1,2}
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    rng = np.random.default_rng(seed)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    idx2 = np.where(y == 2)[0] if np.any(y == 2) else np.array([], dtype=int)

    # balanced subsampling
    n01 = min(len(idx0), len(idx1), max_points_per_group)
    if n01 == 0:
        raise ValueError("Need at least 1 point in each of the first two groups to plot.")

    idx0 = rng.choice(idx0, size=n01, replace=False)
    idx1 = rng.choice(idx1, size=n01, replace=False)
    chosen = [idx0, idx1]

    if idx2.size > 0:
        n2 = min(len(idx2), max_points_per_group)
        idx2 = rng.choice(idx2, size=n2, replace=False)
        chosen.append(idx2)

    idx = np.concatenate(chosen, axis=0)
    coords = coords[idx]
    y = y[idx]

    # central clipping (1–99%)
    x = coords[:, 0]
    yv = coords[:, 1]
    x_lo, x_hi = np.percentile(x, [1, 99])
    y_lo, y_hi = np.percentile(yv, [1, 99])
    mx = 0.08 * (x_hi - x_lo + 1e-8)
    my = 0.08 * (y_hi - y_lo + 1e-8)

    # light flat colors
    c0 = (0.25, 0.55, 0.85, 0.65)  # blue
    c1 = (0.95, 0.60, 0.20, 0.65)  # orange
    c2 = (0.20, 0.75, 0.45, 0.60)  # green (optional 3rd)

    ax.scatter(coords[y == 0, 0], coords[y == 0, 1], s=10, c=[c0], edgecolors="none", label=labels[0])
    ax.scatter(coords[y == 1, 0], coords[y == 1, 1], s=10, c=[c1], edgecolors="none", label=labels[1])
    if labels[2] is not None and np.any(y == 2):
        ax.scatter(coords[y == 2, 0], coords[y == 2, 1], s=10, c=[c2], edgecolors="none", label=labels[2])

    ax.set_xlim(x_lo - mx, x_hi + mx)
    ax.set_ylim(y_lo - my, y_hi + my)
    ax.set_aspect("equal", adjustable="datalim")

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title(title)

    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=(c0[0], c0[1], c0[2], 1.0),
               markeredgecolor="none", markersize=8, label=labels[0]),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=(c1[0], c1[1], c1[2], 1.0),
               markeredgecolor="none", markersize=8, label=labels[1]),
    ]
    if labels[2] is not None and np.any(y == 2):
        handles.append(
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=(c2[0], c2[1], c2[2], 1.0),
                   markeredgecolor="none", markersize=8, label=labels[2])
        )

    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=11, handletextpad=0.6, borderaxespad=0.2)
    savefig(out_path)


def run_pca(
    E_a: np.ndarray,
    E_b: np.ndarray,
    out_dir: str,
    tag: str,
    *,
    E_c: Optional[np.ndarray] = None,
    labels: Tuple[str, str, Optional[str]] = ("IMAGE", "Tabular", None),
    standardize: bool = True,
    random_state: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "pca"), exist_ok=True)

    # stack
    if E_c is None:
        X = np.vstack([E_a, E_b])
        y = np.array([0] * len(E_a) + [1] * len(E_b), dtype=np.int64)
    else:
        X = np.vstack([E_a, E_b, E_c])
        y = np.array([0] * len(E_a) + [1] * len(E_b) + [2] * len(E_c), dtype=np.int64)

    if standardize:
        X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=random_state)
    X2 = pca.fit_transform(X)
    var = pca.explained_variance_ratio_.tolist()

    # save coords
    df = pd.DataFrame({"pc1": X2[:, 0], "pc2": X2[:, 1], "group": y})
    df["label"] = np.where(df["group"].values == 0, labels[0], np.where(df["group"].values == 1, labels[1], labels[2] or "C"))
    df.to_csv(os.path.join(out_dir, f"pca_coords_{tag}.csv"), index=False)

    # save variance
    with open(os.path.join(out_dir, f"pca_variance_{tag}.json"), "w") as f:
        json.dump({"explained_variance_ratio": var}, f, indent=2)

    title = f"PCA (2D) – alignment [{tag}]"
    out_path = os.path.join(out_dir, "pca", f"pca_alignment_{tag}.png")
    scatter_pca(
        coords=X2,
        y=y,
        out_path=out_path,
        title=title,
        explained=(float(var[0]), float(var[1])),
        labels=labels,
    )
    return var


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--emb_a", type=str, required=True, help="Embedding spec for A (e.g., npz:file.npz:emb_img)")
    ap.add_argument("--emb_b", type=str, required=True, help="Embedding spec for B (e.g., npz:file.npz:emb_tab)")
    ap.add_argument("--emb_c", type=str, default=None, help="Optional embedding spec for C (e.g., npz:file.npz:emb_fused)")
    ap.add_argument("--label_a", type=str, default="IMAGE")
    ap.add_argument("--label_b", type=str, default="Tabular")
    ap.add_argument("--label_c", type=str, default="Fused")

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--no_standardize", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    E_a = load_embeddings(args.emb_a)
    E_b = load_embeddings(args.emb_b)
    E_c = load_embeddings(args.emb_c) if args.emb_c else None

    labels = (args.label_a, args.label_b, args.label_c if args.emb_c else None)

    var = run_pca(
        E_a,
        E_b,
        out_dir=args.out_dir,
        tag=args.tag,
        E_c=E_c,
        labels=labels,
        standardize=(not args.no_standardize),
        random_state=args.seed,
    )

    if True:
        print(f"[PCA] explained_variance_ratio = {var}")
        print(f"[PCA] wrote: {os.path.abspath(os.path.join(args.out_dir, 'pca'))}")


if __name__ == "__main__":
    main()
