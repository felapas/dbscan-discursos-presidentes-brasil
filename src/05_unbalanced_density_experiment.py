#!/usr/bin/env python3
"""Complementary experiment on the full unbalanced corpus (5,967 speeches, 8 presidents).

Demonstrates why a single global epsilon in DBSCAN struggles with heterogeneous
local densities (27.1x volume imbalance + stylistic spread), justifying the
balanced sampling in the main study and motivating OPTICS / HDBSCAN.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

import importlib.util
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="n_jobs value -1 overridden.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="hdbscan")

import hdbscan
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

SEED = 42
ALL_PRESIDENTS = [
    "Sarney",
    "Collor",
    "Itamar",
    "FHC",
    "Lula",
    "Dilma",
    "Temer",
    "Bolsonaro",
]

PRES_COLORS = {
    "Sarney": "#8c564b",
    "Collor": "#17becf",
    "Itamar": "#bcbd22",
    "FHC": "#1f77b4",
    "Lula": "#d62728",
    "Dilma": "#ff7f0e",
    "Temer": "#2ca02c",
    "Bolsonaro": "#9467bd",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_preprocess_module():
    """Import leakage-free feature engineering helpers from 02_preprocess_and_features.py."""
    mod_path = SRC_DIR / "02_preprocess_and_features.py"
    spec = importlib.util.spec_from_file_location("prep_mod", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_dbcv_safe(Z: np.ndarray, labels: np.ndarray, y_true: np.ndarray) -> tuple[float, bool]:
    """Compute DBCV on full Z or fallback to 2,000-point stratified subsample if needed."""
    unique_c = set(labels) - {-1}
    if len(unique_c) < 2 or int((labels != -1).sum()) < 4:
        return float("nan"), False

    t0 = time.perf_counter()
    try:
        val = hdbscan.validity.validity_index(
            np.ascontiguousarray(Z, dtype=np.float64),
            np.ascontiguousarray(labels, dtype=np.int64),
            metric="euclidean",
        )
        if np.isfinite(val):
            return float(val), False
    except Exception:
        pass

    # Fallback: stratified subsample of 2,000 points
    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(2000, len(labels)), random_state=SEED)
    idx, _ = next(sss.split(Z, y_true))
    sub_lbl = labels[idx]
    if len(set(sub_lbl) - {-1}) < 2 or int((sub_lbl != -1).sum()) < 4:
        return float("nan"), True
    try:
        val = hdbscan.validity.validity_index(
            np.ascontiguousarray(Z[idx], dtype=np.float64),
            np.ascontiguousarray(sub_lbl, dtype=np.int64),
            metric="euclidean",
        )
        return (float(val) if np.isfinite(val) else float("nan")), True
    except Exception:
        return float("nan"), True


def find_knee_eps(k_dists_sorted: np.ndarray) -> float:
    """Estimate knee epsilon via Kneedle with geometric perpendicular chord fallback."""
    x = np.arange(len(k_dists_sorted), dtype=np.float64)
    eps_knee = None
    try:
        kl = KneeLocator(x, k_dists_sorted, curve="convex", direction="increasing", S=1.0)
        if kl.knee is not None:
            eps_knee = float(k_dists_sorted[int(kl.knee)])
    except Exception:
        eps_knee = None

    if eps_knee is None or eps_knee <= 0:
        p1 = np.array([x[0], k_dists_sorted[0]])
        p2 = np.array([x[-1], k_dists_sorted[-1]])
        line = p2 - p1
        norm_line = np.linalg.norm(line)
        if norm_line > 0:
            pts = np.column_stack([x, k_dists_sorted])
            cross = np.abs(np.cross(line, p1 - pts)) / norm_line
            eps_knee = float(k_dists_sorted[int(np.argmax(cross))])
        else:
            eps_knee = float(np.median(k_dists_sorted))
    return float(eps_knee)


def evaluate_partition(
    Z: np.ndarray,
    labels: np.ndarray,
    y_true: np.ndarray,
    pres_series: pd.Series,
    compute_dbcv_flag: bool = True,
) -> dict:
    """Evaluate clustering metrics plus per-president noise rates."""
    n = len(labels)
    unique_c = set(labels) - {-1}
    n_clusters = len(unique_c)
    noise_mask = labels == -1
    non_noise = ~noise_mask
    noise_count = int(noise_mask.sum())
    noise_pct = 100.0 * noise_count / n

    if n_clusters > 0:
        counts_c = pd.Series(labels[non_noise]).value_counts()
        largest_cluster_pct = 100.0 * float(counts_c.iloc[0]) / n
    else:
        largest_cluster_pct = 0.0

    if compute_dbcv_flag:
        dbcv, used_sub = compute_dbcv_safe(Z, labels, y_true)
    else:
        dbcv, used_sub = float("nan"), False

    if n_clusters >= 2 and int(non_noise.sum()) >= n_clusters + 2:
        try:
            sil_nn = float(silhouette_score(Z[non_noise], labels[non_noise], metric="euclidean"))
        except Exception:
            sil_nn = float("nan")
    else:
        sil_nn = float("nan")

    ari_all = float(adjusted_rand_score(y_true, labels))
    ari_nn = (
        float(adjusted_rand_score(y_true[non_noise], labels[non_noise]))
        if int(non_noise.sum()) >= 2 and len(set(labels[non_noise])) >= 1
        else float("nan")
    )
    nmi_all = float(normalized_mutual_info_score(y_true, labels))
    homog = float(homogeneity_score(y_true, labels))
    compl = float(completeness_score(y_true, labels))
    v_meas = float(v_measure_score(y_true, labels))

    per_pres_noise = {}
    for p in ALL_PRESIDENTS:
        pm = (pres_series == p).to_numpy()
        per_pres_noise[f"noise_{p}"] = 100.0 * float((labels[pm] == -1).mean())

    return {
        "n_clusters": int(n_clusters),
        "noise_count": noise_count,
        "noise_pct": float(noise_pct),
        "largest_cluster_pct": float(largest_cluster_pct),
        "dbcv": float(dbcv),
        "dbcv_subsampled": bool(used_sub),
        "silhouette_no_noise": float(sil_nn),
        "ari_all": ari_all,
        "ari_no_noise": ari_nn,
        "nmi_all": nmi_all,
        "homogeneity": homog,
        "completeness": compl,
        "v_measure": v_meas,
        **per_pres_noise,
    }


def make_contingency_tables(
    pres_series: pd.Series, labels: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build President x Cluster count and row-percentage tables including Noise (-1)."""
    clusters = sorted([int(c) for c in set(labels) if c != -1])
    cols = clusters + ([-1] if -1 in set(labels) else [])
    col_names = {c: ("Ruído (-1)" if c == -1 else f"C{c}") for c in cols}
    ct_counts = pd.crosstab(pres_series, pd.Series(labels, name="cluster"))
    ct_counts = ct_counts.reindex(index=ALL_PRESIDENTS, columns=cols, fill_value=0)
    ct_counts = ct_counts.rename(columns=col_names)
    ct_pct = ct_counts.div(ct_counts.sum(axis=1), axis=0) * 100.0
    return ct_counts, ct_pct


def generate_fig10(
    df_tradeoff: pd.DataFrame,
    df_viz: pd.DataFrame,
    regimes: dict[str, dict],
) -> None:
    """Plot 2x2 panel illustrating the global-epsilon dilemma on unbalanced densities."""
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.2), constrained_layout=True)

    # 1. Top-left: Per-president Noise % and Largest Cluster % vs epsilon
    ax = axes[0, 0]
    eps_vals = df_tradeoff["eps"].to_numpy()
    highlight_pres = ["Sarney", "Collor", "Itamar", "Temer", "Bolsonaro", "Lula"]
    styles = {
        "Sarney": ("-", 2.0, "^"),
        "Collor": ("-", 2.2, "s"),
        "Itamar": ("-", 2.0, "o"),
        "Temer": ("--", 1.7, ""),
        "Bolsonaro": ("--", 1.7, "d"),
        "Lula": (":", 2.0, ""),
    }
    n_map = df_viz["president"].value_counts().to_dict()
    for p in highlight_pres:
        ls, lw, mk = styles[p]
        ax.plot(
            eps_vals,
            df_tradeoff[f"noise_{p}"],
            linestyle=ls,
            linewidth=lw,
            marker=mk if mk else None,
            markersize=4,
            color=PRES_COLORS[p],
            label=f"{p} (n={n_map[p]})",
        )

    ax.set_xlabel("Raio global ε (MinPts = 30 em PCA 50D → UMAP 10D)")
    ax.set_ylabel("% de discursos como Ruído (-1)")
    ax.set_ylim(-3, 108)

    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    ax_r.plot(
        eps_vals,
        df_tradeoff["largest_cluster_pct"],
        color="#111111",
        linewidth=2.3,
        linestyle="-.",
        label="% no maior cluster (chaining)",
    )
    ax_r.set_ylabel("% do corpus no maior cluster", color="#111111")
    ax_r.set_ylim(-3, 108)

    regime_styles = [
        ("tight", "ε apertado", "#d62728", 104, "right"),
        ("mid", "ε interm.", "#ff7f0e", 90, "left"),
        ("wide", "ε largo (joelho)", "#2ca02c", 104, "left"),
    ]
    for key, rlabel, rcol, ytxt, halign in regime_styles:
        re_val = regimes[key]["eps"]
        ax.axvline(re_val, color=rcol, linestyle="--", linewidth=1.3, alpha=0.85)
        ax.text(
            re_val,
            ytxt,
            f" {rlabel} ({re_val:.2f}) ",
            color=rcol,
            fontsize=7.8,
            fontweight="bold",
            va="top",
            ha=halign,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85, edgecolor="none"),
        )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", frameon=True, framealpha=0.9, fontsize=7.6)
    ax.set_title("1. Dilema do ε Global: Ruído nos Menores vs. Fusão (Chaining)")

    # 2. Top-right: Boxplot of k30_dist by president
    ax2 = axes[0, 1]
    order_by_n = sorted(ALL_PRESIDENTS, key=lambda p: n_map[p], reverse=True)
    sns.boxplot(
        data=df_viz,
        x="k30_dist",
        y="president",
        order=order_by_n,
        palette=PRES_COLORS,
        hue="president",
        legend=False,
        ax=ax2,
        fliersize=2.0,
        linewidth=1.1,
    )
    ax2.set_yticks(range(len(order_by_n)))
    ax2.set_yticklabels([f"{p} (n={n_map[p]})" for p in order_by_n], fontsize=9)
    for key, rlabel, rcol, _, _ in regime_styles:
        ax2.axvline(regimes[key]["eps"], color=rcol, linestyle="--", linewidth=1.3, alpha=0.85)
    ax2.set_xlabel("Distância ao 30º vizinho (k = 29) — menor = nuvem mais densa")
    ax2.set_ylabel("")
    ax2.set_title("2. Heterogeneidade de Densidade Local por Presidente")

    # Helper for 2D scatters
    def scatter_regime(ax_s, labels, title, subtitle):
        uniq = sorted([int(c) for c in set(labels) if c != -1])
        cmap = sns.color_palette("tab10", max(len(uniq), 1)) if len(uniq) <= 10 else sns.color_palette("husl", len(uniq))
        pal = {c: mpl.colors.to_hex(cmap[i]) for i, c in enumerate(uniq)}
        x = df_viz["umap2d_x"].to_numpy()
        y = df_viz["umap2d_y"].to_numpy()
        nm = labels == -1
        if nm.any():
            ax_s.scatter(
                x[nm],
                y[nm],
                c="#9e9e9e",
                marker="x",
                s=14,
                alpha=0.55,
                linewidths=0.8,
                label=f"Ruído (-1) [{int(nm.sum())}]",
                zorder=1,
            )
        for c in uniq:
            m = labels == c
            ax_s.scatter(
                x[m],
                y[m],
                c=pal[c],
                s=11,
                alpha=0.72,
                edgecolors="none",
                label=f"C{c} ({int(m.sum())})",
                zorder=2,
            )
        ax_s.set_title(f"{title}\n{subtitle}", fontsize=10.5)
        ax_s.set_xticks([])
        ax_s.set_yticks([])
        if len(uniq) <= 12:
            ax_s.legend(loc="best", frameon=True, framealpha=0.88, fontsize=7.3, ncol=2 if len(uniq) > 4 else 1)

    r_tight = regimes["tight"]
    scatter_regime(
        axes[1, 0],
        df_viz["label_tight"].to_numpy(),
        f"3. Regime ε Apertado (ε = {r_tight['eps']:.3f})",
        f"k={r_tight['n_clusters']} | Ruído={r_tight['noise_pct']:.1f}% | Sarney={r_tight['noise_Sarney']:.0f}%, Collor={r_tight['noise_Collor']:.0f}% ruído",
    )

    r_wide = regimes["wide"]
    scatter_regime(
        axes[1, 1],
        df_viz["label_wide"].to_numpy(),
        f"4. Regime ε Largo / Joelho (ε = {r_wide['eps']:.3f})",
        f"k={r_wide['n_clusters']} | Ruído={r_wide['noise_pct']:.1f}% | Maior cluster={r_wide['largest_cluster_pct']:.1f}% | ARI={r_wide['ari_all']:.2f}",
    )

    fig.suptitle(
        "Corpus Desbalanceado Completo (n = 5.967, 8 Presidentes): Por Que um Único ε Global Falha",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(FIG_DIR / "fig10_unbalanced_density_dilemma.png", bbox_inches="tight")
    plt.close(fig)


def compact_contingency(cnt_df: pd.DataFrame, top_k: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep top_k largest non-noise clusters + aggregated 'Outros' + 'Ruído (-1)' for legible heatmaps."""
    cluster_cols = [c for c in cnt_df.columns if c != "Ruído (-1)"]
    if len(cluster_cols) <= top_k + 1:
        pct_df = cnt_df.div(cnt_df.sum(axis=1), axis=0) * 100.0
        return cnt_df, pct_df
    totals = cnt_df[cluster_cols].sum(axis=0).sort_values(ascending=False)
    keep = totals.index[:top_k].tolist()
    rest = [c for c in cluster_cols if c not in keep]
    out_cnt = cnt_df[keep].copy()
    out_cnt[f"Outros ({len(rest)})"] = cnt_df[rest].sum(axis=1)
    if "Ruído (-1)" in cnt_df.columns:
        out_cnt["Ruído (-1)"] = cnt_df["Ruído (-1)"]
    out_pct = out_cnt.div(out_cnt.sum(axis=1), axis=0) * 100.0
    return out_cnt, out_pct


def generate_fig11(
    ct_tight_cnt: pd.DataFrame,
    ct_wide_cnt: pd.DataFrame,
    ct_hdb_cnt: pd.DataFrame,
    regimes: dict[str, dict],
    ev_hdb: dict,
    n_map: dict[str, int],
) -> None:
    """Plot 3-panel comparison: Tight DBSCAN vs Wide/Chaining DBSCAN vs HDBSCAN."""
    t_cnt, t_pct = compact_contingency(ct_tight_cnt, top_k=6)
    w_cnt, w_pct = compact_contingency(ct_wide_cnt, top_k=8)
    h_cnt, h_pct = compact_contingency(ct_hdb_cnt, top_k=7)

    y_labels = [f"{p} (n={n_map[p]})" for p in t_pct.index]
    widths = [max(4.5, t_pct.shape[1] * 0.72), max(3.6, w_pct.shape[1] * 0.85), max(4.8, h_pct.shape[1] * 0.72)]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(sum(widths) + 1.8, 6.2),
        gridspec_kw={"width_ratios": widths},
        constrained_layout=True,
    )

    panels = [
        (
            axes[0],
            t_cnt,
            t_pct,
            f"1. DBSCAN ε Apertado (ε={regimes['tight']['eps']:.3f})\n"
            f"k={regimes['tight']['n_clusters']} | Ruído={regimes['tight']['noise_pct']:.1f}% | ARI={regimes['tight']['ari_all']:.2f}",
        ),
        (
            axes[1],
            w_cnt,
            w_pct,
            f"2. DBSCAN ε Largo / Joelho (ε={regimes['wide']['eps']:.3f})\n"
            f"k={regimes['wide']['n_clusters']} | Maior={regimes['wide']['largest_cluster_pct']:.1f}% | ARI={regimes['wide']['ari_all']:.2f}",
        ),
        (
            axes[2],
            h_cnt,
            h_pct,
            f"3. HDBSCAN (Densidade Adaptativa)\n"
            f"k={ev_hdb['n_clusters']} | Ruído={ev_hdb['noise_pct']:.1f}% | ARI={ev_hdb['ari_all']:.2f}",
        ),
    ]

    for idx, (ax, cnt_df, pct_df, title) in enumerate(panels):
        annot = pd.DataFrame(index=pct_df.index, columns=pct_df.columns, dtype=object)
        for r in pct_df.index:
            for c in pct_df.columns:
                annot.loc[r, c] = f"{pct_df.loc[r, c]:.0f}%\n({int(cnt_df.loc[r, c])})"

        sns.heatmap(
            pct_df,
            ax=ax,
            cmap="YlGnBu",
            annot=annot,
            fmt="",
            linewidths=0.5,
            linecolor="white",
            vmin=0,
            vmax=100,
            cbar=(idx == 2),
            cbar_kws={"label": "% dos discursos do presidente"} if idx == 2 else {},
            annot_kws={"size": 7.6},
        )
        if "Ruído (-1)" in pct_df.columns:
            nc = list(pct_df.columns).index("Ruído (-1)")
            ax.axvline(nc, color="#d62728", linewidth=2.0, linestyle="--")
        ax.set_yticklabels(y_labels if idx == 0 else [], rotation=0, fontsize=9)
        ax.set_ylabel("Presidente (volume real no corpus)" if idx == 0 else "")
        ax.set_xlabel("Cluster (e Ruído -1)")
        ax.set_title(title, fontsize=10.5, pad=8)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        "Efeito do Desbalanceamento (27,1×): ε Apertado vs. Fusão no ε Largo vs. HDBSCAN (n = 5.967)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.savefig(FIG_DIR / "fig11_unbalanced_contingency_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    prep = load_preprocess_module()

    df_all = pd.read_parquet(PROC_DIR / "discursos_all_clean.parquet")
    df_all["president"] = pd.Categorical(df_all["president"], categories=ALL_PRESIDENTS, ordered=True)
    df_all = df_all.sort_values(["president", "date", "doc_id"]).reset_index(drop=True)
    df_all["president"] = df_all["president"].astype(str)
    df_all["text_no_leak"] = df_all["text_clean"].map(prep.sanitize_for_ngrams)

    pres_to_id = {p: i for i, p in enumerate(ALL_PRESIDENTS)}
    y_true = df_all["president"].map(pres_to_id).to_numpy()
    n_map = df_all["president"].value_counts().to_dict()
    print(f"Loaded full clean unbalanced corpus: n={len(df_all)} across {len(ALL_PRESIDENTS)} presidents.")
    print("Counts:", {p: n_map[p] for p in ALL_PRESIDENTS})

    # Replicate exact 3-block pure-style feature engineering on all 5,967 speeches
    stylo_rows = [prep.extract_stylo23(t) for t in df_all["text_clean"]]
    df_stylo = pd.DataFrame(stylo_rows)
    X_stylo_z = StandardScaler().fit_transform(df_stylo.to_numpy(dtype=np.float64))
    X_stylo_l2 = normalize(X_stylo_z, norm="l2")

    func_vec = TfidfVectorizer(
        vocabulary=prep.FUNCTION_WORDS,
        lowercase=True,
        token_pattern=r"(?u)\b[a-záàâãéêíóôõúüç]+\b",
        sublinear_tf=True,
        norm="l2",
    )
    X_func = func_vec.fit_transform(df_all["text_no_leak"]).toarray()
    keep_func = (X_func > 0).sum(axis=0) >= 5
    X_func = normalize(X_func[:, keep_func], norm="l2")

    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        max_features=250,
        min_df=0.05,
        max_df=0.95,
        sublinear_tf=True,
        norm="l2",
    )
    X_char = char_vec.fit_transform(df_all["text_no_leak"]).toarray()

    X_all_highdim = normalize(np.hstack([X_stylo_l2, X_func, X_char]), norm="l2")
    print(f"X_all_highdim shape: {X_all_highdim.shape}")

    # Winning DR pipeline: PCA(50D) -> UMAP(10D) + PCA(50D) -> UMAP(2D) for visualization
    pca50 = PCA(n_components=50, random_state=SEED).fit_transform(X_all_highdim)
    Z_all_10d = umap.UMAP(
        n_components=10, n_neighbors=15, min_dist=0.0, metric="euclidean", random_state=SEED
    ).fit_transform(pca50)
    Z_all_2d = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=SEED
    ).fit_transform(pca50)

    # Local density heterogeneity per president
    nn = NearestNeighbors(n_neighbors=30, metric="euclidean").fit(Z_all_10d)
    dists, _ = nn.kneighbors(Z_all_10d)
    k15_dist = dists[:, 14]
    k30_dist = dists[:, 29]

    df_dens_src = pd.DataFrame(
        {"president": df_all["president"], "k15_dist": k15_dist, "k30_dist": k30_dist}
    )
    dens_rows = []
    for p in ALL_PRESIDENTS:
        sub = df_dens_src[df_dens_src["president"] == p]
        dens_rows.append(
            {
                "president": p,
                "n_speeches": int(len(sub)),
                "k15_p25": float(sub["k15_dist"].quantile(0.25)),
                "k15_median": float(sub["k15_dist"].median()),
                "k15_p75": float(sub["k15_dist"].quantile(0.75)),
                "k15_p90": float(sub["k15_dist"].quantile(0.90)),
                "k30_p25": float(sub["k30_dist"].quantile(0.25)),
                "k30_median": float(sub["k30_dist"].median()),
                "k30_p75": float(sub["k30_dist"].quantile(0.75)),
                "k30_p90": float(sub["k30_dist"].quantile(0.90)),
            }
        )
    df_dens = pd.DataFrame(dens_rows)
    df_dens.to_csv(PROC_DIR / "unbalanced_density_by_president.csv", index=False)
    print("\n=== LOCAL DENSITY BY PRESIDENT (k15 and k30 distances) ===")
    print(df_dens.round(4).to_string(index=False))

    # Global epsilon tradeoff curve (MinPts = 30)
    k30_sorted = np.sort(k30_dist)
    eps_knee = find_knee_eps(k30_sorted)
    raw_eps = np.concatenate(
        [
            np.quantile(k30_dist, [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 0.96]),
            [eps_knee, eps_knee * 1.2, eps_knee * 1.5, eps_knee * 1.9],
        ]
    )
    raw_eps = np.sort(np.unique(np.round(raw_eps, 4)))
    # Keep at least ~0.012 spacing while preserving exact rounded eps_knee
    eps_knee_r = round(eps_knee, 4)
    eps_grid: list[float] = []
    for e in raw_eps:
        if not eps_grid or abs(e - eps_grid[-1]) >= 0.012 or abs(e - eps_knee_r) < 1e-4:
            eps_grid.append(float(e))
    # Guarantee eps_knee_r is present
    if all(abs(e - eps_knee_r) > 1e-4 for e in eps_grid):
        eps_grid = sorted(eps_grid + [eps_knee_r])

    print(f"\nRunning DBSCAN tradeoff sweep (MinPts=30) over {len(eps_grid)} eps values (knee={eps_knee_r})...")
    tradeoff_rows = []
    labels_by_eps: dict[float, np.ndarray] = {}
    for eps in eps_grid:
        lbl = DBSCAN(eps=float(eps), min_samples=30, metric="euclidean").fit_predict(Z_all_10d)
        labels_by_eps[float(eps)] = lbl
        ev = evaluate_partition(Z_all_10d, lbl, y_true, df_all["president"], compute_dbcv_flag=True)
        tradeoff_rows.append(
            {
                "eps": float(eps),
                "is_knee": bool(abs(eps - eps_knee_r) < 1e-4),
                **ev,
            }
        )

    df_trade = pd.DataFrame(tradeoff_rows)
    df_trade.to_csv(PROC_DIR / "unbalanced_eps_tradeoff.csv", index=False)
    print("\n=== UNBALANCED EPSILON TRADEOFF CURVE (MinPts=30) ===")
    show_cols = [
        "eps", "is_knee", "n_clusters", "noise_pct", "largest_cluster_pct",
        "dbcv", "ari_all", "homogeneity", "noise_Collor", "noise_Itamar", "noise_Lula",
    ]
    print(df_trade[show_cols].round(3).to_string(index=False))

    # Select 3 strictly ordered representative regimes: tight < mid < wide
    # 1) tight: noise_pct closest to 27% within [15%, 45%]
    cand_t = df_trade[(df_trade["noise_pct"] >= 15.0) & (df_trade["noise_pct"] <= 45.0) & (df_trade["n_clusters"] >= 2)]
    if len(cand_t) == 0:
        cand_t = df_trade[df_trade["n_clusters"] >= 2]
    row_tight = cand_t.iloc[(cand_t["noise_pct"] - 27.0).abs().argmin()].to_dict()

    # 2) mid: best DBCV within intermediate noise band (2% <= noise <= 18%) above tight eps
    cand_m = df_trade[
        (df_trade["eps"] > float(row_tight["eps"]))
        & (df_trade["noise_pct"] >= 2.0)
        & (df_trade["noise_pct"] <= 18.0)
        & (df_trade["n_clusters"] >= 2)
    ]
    if len(cand_m) > 0:
        row_mid = cand_m.sort_values("dbcv", ascending=False).iloc[0].to_dict()
    else:
        cand_m2 = df_trade[(df_trade["eps"] > float(row_tight["eps"])) & (df_trade["n_clusters"] >= 2)]
        row_mid = cand_m2.iloc[len(cand_m2) // 2].to_dict()

    # 3) wide: knee or first eps > mid where noise_pct <= 0.3% (or largest_cluster_pct >= 50%, capturing chaining)
    cand_w = df_trade[
        (df_trade["eps"] > float(row_mid["eps"]))
        & (df_trade["n_clusters"] >= 2)
        & ((df_trade["noise_pct"] <= 0.3) | (df_trade["largest_cluster_pct"] >= 50.0))
    ]
    if len(cand_w) > 0:
        row_wide = cand_w.sort_values("eps").iloc[0].to_dict()
    else:
        cand_w2 = df_trade[(df_trade["eps"] > float(row_mid["eps"])) & (df_trade["n_clusters"] >= 2)]
        row_wide = cand_w2.sort_values(["noise_pct", "eps"]).iloc[0].to_dict()

    regimes = {"tight": row_tight, "mid": row_mid, "wide": row_wide}
    lbl_tight = labels_by_eps[float(row_tight["eps"])]
    lbl_mid = labels_by_eps[float(row_mid["eps"])]
    lbl_wide = labels_by_eps[float(row_wide["eps"])]

    # HDBSCAN comparison on the same unbalanced Z_all_10d
    hdb = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=15, metric="euclidean")
    lbl_hdb = hdb.fit_predict(Z_all_10d)
    ev_hdb = evaluate_partition(Z_all_10d, lbl_hdb, y_true, df_all["president"], compute_dbcv_flag=True)

    # Contingency tables for all 4 unbalanced partitions
    ct_tight_cnt, ct_tight_pct = make_contingency_tables(df_all["president"], lbl_tight)
    ct_mid_cnt, ct_mid_pct = make_contingency_tables(df_all["president"], lbl_mid)
    ct_wide_cnt, ct_wide_pct = make_contingency_tables(df_all["president"], lbl_wide)
    ct_hdb_cnt, ct_hdb_pct = make_contingency_tables(df_all["president"], lbl_hdb)

    ct_tight_cnt.to_csv(PROC_DIR / "unbalanced_contingency_tight_counts.csv")
    ct_tight_pct.round(2).to_csv(PROC_DIR / "unbalanced_contingency_tight_pct.csv")
    ct_mid_cnt.to_csv(PROC_DIR / "unbalanced_contingency_mid_counts.csv")
    ct_mid_pct.round(2).to_csv(PROC_DIR / "unbalanced_contingency_mid_pct.csv")
    ct_wide_cnt.to_csv(PROC_DIR / "unbalanced_contingency_wide_counts.csv")
    ct_wide_pct.round(2).to_csv(PROC_DIR / "unbalanced_contingency_wide_pct.csv")
    ct_hdb_cnt.to_csv(PROC_DIR / "unbalanced_contingency_hdbscan_counts.csv")
    ct_hdb_pct.round(2).to_csv(PROC_DIR / "unbalanced_contingency_hdbscan_pct.csv")

    print("\n=== CONTINGENCY: TIGHT EPSILON (%) ===")
    print(ct_tight_pct.round(1).to_string())
    print("\n=== CONTINGENCY: MID EPSILON (%) ===")
    print(ct_mid_pct.round(1).to_string())
    print("\n=== CONTINGENCY: WIDE EPSILON (%) ===")
    print(ct_wide_pct.round(1).to_string())
    print("\n=== CONTINGENCY: HDBSCAN (%) ===")
    print(ct_hdb_pct.round(1).to_string())

    # Comparison summary against main balanced DBSCAN
    df_main = pd.read_csv(PROC_DIR / "results_summary.csv")
    bal_row = df_main[
        (df_main["scenario"] == "08_pca50_umap10") & (df_main["selection_mode"] == "best_dbcv")
    ].iloc[0]

    # Compute largest_cluster_pct for balanced benchmark
    df_viz_bal = pd.read_parquet(PROC_DIR / "viz_points_2d.parquet")
    bal_lbl = df_viz_bal["best_cluster"].to_numpy()
    bal_largest_pct = 100.0 * float(pd.Series(bal_lbl[bal_lbl != -1]).value_counts().iloc[0]) / len(bal_lbl)

    comp_rows = [
        {
            "regime": "balanced_main_dbscan",
            "description": "DBSCAN Balanceado (n=1.800, 6 pres., MinPts=30)",
            "n_speeches": 1800,
            "n_presidents": 6,
            "eps": float(bal_row["eps"]),
            "n_clusters": int(bal_row["n_clusters"]),
            "noise_pct": float(bal_row["noise_pct"]),
            "largest_cluster_pct": bal_largest_pct,
            "dbcv": float(bal_row["dbcv"]),
            "silhouette_no_noise": float(bal_row["silhouette_no_noise"]),
            "ari_all": float(bal_row["ari_all"]),
            "homogeneity": float(bal_row["homogeneity"]),
            "completeness": float(bal_row["completeness"]),
        },
        {
            "regime": "unbalanced_tight",
            "description": "DBSCAN Desbalanceado — ε Apertado (MinPts=30)",
            "n_speeches": len(df_all),
            "n_presidents": len(ALL_PRESIDENTS),
            **{k: row_tight[k] for k in ["eps", "n_clusters", "noise_pct", "largest_cluster_pct", "dbcv", "silhouette_no_noise", "ari_all", "homogeneity", "completeness"]},
        },
        {
            "regime": "unbalanced_mid",
            "description": "DBSCAN Desbalanceado — ε Intermediário (MinPts=30)",
            "n_speeches": len(df_all),
            "n_presidents": len(ALL_PRESIDENTS),
            **{k: row_mid[k] for k in ["eps", "n_clusters", "noise_pct", "largest_cluster_pct", "dbcv", "silhouette_no_noise", "ari_all", "homogeneity", "completeness"]},
        },
        {
            "regime": "unbalanced_wide",
            "description": "DBSCAN Desbalanceado — ε Largo (MinPts=30)",
            "n_speeches": len(df_all),
            "n_presidents": len(ALL_PRESIDENTS),
            **{k: row_wide[k] for k in ["eps", "n_clusters", "noise_pct", "largest_cluster_pct", "dbcv", "silhouette_no_noise", "ari_all", "homogeneity", "completeness"]},
        },
        {
            "regime": "unbalanced_hdbscan",
            "description": "HDBSCAN Desbalanceado (min_cluster_size=30, min_samples=15)",
            "n_speeches": len(df_all),
            "n_presidents": len(ALL_PRESIDENTS),
            "eps": float("nan"),
            **{k: ev_hdb[k] for k in ["n_clusters", "noise_pct", "largest_cluster_pct", "dbcv", "silhouette_no_noise", "ari_all", "homogeneity", "completeness"]},
        },
    ]
    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(PROC_DIR / "unbalanced_comparison_summary.csv", index=False)
    print("\n=== COMPARISON SUMMARY ===")
    print(df_comp.round(4).to_string(index=False))

    df_viz_unb = pd.DataFrame(
        {
            "doc_id": df_all["doc_id"],
            "president": df_all["president"],
            "date": df_all["date"],
            "word_count": df_all["word_count"],
            "umap2d_x": Z_all_2d[:, 0],
            "umap2d_y": Z_all_2d[:, 1],
            "k15_dist": k15_dist,
            "k30_dist": k30_dist,
            "label_tight": lbl_tight,
            "label_mid": lbl_mid,
            "label_wide": lbl_wide,
            "label_hdbscan": lbl_hdb,
        }
    )
    df_viz_unb.to_parquet(PROC_DIR / "unbalanced_viz_2d.parquet", index=False)

    generate_fig10(df_trade, df_viz_unb, regimes)
    generate_fig11(ct_tight_cnt, ct_wide_cnt, ct_hdb_cnt, regimes, ev_hdb, n_map)
    print("\nGenerated fig10_unbalanced_density_dilemma.png and fig11_unbalanced_contingency_heatmap.png")


if __name__ == "__main__":
    main()
