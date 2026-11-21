#!/usr/bin/env python3
"""Generate publication-quality Portuguese figures for the EST011 DBSCAN seminar."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

import json
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
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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

PRES_COLORS = {
    "Sarney": "#8c564b",
    "FHC": "#1f77b4",
    "Lula": "#d62728",
    "Dilma": "#ff7f0e",
    "Temer": "#2ca02c",
    "Bolsonaro": "#9467bd",
}

STYLO_LABELS = {
    "avg_sent_len": "Tam. médio sentença",
    "std_sent_len": "Desvio tam. sentença",
    "short_sent_ratio": "Prop. sentenças curtas (≤8)",
    "long_sent_ratio": "Prop. sentenças longas (≥35)",
    "avg_word_len": "Tam. médio palavra",
    "long_word_ratio": "Prop. palavras longas (≥7)",
    "mtld": "Diversidade lexical (MTLD)",
    "mattr_100": "Diversidade (MATTR-100)",
    "flesch_pt": "Legibilidade (Flesch-PT)",
    "first_person_sg_rate": "1ª pessoa singular (/1k)",
    "first_person_pl_rate": "1ª pessoa plural (/1k)",
    "first_person_sg_share": "Razão 1ª sg / (sg+pl)",
    "vocative_rate": "Vocativos (/1k)",
    "exclamation_rate": "Exclamações ! (/1k)",
    "question_rate": "Interrogações ? (/1k)",
    "semicolon_rate": "Ponto e vírgula ; (/1k)",
    "colon_rate": "Dois-pontos : (/1k)",
    "comma_per_sent": "Vírgulas / sentença",
    "orality_rate": "Marcadores oralidade (/1k)",
    "formal_connector_rate": "Conectivos cultos (/1k)",
    "gerund_rate": "Gerúndios -ndo (/1k)",
    "adverb_mente_rate": "Advérbios -mente (/1k)",
    "nominalization_rate": "Nominalizações (/1k)",
}

SCENARIO_LABELS = {
    "01_orig_euclid": "Original 528D (Euclid.)",
    "02_orig_cosine": "Original 528D (Cosseno)",
    "03_pca_2d": "PCA 2D",
    "04_pca_10d": "PCA 10D",
    "05_svd_10d": "Truncated SVD 10D",
    "06_umap_10d_md0": "UMAP 10D (md=0.0)",
    "07_umap_10d_md05": "UMAP 10D (md=0.5)",
    "08_pca50_umap10": "PCA 50D → UMAP 10D",
    "09_tsne_perp10": "t-SNE 2D (perp=10)",
    "10_tsne_perp50": "t-SNE 2D (perp=50)",
    "11_stylo23_pca10": "Só Métricas 23D → PCA 10D",
    "12_tfidf3000_svd50_umap10": "TF-IDF 3000D → SVD50 → UMAP10",
    "13_kmeans_on_best": "K-Means (K=6) no melhor espaço",
}


def get_cluster_palette(labels: np.ndarray) -> dict[int, str]:
    """Return consistent color mapping with Noise (-1) always grey."""
    uniq = sorted([int(c) for c in set(labels) if c != -1])
    cmap = sns.color_palette("tab10", max(len(uniq), 1))
    if len(uniq) > 10:
        cmap = sns.color_palette("husl", len(uniq))
    pal = {c: mpl.colors.to_hex(cmap[i]) for i, c in enumerate(uniq)}
    pal[-1] = "#9e9e9e"
    return pal


def plot_scatter_clusters(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    title: str,
    subtitle: str = "",
    point_types: np.ndarray | None = None,
    show_legend: bool = True,
) -> None:
    """Scatter 2D embedding colored by cluster labels, highlighting Noise (-1) as grey crosses."""
    pal = get_cluster_palette(labels)
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(
            x[noise_mask],
            y[noise_mask],
            c=pal[-1],
            marker="x",
            s=18,
            alpha=0.55,
            linewidths=0.9,
            label=f"Ruído (-1) [{int(noise_mask.sum())}]",
            zorder=1,
        )

    clusters = sorted([int(c) for c in set(labels) if c != -1])
    for c in clusters:
        mask = labels == c
        if point_types is not None:
            core_m = mask & (point_types == "core")
            bord_m = mask & (point_types == "border")
            if core_m.any():
                ax.scatter(
                    x[core_m],
                    y[core_m],
                    c=pal[c],
                    s=18,
                    alpha=0.82,
                    edgecolors="none",
                    label=f"C{c} ({int(mask.sum())})",
                    zorder=3,
                )
            if bord_m.any():
                ax.scatter(
                    x[bord_m],
                    y[bord_m],
                    facecolors="none",
                    edgecolors=pal[c],
                    s=20,
                    alpha=0.75,
                    linewidths=0.8,
                    zorder=2,
                )
        else:
            ax.scatter(
                x[mask],
                y[mask],
                c=pal[c],
                s=16,
                alpha=0.78,
                edgecolors="none",
                label=f"C{c} ({int(mask.sum())})",
                zorder=2,
            )

    full_title = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full_title, fontsize=10.5, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_legend and len(clusters) <= 12:
        ax.legend(
            loc="best",
            frameon=True,
            framealpha=0.85,
            fontsize=7.5,
            ncol=2 if len(clusters) > 5 else 1,
            markerscale=1.1,
        )


def fig01_corpus_overview(meta: dict, df_bal: pd.DataFrame) -> None:
    raw_counts = meta["raw_counts_all_clean"]
    order_all = ["Sarney", "Collor", "Itamar", "FHC", "Lula", "Dilma", "Temer", "Bolsonaro"]
    counts = [raw_counts.get(p, 0) for p in order_all]
    sampled = [meta["n_per_president"] if p in meta["presidents"] else 0 for p in order_all]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    y_pos = np.arange(len(order_all))
    bars_raw = ax1.barh(
        y_pos - 0.18, counts, height=0.35, color="#b0bec5", label="Corpus limpo disponível"
    )
    bars_bal = ax1.barh(
        y_pos + 0.18,
        sampled,
        height=0.35,
        color=[PRES_COLORS.get(p, "#607d8b") for p in order_all],
        label=f"Amostra balanceada (n={meta['n_per_president']}/pres.)",
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(order_all)
    ax1.invert_yaxis()
    ax1.set_xlabel("Número de discursos (≥ 120 palavras, PT-BR)")
    ax1.set_title("Acervo Limpo vs. Amostra Balanceada para DBSCAN")
    for b, val in zip(bars_raw, counts):
        ax1.text(val + 25, b.get_y() + b.get_height() / 2, f"{val}", va="center", fontsize=8.5)
    for b, val in zip(bars_bal, sampled):
        if val > 0:
            ax1.text(val + 25, b.get_y() + b.get_height() / 2, f"{val}", va="center", fontsize=8.5, fontweight="bold")
    ax1.set_xlim(0, max(counts) * 1.16)
    ax1.legend(loc="lower right", frameon=True)

    pres_order = meta["presidents"]
    sns.boxplot(
        data=df_bal,
        x="word_count",
        y="president",
        order=pres_order,
        palette=PRES_COLORS,
        hue="president",
        legend=False,
        ax=ax2,
        fliersize=2.5,
        linewidth=1.1,
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("Comprimento do discurso em palavras (escala log)")
    ax2.set_ylabel("")
    ax2.set_title("Distribuição de Extensão por Presidente (n = 1.800)")
    medians = df_bal.groupby("president")["word_count"].median()
    for i, p in enumerate(pres_order):
        m = medians[p]
        ax2.text(m, i - 0.43, f"med={int(m)}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.set_ylim(len(pres_order) - 0.35, -0.75)

    fig.suptitle(
        "Corpus de Discursos Presidenciais Brasileiros (1985–2022) — Controle de Desbalanceamento",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.savefig(FIG_DIR / "fig01_corpus_overview.png", bbox_inches="tight")
    plt.close(fig)


def fig02_curse_of_dimensionality(plot_npz: dict, df_summary: pd.DataFrame, best_scen_id: str) -> None:
    best_sum = df_summary[df_summary["selection_mode"] == "best_dbcv"].set_index("scenario")
    specs = [
        ("01_orig_euclid", "orig_euclid", "Espaço Original (528D, Euclidiana)", "#c0392b"),
        ("04_pca_10d", "pca10", "PCA Linear (10D, Euclidiana)", "#2980b9"),
        (best_scen_id, "best", f"{SCENARIO_LABELS.get(best_scen_id, best_scen_id)} (Vencedor)", "#27ae60"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.6), constrained_layout=True)
    for col_idx, (scen_id, prefix, title, color) in enumerate(specs):
        dists = plot_npz[f"{prefix}_dists"]
        kdist = plot_npz[f"{prefix}_kdist"]
        knee = float(plot_npz[f"{prefix}_knee"][0])
        row_stat = best_sum.loc[scen_id]

        # Top row: pairwise distance distribution
        ax_top = axes[0, col_idx]
        ax_top.hist(dists, bins=45, density=True, color=color, alpha=0.72, edgecolor="white", linewidth=0.3)
        p05, p50, p95 = row_stat["d_p05"], row_stat["d_p50"], row_stat["d_p95"]
        ax_top.axvline(p05, color="#333333", linestyle="--", linewidth=1.0)
        ax_top.axvline(p50, color="#000000", linestyle="-", linewidth=1.3)
        ax_top.axvline(p95, color="#333333", linestyle="--", linewidth=1.0)
        ax_top.set_title(title)
        ax_top.set_xlabel("Distância par-a-par")
        ax_top.set_ylabel("Densidade" if col_idx == 0 else "")
        ax_top.set_ylim(0, ax_top.get_ylim()[1] * 1.34)
        ax_top.text(
            0.97,
            0.94,
            f"CV = {row_stat['cv_dist']:.3f}\nContraste = {row_stat['rel_contrast']:.3f}\n"
            f"P05={p05:.2f} | Med={p50:.2f} | P95={p95:.2f}",
            transform=ax_top.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="#cccccc"),
        )

        # Bottom row: k-distance plot with knee
        ax_bot = axes[1, col_idx]
        x = np.arange(len(kdist))
        ax_bot.plot(x, kdist, color=color, linewidth=1.8)
        ax_bot.axhline(knee, color="#e67e22", linestyle="--", linewidth=1.3, label=f"Joelho ε = {knee:.3f}")
        ax_bot.set_xlabel("Discursos ordenados (k = 14, MinPts = 15)")
        ax_bot.set_ylabel("Distância ao 14º vizinho" if col_idx == 0 else "")
        ax_bot.legend(loc="upper left", frameon=True, fontsize=8.5)
        ax_bot.text(
            0.97,
            0.12,
            f"DBSCAN (auto joelho):\nk = {int(df_summary[(df_summary['scenario']==scen_id)&(df_summary['selection_mode']=='auto_knee')]['n_clusters'].iloc[0])}, "
            f"ruído = {df_summary[(df_summary['scenario']==scen_id)&(df_summary['selection_mode']=='auto_knee')]['noise_pct'].iloc[0]:.1f}%",
            transform=ax_bot.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="#cccccc"),
        )

    fig.suptitle(
        "Maldição da Dimensionalidade no DBSCAN: Concentração de Distâncias e Curva k-distance",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(FIG_DIR / "fig02_curse_of_dimensionality.png", bbox_inches="tight")
    plt.close(fig)


def fig03_dr_comparison_metrics(df_summary: pd.DataFrame) -> None:
    df_sub = df_summary[df_summary["selection_mode"].isin(["best_dbcv", "fixed_k6"])].copy()
    order = [s for s in SCENARIO_LABELS if s in set(df_sub["scenario"])]
    df_sub["label"] = df_sub["scenario"].map(SCENARIO_LABELS)
    df_sub["order_idx"] = df_sub["scenario"].map({s: i for i, s in enumerate(order)})
    df_sub = df_sub.sort_values("order_idx")

    y = np.arange(len(df_sub))
    labels = df_sub["label"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 6.8), sharey=True, constrained_layout=True)

    # Panel 1: DBCV & Silhouette
    ax = axes[0]
    h = 0.36
    dbcv_vals = df_sub["dbcv"].fillna(0.0).to_numpy()
    sil_vals = df_sub["silhouette_no_noise"].fillna(0.0).to_numpy()
    ax.barh(y - h / 2, dbcv_vals, height=h, color="#1f77b4", label="DBCV (densidade)")
    ax.barh(y + h / 2, sil_vals, height=h, color="#6baed6", label="Silhouette (sem ruído)")
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-0.35, 1.12)
    ax.set_xlabel("Escore (-1 a +1)")
    ax.set_title("Validação Interna", pad=26)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True, fontsize=8)
    for i, (d, s, k) in enumerate(zip(df_sub["dbcv"], df_sub["silhouette_no_noise"], df_sub["n_clusters"])):
        txt = f"k={int(k)}" if pd.isna(d) else f"{d:.2f} (k={int(k)})"
        xpos = max(0.02, (d if pd.notna(d) else 0.0) + 0.02)
        ax.text(xpos, i - h / 2, txt, va="center", fontsize=7.5)

    # Panel 2: External alignment (ARI & Homogeneity)
    ax = axes[1]
    ax.barh(y - h / 2, df_sub["ari_all"], height=h, color="#2ca02c", label="ARI (todos pontos)")
    ax.barh(y + h / 2, df_sub["homogeneity"], height=h, color="#98df8a", label="Homogeneidade")
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Alinhamento com Presidente (0 a 1)")
    ax.set_title("Validação Externa (Gabarito Oculto)", pad=26)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True, fontsize=8)
    for i, (a, ho) in enumerate(zip(df_sub["ari_all"], df_sub["homogeneity"])):
        ax.text(max(a, 0.0) + 0.015, i - h / 2, f"{a:.2f}", va="center", fontsize=7.5)

    # Panel 3: Noise %
    ax = axes[2]
    colors_n = ["#d62728" if v > 40 else ("#ff7f0e" if v > 15 else "#7f7f7f") for v in df_sub["noise_pct"]]
    ax.barh(y, df_sub["noise_pct"], height=0.58, color=colors_n)
    ax.set_xlim(0, 108)
    ax.set_xlabel("% de discursos classificados como Ruído (-1)")
    ax.set_title("Taxa de Ruído", pad=26)
    for i, v in enumerate(df_sub["noise_pct"]):
        ax.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=8)

    fig.suptitle(
        "Impacto da Redução de Dimensionalidade no DBSCAN (Seleção Não-Supervisionada por DBCV)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.savefig(FIG_DIR / "fig03_dr_comparison_metrics.png", bbox_inches="tight")
    plt.close(fig)


def fig04_umap_vs_tsne_pitfall(df_viz: pd.DataFrame, df_summary: pd.DataFrame) -> None:
    best_sum = df_summary[df_summary["selection_mode"] == "best_dbcv"].set_index("scenario")
    panels = [
        ("06_umap_10d_md0", "umap2d_md0_x", "umap2d_md0_y", "label_umap_md0", "UMAP (min_dist = 0.0)"),
        ("07_umap_10d_md05", "umap2d_md05_x", "umap2d_md05_y", "label_umap_md05", "UMAP (min_dist = 0.5)"),
        ("09_tsne_perp10", "tsne_p10_x", "tsne_p10_y", "label_tsne_p10", "t-SNE 2D (perplexity = 10) [Alerta]"),
        ("10_tsne_perp50", "tsne_p50_x", "tsne_p50_y", "label_tsne_p50", "t-SNE 2D (perplexity = 50) [Alerta]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.5), constrained_layout=True)
    for ax, (scen, xc, yc, lc, title) in zip(axes.ravel(), panels):
        row = best_sum.loc[scen]
        sub = (
            f"k={int(row['n_clusters'])} | Ruído={row['noise_pct']:.1f}% | "
            f"DBCV={row['dbcv']:.2f} | ARI={row['ari_all']:.2f}"
        )
        plot_scatter_clusters(
            ax,
            df_viz[xc].to_numpy(),
            df_viz[yc].to_numpy(),
            df_viz[lc].to_numpy(),
            title=title,
            subtitle=sub,
            show_legend=True,
        )

    fig.suptitle(
        "Efeito dos Hiperparâmetros de Manifold na Densidade: UMAP (min_dist) vs. t-SNE (perplexity)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.savefig(FIG_DIR / "fig04_umap_vs_tsne_pitfall.png", bbox_inches="tight")
    plt.close(fig)


def fig05_sensitivity_heatmap(df_grid: pd.DataFrame, df_summary: pd.DataFrame, best_scen_id: str) -> None:
    best_row = df_summary[
        (df_summary["scenario"] == best_scen_id) & (df_summary["selection_mode"] == "best_dbcv")
    ].iloc[0]
    best_eps_val = float(best_row["eps"])
    best_mp = int(best_row["min_pts"])

    # Compute a clean rectangular (MinPts x eps) grid on the winning space so heatmaps have zero NaNs
    X_high = np.load(PROC_DIR / "X_style_highdim.npy")
    if best_scen_id == "08_pca50_umap10":
        pca50 = PCA(n_components=50, random_state=42).fit_transform(X_high)
        Z = umap.UMAP(
            n_components=10, n_neighbors=15, min_dist=0.0, metric="euclidean", random_state=42
        ).fit_transform(pca50)
    else:
        Z = umap.UMAP(
            n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42
        ).fit_transform(X_high)

    min_pts_list = [10, 15, 20, 30]
    eps_list = sorted({round(v, 3) for v in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.533, best_eps_val, 0.65, 0.75, 0.90]})
    rect_rows = []
    Z64 = np.ascontiguousarray(Z, dtype=np.float64)
    for mp in min_pts_list:
        for eps in eps_list:
            lbl = DBSCAN(eps=float(eps), min_samples=mp, metric="euclidean").fit_predict(Z)
            k = len(set(lbl) - {-1})
            noise_pct = 100.0 * float((lbl == -1).sum()) / len(lbl)
            if k >= 2 and int((lbl != -1).sum()) >= 4:
                try:
                    dbcv_val = float(hdbscan.validity.validity_index(Z64, lbl.astype(np.int64), metric="euclidean"))
                except Exception:
                    dbcv_val = np.nan
            else:
                dbcv_val = np.nan
            rect_rows.append({"min_pts": mp, "eps_r": eps, "n_clusters": k, "noise_pct": noise_pct, "dbcv": dbcv_val})
    sub = pd.DataFrame(rect_rows)

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.7), constrained_layout=True)
    metrics = [
        ("n_clusters", "Nº de Clusters (k)", "viridis", ".0f"),
        ("noise_pct", "% de Ruído (-1)", "magma_r", ".1f"),
        ("dbcv", "DBCV (Qualidade por Densidade)", "RdYlGn", ".2f"),
    ]
    best_eps_r = round(best_eps_val, 3)

    for ax, (col, title, cmap, fmt) in zip(axes, metrics):
        piv = sub.pivot(index="min_pts", columns="eps_r", values=col)
        sns.heatmap(
            piv,
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt=fmt,
            linewidths=0.4,
            linecolor="white",
            annot_kws={"size": 8},
            cbar_kws={"shrink": 0.85},
        )
        ax.set_title(title)
        ax.set_xlabel("Raio ε")
        ax.set_ylabel("MinPts" if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=40)

        if best_mp in piv.index and best_eps_r in piv.columns:
            ri = list(piv.index).index(best_mp)
            ci = list(piv.columns).index(best_eps_r)
            ax.add_patch(
                mpl.patches.Rectangle((ci, ri), 1, 1, fill=False, edgecolor="#000000", linewidth=2.4)
            )

    fig.suptitle(
        f"Sensibilidade (ε × MinPts) no Espaço Vencedor: {SCENARIO_LABELS.get(best_scen_id, best_scen_id)}",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.savefig(FIG_DIR / "fig05_sensitivity_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def fig06_clusters_vs_presidents_2d(df_viz: pd.DataFrame, df_summary: pd.DataFrame, best_scen_id: str) -> None:
    best_row = df_summary[
        (df_summary["scenario"] == best_scen_id) & (df_summary["selection_mode"] == "best_dbcv")
    ].iloc[0]
    x = df_viz["umap2d_x"].to_numpy()
    y = df_viz["umap2d_y"].to_numpy()
    labels = df_viz["best_cluster"].to_numpy()
    ptypes = df_viz["point_type"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.8), constrained_layout=True)

    sub_left = (
        f"k={int(best_row['n_clusters'])} clusters | Ruído={best_row['noise_pct']:.1f}% | "
        f"DBCV={best_row['dbcv']:.2f} | Cheios=core, Contorno=border"
    )
    plot_scatter_clusters(
        ax1, x, y, labels, title="1. O que o DBSCAN Enxerga (Sem Rótulos)", subtitle=sub_left, point_types=ptypes
    )

    for pres, color in PRES_COLORS.items():
        m = df_viz["president"].to_numpy() == pres
        if not m.any():
            continue
        ax2.scatter(x[m], y[m], c=color, s=16, alpha=0.68, label=pres, edgecolors="none")
        mx, my = float(np.median(x[m])), float(np.median(y[m]))
        ax2.text(
            mx,
            my,
            pres,
            fontsize=9.5,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.86, edgecolor=color, linewidth=1.3),
        )
    ax2.set_title(
        f"2. Revelação: Quem Falou Cada Discurso?\nARI={best_row['ari_all']:.2f} | Homogeneidade={best_row['homogeneity']:.2f}",
        fontsize=10.5,
        pad=6,
    )
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.legend(loc="best", frameon=True, framealpha=0.88, fontsize=8.5)

    fig.suptitle(
        f"Comparação Espacial 2D ({SCENARIO_LABELS.get(best_scen_id, best_scen_id)})",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.savefig(FIG_DIR / "fig06_clusters_vs_presidents_2d.png", bbox_inches="tight")
    plt.close(fig)


def fig07_heatmap_president_cluster(ct_counts: pd.DataFrame, ct_pct: pd.DataFrame) -> None:
    annot = pd.DataFrame(index=ct_pct.index, columns=ct_pct.columns, dtype=object)
    for r in ct_pct.index:
        for c in ct_pct.columns:
            annot.loc[r, c] = f"{ct_pct.loc[r, c]:.1f}%\n({int(ct_counts.loc[r, c])})"

    fig, ax = plt.subplots(figsize=(max(8.5, 1.25 * ct_pct.shape[1] + 2.5), 5.0), constrained_layout=True)
    sns.heatmap(
        ct_pct,
        ax=ax,
        cmap="YlGnBu",
        annot=annot,
        fmt="",
        linewidths=0.6,
        linecolor="white",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "% dos discursos do presidente"},
        annot_kws={"size": 8.5},
    )
    if "Ruído (-1)" in ct_pct.columns:
        noise_col = list(ct_pct.columns).index("Ruído (-1)")
        ax.axvline(noise_col, color="#d62728", linewidth=2.2, linestyle="--")

    ax.set_title(
        "Distribuição Estilística por Presidente (% por Linha e Contagem Absoluta)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Cluster descoberto pelo DBSCAN (e Ruído -1)")
    ax.set_ylabel("Presidente (300 discursos cada)")
    fig.savefig(FIG_DIR / "fig07_heatmap_president_cluster.png", bbox_inches="tight")
    plt.close(fig)


def fig08_cluster_stylometric_profiles(df_profiles: pd.DataFrame, df_loadings: pd.DataFrame) -> None:
    stylo_cols = [c for c in df_loadings.index if c in df_profiles.columns]
    # Keep top 16 most discriminative metrics across clusters (highest std across cluster means)
    prof_mat = df_profiles.set_index("cluster_label")[stylo_cols].T
    top_feats = prof_mat.std(axis=1).sort_values(ascending=False).head(16).index.tolist()
    prof_top = prof_mat.loc[top_feats].copy()
    prof_top.index = [STYLO_LABELS.get(f, f) for f in prof_top.index]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14.8, 6.8), gridspec_kw={"width_ratios": [1.35, 1.0]}, constrained_layout=True
    )

    vmax = float(np.nanpercentile(np.abs(prof_top.to_numpy()), 95))
    vmax = max(vmax, 1.0)
    sns.heatmap(
        prof_top,
        ax=ax1,
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        annot_kws={"size": 8},
        cbar_kws={"label": "Z-score médio (core points)"},
    )
    ax1.set_title("Perfil Estilométrico dos Clusters (Top 16 Métricas)")
    ax1.set_xlabel("Cluster / Ruído")
    ax1.set_ylabel("")
    ax1.tick_params(axis="x", rotation=25)

    # Right panel: PCA loadings (PC1 & PC2) on Bloco A
    ev1 = float(df_loadings["explained_var_PC1"].iloc[0]) * 100
    ev2 = float(df_loadings["explained_var_PC2"].iloc[0]) * 100
    top_load_idx = (
        df_loadings[["PC1", "PC2"]].abs().max(axis=1).sort_values(ascending=False).head(14).index
    )
    sub_l = df_loadings.loc[top_load_idx, ["PC1", "PC2"]].sort_values("PC1")
    y_l = np.arange(len(sub_l))
    h = 0.36
    ax2.barh(y_l - h / 2, sub_l["PC1"], height=h, color="#1f77b4", label=f"PC1 ({ev1:.1f}% var.)")
    ax2.barh(y_l + h / 2, sub_l["PC2"], height=h, color="#ff7f0e", label=f"PC2 ({ev2:.1f}% var.)")
    ax2.axvline(0, color="#444444", linewidth=0.9)
    ax2.set_yticks(y_l)
    ax2.set_yticklabels([STYLO_LABELS.get(f, f) for f in sub_l.index], fontsize=8.5)
    ax2.set_xlabel("Peso (Loading) nas Componentes Principais")
    ax2.set_title("Eixos Latentes do Estilo (PCA no Bloco A)")
    ax2.legend(loc="lower right", frameon=True)

    fig.suptitle(
        "Interpretação Substantiva: O Que Define Cada Cluster Estilístico?",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.savefig(FIG_DIR / "fig08_cluster_stylometric_profiles.png", bbox_inches="tight")
    plt.close(fig)


def fig09_dbscan_vs_kmeans(df_viz: pd.DataFrame, df_summary: pd.DataFrame, best_scen_id: str) -> None:
    best_row = df_summary[
        (df_summary["scenario"] == best_scen_id) & (df_summary["selection_mode"] == "best_dbcv")
    ].iloc[0]
    km_row = df_summary[df_summary["scenario"] == "13_kmeans_on_best"].iloc[0]

    x = df_viz["umap2d_x"].to_numpy()
    y = df_viz["umap2d_y"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6), constrained_layout=True)
    plot_scatter_clusters(
        ax1,
        x,
        y,
        df_viz["best_cluster"].to_numpy(),
        title="DBSCAN (Baseado em Densidade)",
        subtitle=(
            f"k={int(best_row['n_clusters'])} | Ruído={best_row['noise_pct']:.1f}% | "
            f"DBCV={best_row['dbcv']:.2f} | ARI={best_row['ari_all']:.2f}"
        ),
        point_types=df_viz["point_type"].to_numpy(),
    )
    plot_scatter_clusters(
        ax2,
        x,
        y,
        df_viz["kmeans_cluster"].to_numpy(),
        title="K-Means (K = 6 fixo, Centróides)",
        subtitle=(
            f"k=6 | Ruído=0.0% (força outliers) | "
            f"DBCV={km_row['dbcv']:.2f} | ARI={km_row['ari_all']:.2f}"
        ),
    )

    fig.suptitle(
        "DBSCAN vs. K-Means no Mesmo Espaço Reduzido: Detecção de Ruído e Formas Livres",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.savefig(FIG_DIR / "fig09_dbscan_vs_kmeans.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    meta = json.loads((PROC_DIR / "feature_names.json").read_text(encoding="utf-8"))
    df_bal = pd.read_parquet(PROC_DIR / "discursos_balanced.parquet")
    df_summary = pd.read_csv(PROC_DIR / "results_summary.csv")
    df_grid = pd.read_csv(PROC_DIR / "sensitivity_grid.csv")
    df_viz = pd.read_parquet(PROC_DIR / "viz_points_2d.parquet")
    ct_counts = pd.read_csv(PROC_DIR / "president_cluster_counts.csv", index_col=0)
    ct_pct = pd.read_csv(PROC_DIR / "president_cluster_pct.csv", index_col=0)
    df_profiles = pd.read_csv(PROC_DIR / "cluster_stylo_profiles.csv")
    df_loadings = pd.read_csv(PROC_DIR / "pca_loadings.csv", index_col=0)
    plot_npz = np.load(PROC_DIR / "plot_data.npz")

    best_scen_id = str(df_viz["best_scenario"].iloc[0])
    print(f"Generating 9 seminar figures for best_scen_id={best_scen_id}...")

    fig01_corpus_overview(meta, df_bal)
    fig02_curse_of_dimensionality(plot_npz, df_summary, best_scen_id)
    fig03_dr_comparison_metrics(df_summary)
    fig04_umap_vs_tsne_pitfall(df_viz, df_summary)
    fig05_sensitivity_heatmap(df_grid, df_summary, best_scen_id)
    fig06_clusters_vs_presidents_2d(df_viz, df_summary, best_scen_id)
    fig07_heatmap_president_cluster(ct_counts, ct_pct)
    fig08_cluster_stylometric_profiles(df_profiles, df_loadings)
    fig09_dbscan_vs_kmeans(df_viz, df_summary, best_scen_id)

    figs = sorted(FIG_DIR.glob("*.png"))
    print(f"Generated {len(figs)} figures in {FIG_DIR}:")
    for f in figs:
        print(f" - {f.name} ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
