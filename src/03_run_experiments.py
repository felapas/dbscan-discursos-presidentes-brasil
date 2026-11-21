#!/usr/bin/env python3
"""Run controlled DBSCAN x Dimensionality Reduction experiments on Brazilian presidential speeches."""

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
import numpy as np
import pandas as pd
import umap
from kneed import KneeLocator
from scipy import sparse
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.neighbors import NearestNeighbors

SEED = 42
MIN_PTS_GRID = [10, 15, 20, 30]
DEFAULT_MIN_PTS = 15

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_ROOT / "data" / "processed"


def compute_dbcv(Z: np.ndarray, labels: np.ndarray) -> float:
    """Compute Density-Based Clustering Validation (DBCV) safely."""
    unique_clusters = set(labels) - {-1}
    if len(unique_clusters) < 2:
        return float("nan")
    # Need at least 2 points in >= 2 clusters
    non_noise = labels != -1
    if int(non_noise.sum()) < 4:
        return float("nan")
    try:
        val = hdbscan.validity.validity_index(
            np.ascontiguousarray(Z, dtype=np.float64),
            np.ascontiguousarray(labels, dtype=np.int64),
            metric="euclidean",
        )
        return float(val)
    except Exception:
        return float("nan")


def compute_k_distance(Z: np.ndarray, min_pts: int, metric: str) -> tuple[np.ndarray, float]:
    """Compute sorted k-distances (k = min_pts - 1) and knee epsilon."""
    nn = NearestNeighbors(n_neighbors=min_pts, metric=metric)
    nn.fit(Z)
    dists, _ = nn.kneighbors(Z)
    k_dists = np.sort(dists[:, min_pts - 1])
    x = np.arange(len(k_dists), dtype=np.float64)

    eps_knee = None
    try:
        kl = KneeLocator(x, k_dists, curve="convex", direction="increasing", S=1.0)
        if kl.knee is not None:
            eps_knee = float(k_dists[int(kl.knee)])
    except Exception:
        eps_knee = None

    if eps_knee is None or eps_knee <= 0:
        # Geometric fallback: maximum perpendicular distance to chord connecting endpoints
        p1 = np.array([x[0], k_dists[0]])
        p2 = np.array([x[-1], k_dists[-1]])
        line = p2 - p1
        norm_line = np.linalg.norm(line)
        if norm_line > 0:
            pts = np.column_stack([x, k_dists])
            cross = np.abs(np.cross(line, p1 - pts)) / norm_line
            eps_knee = float(k_dists[int(np.argmax(cross))])
        else:
            eps_knee = float(np.median(k_dists))

    return k_dists, float(eps_knee)


def distance_concentration_stats(Z: np.ndarray, metric: str, rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    """Compute exact pairwise distance summary and a 50k sample for histograms."""
    dists = pdist(Z, metric=metric)
    d_min = float(np.min(dists))
    d_p01, d_p05, d_p50, d_p95, d_p99 = [
        float(v) for v in np.quantile(dists, [0.01, 0.05, 0.50, 0.95, 0.99])
    ]
    d_max = float(np.max(dists))
    mean_d = float(np.mean(dists))
    std_d = float(np.std(dists))
    cv_dist = std_d / (mean_d + 1e-12)
    rel_contrast = (d_p95 - d_p05) / (d_p50 + 1e-12)
    max_min_ratio = (d_max - d_min) / (d_min + 1e-9)

    sample_idx = rng.choice(len(dists), size=min(50000, len(dists)), replace=False)
    sample = dists[sample_idx].astype(np.float32)

    stats = {
        "d_min": d_min,
        "d_p01": d_p01,
        "d_p05": d_p05,
        "d_p50": d_p50,
        "d_p95": d_p95,
        "d_p99": d_p99,
        "d_max": d_max,
        "cv_dist": cv_dist,
        "rel_contrast": rel_contrast,
        "max_min_ratio": max_min_ratio,
    }
    return stats, sample


def evaluate_clustering(
    Z: np.ndarray,
    Z_for_dbcv: np.ndarray,
    labels: np.ndarray,
    y_true: np.ndarray,
    metric: str,
) -> dict:
    """Compute internal and external clustering quality metrics."""
    n = len(labels)
    unique_clusters = set(labels) - {-1}
    n_clusters = len(unique_clusters)
    noise_mask = labels == -1
    non_noise_mask = ~noise_mask
    noise_count = int(noise_mask.sum())
    noise_pct = 100.0 * noise_count / n

    dbcv = compute_dbcv(Z_for_dbcv, labels)

    if n_clusters >= 2 and int(non_noise_mask.sum()) >= n_clusters + 2:
        try:
            sil_nn = float(silhouette_score(Z[non_noise_mask], labels[non_noise_mask], metric=metric))
        except Exception:
            sil_nn = float("nan")
    else:
        sil_nn = float("nan")

    ari_all = float(adjusted_rand_score(y_true, labels))
    if int(non_noise_mask.sum()) >= 2 and len(set(labels[non_noise_mask])) >= 1:
        ari_nn = float(adjusted_rand_score(y_true[non_noise_mask], labels[non_noise_mask]))
    else:
        ari_nn = float("nan")

    nmi_all = float(normalized_mutual_info_score(y_true, labels))
    homog = float(homogeneity_score(y_true, labels))
    compl = float(completeness_score(y_true, labels))
    v_meas = float(v_measure_score(y_true, labels))

    return {
        "n_clusters": int(n_clusters),
        "noise_count": noise_count,
        "noise_pct": float(noise_pct),
        "dbcv": float(dbcv),
        "silhouette_no_noise": float(sil_nn),
        "ari_all": ari_all,
        "ari_no_noise": ari_nn,
        "nmi_all": nmi_all,
        "homogeneity": homog,
        "completeness": compl,
        "v_measure": v_meas,
    }


def run_grid_for_space(
    scenario: str,
    label_desc: str,
    Z: np.ndarray,
    metric: str,
    y_true: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[dict], dict, dict, dict]:
    """Run k-distance knee selection and full (eps x MinPts) sensitivity grid for one space."""
    # For cosine metric on L2-normalized high-dim space, Euclidean on Z matches distance ordering for DBCV
    Z_for_dbcv = Z
    dist_stats, dist_sample = distance_concentration_stats(Z, metric, rng)

    grid_rows: list[dict] = []
    auto_knee_row = None
    auto_knee_labels = None
    auto_knee_core_idx = None
    k_dist_15 = None
    eps_knee_15 = None

    for min_pts in MIN_PTS_GRID:
        k_dists, eps_knee = compute_k_distance(Z, min_pts=min_pts, metric=metric)
        if min_pts == DEFAULT_MIN_PTS:
            k_dist_15 = k_dists
            eps_knee_15 = eps_knee

        # Grid of eps values covering around the knee + quantiles of k-distance distribution
        eps_candidates = np.concatenate(
            [
                np.linspace(0.45 * eps_knee, 1.85 * eps_knee, 12),
                np.quantile(k_dists, np.linspace(0.08, 0.92, 9)),
            ]
        )
        eps_candidates = np.unique(np.round(eps_candidates[eps_candidates > 1e-6], 6))

        # Ensure exact eps_knee is always evaluated
        if eps_knee > 1e-6 and not np.any(np.isclose(eps_candidates, eps_knee, atol=1e-6)):
            eps_candidates = np.sort(np.append(eps_candidates, eps_knee))

        for eps in eps_candidates:
            db = DBSCAN(eps=float(eps), min_samples=min_pts, metric=metric)
            labels = db.fit_predict(Z)
            metrics = evaluate_clustering(Z, Z_for_dbcv, labels, y_true, metric=metric)
            row = {
                "scenario": scenario,
                "description": label_desc,
                "dim": int(Z.shape[1]),
                "metric": metric,
                "min_pts": int(min_pts),
                "eps": float(eps),
                "is_knee": bool(np.isclose(eps, eps_knee, atol=1e-6)),
                **metrics,
                **dist_stats,
            }
            grid_rows.append(row)

            if min_pts == DEFAULT_MIN_PTS and row["is_knee"]:
                auto_knee_row = {**row, "selection_mode": "auto_knee"}
                auto_knee_labels = labels.copy()
                auto_knee_core_idx = db.core_sample_indices_.copy()

    # Unsupervised best DBCV selection with genuine noise detection (1.0% <= noise <= 45.0%)
    df_g = pd.DataFrame(grid_rows)
    unrestricted = df_g[(df_g["n_clusters"] >= 2) & (df_g["noise_pct"] <= 45.0) & df_g["dbcv"].notna()]
    valid = df_g[
        (df_g["n_clusters"] >= 2)
        & (df_g["noise_pct"] >= 1.0)
        & (df_g["noise_pct"] <= 45.0)
        & df_g["dbcv"].notna()
    ]
    if len(valid) == 0:
        valid = unrestricted
    if len(valid) == 0:
        valid = df_g[(df_g["n_clusters"] >= 2) & df_g["dbcv"].notna()]
    if len(valid) > 0:
        best_idx = int(valid["dbcv"].idxmax())
        best_spec = df_g.loc[best_idx].to_dict()
        if len(unrestricted) > 0:
            unr_row = df_g.loc[int(unrestricted["dbcv"].idxmax())]
            if unr_row["noise_pct"] < 1.0:
                print(
                    f"  [note] {scenario}: unrestricted max DBCV={unr_row['dbcv']:.4f} had "
                    f"noise={unr_row['noise_pct']:.2f}% (k={int(unr_row['n_clusters'])}, eps={unr_row['eps']:.4f}); "
                    f"with >=1% noise constraint -> DBCV={best_spec['dbcv']:.4f} "
                    f"(noise={best_spec['noise_pct']:.2f}%, k={int(best_spec['n_clusters'])}, eps={best_spec['eps']:.4f})"
                )
    else:
        # Space collapsed (<= 1 cluster everywhere); prefer row with min_pts=15 closest to knee
        fallback = df_g[(df_g["min_pts"] == DEFAULT_MIN_PTS) & df_g["is_knee"]]
        best_spec = fallback.iloc[0].to_dict() if len(fallback) else df_g.iloc[0].to_dict()

    db_best = DBSCAN(eps=float(best_spec["eps"]), min_samples=int(best_spec["min_pts"]), metric=metric)
    best_labels = db_best.fit_predict(Z)
    best_row = {**best_spec, "selection_mode": "best_dbcv"}

    artifacts = {
        "k_dist_15": k_dist_15,
        "eps_knee_15": eps_knee_15,
        "dist_sample": dist_sample,
        "auto_knee_labels": auto_knee_labels,
        "auto_knee_core_idx": auto_knee_core_idx,
        "best_labels": best_labels,
        "best_core_idx": db_best.core_sample_indices_.copy(),
    }
    return grid_rows, auto_knee_row, best_row, artifacts


def main() -> None:
    rng = np.random.default_rng(SEED)

    df_bal = pd.read_parquet(PROC_DIR / "discursos_balanced.parquet")
    df_stylo = pd.read_parquet(PROC_DIR / "features_stylo23.parquet")
    X_stylo23_z = np.load(PROC_DIR / "X_stylo23_z.npy")
    X_style_highdim = np.load(PROC_DIR / "X_style_highdim.npy")
    X_tfidf_3000 = sparse.load_npz(PROC_DIR / "X_tfidf_3000.npz")
    meta = json.loads((PROC_DIR / "feature_names.json").read_text(encoding="utf-8"))

    pres_order = meta["presidents"]
    pres_to_id = {p: i for i, p in enumerate(pres_order)}
    y_true = df_bal["president"].map(pres_to_id).to_numpy()

    print("=== BUILDING DIMENSIONALITY REDUCTION SPACES ===")
    pca_2 = PCA(n_components=2, random_state=SEED).fit_transform(X_style_highdim)
    pca_10 = PCA(n_components=10, random_state=SEED).fit_transform(X_style_highdim)
    pca_50 = PCA(n_components=50, random_state=SEED).fit_transform(X_style_highdim)
    svd_10 = TruncatedSVD(n_components=10, random_state=SEED).fit_transform(X_style_highdim)

    umap_10_md0 = umap.UMAP(
        n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=SEED
    ).fit_transform(X_style_highdim)
    umap_10_md05 = umap.UMAP(
        n_components=10, n_neighbors=15, min_dist=0.5, metric="cosine", random_state=SEED
    ).fit_transform(X_style_highdim)
    pca50_umap10 = umap.UMAP(
        n_components=10, n_neighbors=15, min_dist=0.0, metric="euclidean", random_state=SEED
    ).fit_transform(pca_50)

    tsne_p10 = TSNE(
        n_components=2, perplexity=10.0, metric="cosine", init="random", random_state=SEED
    ).fit_transform(X_style_highdim)
    tsne_p50 = TSNE(
        n_components=2, perplexity=50.0, metric="cosine", init="random", random_state=SEED
    ).fit_transform(X_style_highdim)

    stylo23_pca10 = PCA(n_components=10, random_state=SEED).fit_transform(X_stylo23_z)
    tfidf_svd50 = TruncatedSVD(n_components=50, random_state=SEED).fit_transform(X_tfidf_3000)
    tfidf_umap10 = umap.UMAP(
        n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=SEED
    ).fit_transform(tfidf_svd50)

    # Dedicated 2D UMAP projections for UMAP min_dist comparison (fig04)
    umap_2d_md0 = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=SEED
    ).fit_transform(X_style_highdim)
    umap_2d_md05 = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.5, metric="cosine", random_state=SEED
    ).fit_transform(X_style_highdim)

    scenarios = [
        ("01_orig_euclid", "Original 528D (Euclidiana)", X_style_highdim, "euclidean"),
        ("02_orig_cosine", "Original 528D (Cosseno)", X_style_highdim, "cosine"),
        ("03_pca_2d", "PCA 2D (Euclidiana)", pca_2, "euclidean"),
        ("04_pca_10d", "PCA 10D (Euclidiana)", pca_10, "euclidean"),
        ("05_svd_10d", "Truncated SVD 10D (Euclidiana)", svd_10, "euclidean"),
        ("06_umap_10d_md0", "UMAP 10D (min_dist=0.0)", umap_10_md0, "euclidean"),
        ("07_umap_10d_md05", "UMAP 10D (min_dist=0.5)", umap_10_md05, "euclidean"),
        ("08_pca50_umap10", "PCA 50D → UMAP 10D", pca50_umap10, "euclidean"),
        ("09_tsne_perp10", "t-SNE 2D (perplexity=10)", tsne_p10, "euclidean"),
        ("10_tsne_perp50", "t-SNE 2D (perplexity=50)", tsne_p50, "euclidean"),
        ("11_stylo23_pca10", "Só Bloco A 23D → PCA 10D", stylo23_pca10, "euclidean"),
        ("12_tfidf3000_svd50_umap10", "TF-IDF 3000D → SVD 50D → UMAP 10D", tfidf_umap10, "euclidean"),
    ]

    all_grid_rows: list[dict] = []
    summary_rows: list[dict] = []
    space_artifacts: dict[str, dict] = {}
    space_arrays: dict[str, np.ndarray] = {}

    for scen_id, desc, Z, metric in scenarios:
        print(f"Running grid for {scen_id} (dim={Z.shape[1]}, metric={metric})...")
        g_rows, knee_row, best_row, arts = run_grid_for_space(
            scen_id, desc, Z, metric, y_true, rng
        )
        all_grid_rows.extend(g_rows)
        summary_rows.append(knee_row)
        summary_rows.append(best_row)
        space_artifacts[scen_id] = arts
        space_arrays[scen_id] = Z

    # Identify best pure-style DBSCAN scenario (among 01..08, maximizing unsupervised DBCV)
    pure_style_ids = [
        "01_orig_euclid", "02_orig_cosine", "03_pca_2d", "04_pca_10d",
        "05_svd_10d", "06_umap_10d_md0", "07_umap_10d_md05", "08_pca50_umap10",
    ]
    df_sum = pd.DataFrame(summary_rows)
    cand_best = df_sum[
        (df_sum["scenario"].isin(pure_style_ids))
        & (df_sum["selection_mode"] == "best_dbcv")
        & (df_sum["n_clusters"] >= 2)
    ].copy()
    if len(cand_best) == 0:
        raise RuntimeError("All pure-style DR scenarios collapsed to <= 1 cluster!")
    winner_row = cand_best.sort_values("dbcv", ascending=False).iloc[0]
    best_scen_id = str(winner_row["scenario"])
    print(f"\n>>> Best pure-style DBSCAN scenario (by unsupervised DBCV with >=1% noise): {best_scen_id}")

    # Aligned 2D UMAP projection for visualizing the winning pure-style space
    if best_scen_id == "08_pca50_umap10":
        umap_2d_viz = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=SEED
        ).fit_transform(pca_50)
    else:
        umap_2d_viz = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED
        ).fit_transform(X_style_highdim)

    # 13_kmeans_on_best: KMeans(K=6) on exact same space as best_scen_id
    Z_best = space_arrays[best_scen_id]
    km = KMeans(n_clusters=len(pres_order), n_init=20, random_state=SEED)
    km_labels = km.fit_predict(Z_best)
    km_metrics = evaluate_clustering(Z_best, Z_best, km_labels, y_true, metric="euclidean")
    best_dist_stats = {
        k: winner_row[k]
        for k in [
            "d_min", "d_p01", "d_p05", "d_p50", "d_p95", "d_p99", "d_max",
            "cv_dist", "rel_contrast", "max_min_ratio",
        ]
    }
    km_row = {
        "scenario": "13_kmeans_on_best",
        "description": f"K-Means (K=6) em {winner_row['description']}",
        "dim": int(Z_best.shape[1]),
        "metric": "euclidean",
        "min_pts": 0,
        "eps": float("nan"),
        "is_knee": False,
        "selection_mode": "fixed_k6",
        **km_metrics,
        **best_dist_stats,
    }
    summary_rows.append(km_row)

    df_summary = pd.DataFrame(summary_rows)
    df_grid = pd.DataFrame(all_grid_rows)
    df_summary.to_csv(PROC_DIR / "results_summary.csv", index=False)
    df_grid.to_csv(PROC_DIR / "sensitivity_grid.csv", index=False)

    print("\n=== RESULTS SUMMARY (best_dbcv & fixed_k6) ===")
    cols_show = [
        "scenario", "selection_mode", "dim", "min_pts", "eps",
        "n_clusters", "noise_pct", "dbcv", "silhouette_no_noise",
        "ari_all", "homogeneity", "completeness", "cv_dist", "rel_contrast",
    ]
    print(
        df_summary[df_summary["selection_mode"].isin(["best_dbcv", "fixed_k6"])][cols_show]
        .round(4)
        .to_string(index=False)
    )
    print("\n=== RESULTS SUMMARY (auto_knee MinPts=15) ===")
    print(
        df_summary[df_summary["selection_mode"] == "auto_knee"][cols_show]
        .round(4)
        .to_string(index=False)
    )

    # Stability test across 10 UMAP random seeds (on winning UMAP recipe or 06_umap_10d_md0)
    stab_target = best_scen_id if "umap" in best_scen_id else "06_umap_10d_md0"
    print(f"\nRunning 10-seed UMAP stability check for {stab_target}...")
    stab_rows = []
    for s in range(10):
        if stab_target == "08_pca50_umap10":
            Z_s = umap.UMAP(
                n_components=10, n_neighbors=15, min_dist=0.0, metric="euclidean", random_state=s
            ).fit_transform(pca_50)
        elif stab_target == "07_umap_10d_md05":
            Z_s = umap.UMAP(
                n_components=10, n_neighbors=15, min_dist=0.5, metric="cosine", random_state=s
            ).fit_transform(X_style_highdim)
        else:
            Z_s = umap.UMAP(
                n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=s
            ).fit_transform(X_style_highdim)

        _, k_s, b_s, _ = run_grid_for_space(stab_target, stab_target, Z_s, "euclidean", y_true, rng)
        stab_rows.append({"seed": s, **b_s})
    df_stab = pd.DataFrame(stab_rows)
    df_stab.to_csv(PROC_DIR / "stability_umap.csv", index=False)
    print("UMAP stability (10 seeds) mean ± std:")
    for m_col in ["n_clusters", "noise_pct", "dbcv", "ari_all", "homogeneity"]:
        print(f"  {m_col:14s}: {df_stab[m_col].mean():.4f} ± {df_stab[m_col].std():.4f}")

    # Qualitative profiles & visualization points for best pure-style DBSCAN scenario
    best_labels = space_artifacts[best_scen_id]["best_labels"]
    best_core_set = set(int(i) for i in space_artifacts[best_scen_id]["best_core_idx"])
    point_type = [
        "noise" if best_labels[i] == -1 else ("core" if i in best_core_set else "border")
        for i in range(len(best_labels))
    ]

    # Map visualization coordinates (if winner is PCA 2D/10D vs UMAP, still store dedicated UMAP 2D + PCA 2D)
    df_viz = pd.DataFrame(
        {
            "doc_id": df_bal["doc_id"],
            "president": df_bal["president"],
            "date": df_bal["date"],
            "word_count": df_bal["word_count"],
            "umap2d_x": umap_2d_viz[:, 0],
            "umap2d_y": umap_2d_viz[:, 1],
            "pca2d_x": pca_2[:, 0],
            "pca2d_y": pca_2[:, 1],
            "umap2d_md0_x": umap_2d_md0[:, 0],
            "umap2d_md0_y": umap_2d_md0[:, 1],
            "label_umap_md0": space_artifacts["06_umap_10d_md0"]["best_labels"],
            "umap2d_md05_x": umap_2d_md05[:, 0],
            "umap2d_md05_y": umap_2d_md05[:, 1],
            "label_umap_md05": space_artifacts["07_umap_10d_md05"]["best_labels"],
            "tsne_p10_x": tsne_p10[:, 0],
            "tsne_p10_y": tsne_p10[:, 1],
            "label_tsne_p10": space_artifacts["09_tsne_perp10"]["best_labels"],
            "tsne_p50_x": tsne_p50[:, 0],
            "tsne_p50_y": tsne_p50[:, 1],
            "label_tsne_p50": space_artifacts["10_tsne_perp50"]["best_labels"],
            "best_scenario": best_scen_id,
            "best_cluster": best_labels,
            "point_type": point_type,
            "kmeans_cluster": km_labels,
        }
    )
    df_viz.to_parquet(PROC_DIR / "viz_points_2d.parquet", index=False)

    # Contingency tables: President x Cluster (including Ruído (-1))
    cluster_cols = sorted([c for c in set(best_labels) if c != -1]) + ([-1] if -1 in set(best_labels) else [])
    col_names = {c: ("Ruído (-1)" if c == -1 else f"Cluster {c}") for c in cluster_cols}
    ct_counts = pd.crosstab(df_bal["president"], pd.Series(best_labels, name="cluster"))
    ct_counts = ct_counts.reindex(index=pres_order, columns=cluster_cols, fill_value=0)
    ct_counts = ct_counts.rename(columns=col_names)
    ct_pct = ct_counts.div(ct_counts.sum(axis=1), axis=0) * 100.0
    ct_counts.to_csv(PROC_DIR / "president_cluster_counts.csv")
    ct_pct.round(2).to_csv(PROC_DIR / "president_cluster_pct.csv")

    print("\n=== CONTINGENCY MATRIX (% BY PRESIDENT) ===")
    print(ct_pct.round(1).to_string())

    # Stylometric Z-score profile of each cluster (using core points; and all points for Noise -1)
    stylo_cols = meta["stylo_cols"]
    df_z = pd.DataFrame(X_stylo23_z, columns=stylo_cols)
    df_z["cluster"] = best_labels
    df_z["is_core"] = [pt == "core" for pt in point_type]

    profile_rows = []
    for c in cluster_cols:
        if c == -1:
            sub = df_z[df_z["cluster"] == -1]
            sub_raw = df_stylo[best_labels == -1]
            label_str = "Ruído (-1)"
        else:
            sub = df_z[(df_z["cluster"] == c) & (df_z["is_core"])]
            sub_raw = df_stylo[(best_labels == c) & (np.array(point_type) == "core")]
            if len(sub) == 0:
                sub = df_z[df_z["cluster"] == c]
                sub_raw = df_stylo[best_labels == c]
            label_str = f"Cluster {c}"
        row_z = sub[stylo_cols].mean().to_dict()
        row_raw = {f"raw_{k}": float(v) for k, v in sub_raw[stylo_cols].mean().items()}
        pres_share = (
            df_bal.loc[best_labels == c, "president"]
            .value_counts(normalize=True)
            .head(2)
            .to_dict()
        )
        top_pres = ", ".join(f"{k} ({v*100:.0f}%)" for k, v in pres_share.items())
        profile_rows.append(
            {
                "cluster_id": c,
                "cluster_label": label_str,
                "n_total": int((best_labels == c).sum()),
                "n_core": int(len(sub)),
                "top_presidents": top_pres,
                **row_z,
                **row_raw,
            }
        )
    df_profiles = pd.DataFrame(profile_rows)
    df_profiles.to_csv(PROC_DIR / "cluster_stylo_profiles.csv", index=False)

    # Top over-represented function words per cluster (from Bloco B slice in X_style_highdim)
    n_stylo = meta["dims"]["stylo23"]
    n_func = meta["dims"]["func"]
    func_names = [fn.replace("func_", "") for fn in meta["func_names"]]
    X_func_block = X_style_highdim[:, n_stylo : n_stylo + n_func]
    func_rows = []
    for c in cluster_cols:
        mask_c = best_labels == c
        if c != -1:
            mask_core = mask_c & (np.array(point_type) == "core")
            if mask_core.sum() > 0:
                mask_c = mask_core
        diff = X_func_block[mask_c].mean(axis=0) - X_func_block[~mask_c].mean(axis=0)
        top_idx = np.argsort(diff)[::-1][:10]
        for rank, idx in enumerate(top_idx, start=1):
            func_rows.append(
                {
                    "cluster_id": c,
                    "cluster_label": "Ruído (-1)" if c == -1 else f"Cluster {c}",
                    "rank": rank,
                    "function_word": func_names[idx],
                    "mean_diff_tfidf": float(diff[idx]),
                }
            )
    pd.DataFrame(func_rows).to_csv(PROC_DIR / "cluster_top_func_words.csv", index=False)

    # PCA loadings on interpretable Bloco A (23 metrics)
    pca_stylo = PCA(n_components=3, random_state=SEED).fit(X_stylo23_z)
    df_loadings = pd.DataFrame(
        pca_stylo.components_.T,
        index=stylo_cols,
        columns=["PC1", "PC2", "PC3"],
    )
    df_loadings.index.name = "feature"
    df_loadings["explained_var_PC1"] = float(pca_stylo.explained_variance_ratio_[0])
    df_loadings["explained_var_PC2"] = float(pca_stylo.explained_variance_ratio_[1])
    df_loadings["explained_var_PC3"] = float(pca_stylo.explained_variance_ratio_[2])
    df_loadings.to_csv(PROC_DIR / "pca_loadings.csv")

    # Representative core speeches per cluster & most isolated noise (-1) speeches
    nn_best = NearestNeighbors(n_neighbors=16, metric="euclidean").fit(Z_best)
    dists_best, _ = nn_best.kneighbors(Z_best)
    kth_dist_best = dists_best[:, 15]

    rep_payload: dict = {"best_scenario": best_scen_id, "clusters": {}, "noise": []}
    for c in [x for x in cluster_cols if x != -1]:
        core_mask = (best_labels == c) & (np.array(point_type) == "core")
        idxs = np.where(core_mask)[0]
        if len(idxs) == 0:
            idxs = np.where(best_labels == c)[0]
        centroid = Z_best[idxs].mean(axis=0, keepdims=True)
        d_cent = cdist(Z_best[idxs], centroid, metric="euclidean").ravel()
        chosen = idxs[np.argsort(d_cent)[:2]]
        items = []
        for idx in chosen:
            items.append(
                {
                    "doc_id": str(df_bal.loc[idx, "doc_id"]),
                    "president": str(df_bal.loc[idx, "president"]),
                    "date": str(df_bal.loc[idx, "date"]),
                    "word_count": int(df_bal.loc[idx, "word_count"]),
                    "flesch_pt": round(float(df_stylo.loc[idx, "flesch_pt"]), 1),
                    "avg_sent_len": round(float(df_stylo.loc[idx, "avg_sent_len"]), 1),
                    "orality_rate": round(float(df_stylo.loc[idx, "orality_rate"]), 1),
                    "formal_connector_rate": round(float(df_stylo.loc[idx, "formal_connector_rate"]), 1),
                    "excerpt": str(df_bal.loc[idx, "text_clean"])[:380],
                }
            )
        rep_payload["clusters"][f"Cluster {c}"] = items

    noise_idxs = np.where(best_labels == -1)[0]
    if len(noise_idxs) > 0:
        # Pick 6 diverse/isolated noise speeches across presidents
        noise_sorted = noise_idxs[np.argsort(kth_dist_best[noise_idxs])[::-1]]
        picked = []
        seen_pres = set()
        for idx in noise_sorted:
            p = str(df_bal.loc[idx, "president"])
            if p not in seen_pres:
                picked.append(idx)
                seen_pres.add(p)
            if len(picked) >= 6:
                break
        for idx in noise_sorted:
            if len(picked) >= 6:
                break
            if idx not in picked:
                picked.append(idx)
        for idx in picked:
            rep_payload["noise"].append(
                {
                    "doc_id": str(df_bal.loc[idx, "doc_id"]),
                    "president": str(df_bal.loc[idx, "president"]),
                    "date": str(df_bal.loc[idx, "date"]),
                    "word_count": int(df_bal.loc[idx, "word_count"]),
                    "k15_dist": round(float(kth_dist_best[idx]), 4),
                    "flesch_pt": round(float(df_stylo.loc[idx, "flesch_pt"]), 1),
                    "avg_sent_len": round(float(df_stylo.loc[idx, "avg_sent_len"]), 1),
                    "mtld": round(float(df_stylo.loc[idx, "mtld"]), 1),
                    "question_rate": round(float(df_stylo.loc[idx, "question_rate"]), 1),
                    "orality_rate": round(float(df_stylo.loc[idx, "orality_rate"]), 1),
                    "formal_connector_rate": round(float(df_stylo.loc[idx, "formal_connector_rate"]), 1),
                    "excerpt": str(df_bal.loc[idx, "text_clean"])[:380],
                }
            )
    (PROC_DIR / "representative_and_noise_speeches.json").write_text(
        json.dumps(rep_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Save arrays for plotting (distributions & k-distance curves)
    np.savez_compressed(
        PROC_DIR / "plot_data.npz",
        orig_euclid_dists=space_artifacts["01_orig_euclid"]["dist_sample"],
        orig_euclid_kdist=space_artifacts["01_orig_euclid"]["k_dist_15"],
        orig_euclid_knee=np.array([space_artifacts["01_orig_euclid"]["eps_knee_15"]]),
        orig_cosine_dists=space_artifacts["02_orig_cosine"]["dist_sample"],
        orig_cosine_kdist=space_artifacts["02_orig_cosine"]["k_dist_15"],
        orig_cosine_knee=np.array([space_artifacts["02_orig_cosine"]["eps_knee_15"]]),
        pca10_dists=space_artifacts["04_pca_10d"]["dist_sample"],
        pca10_kdist=space_artifacts["04_pca_10d"]["k_dist_15"],
        pca10_knee=np.array([space_artifacts["04_pca_10d"]["eps_knee_15"]]),
        umap10_dists=space_artifacts["06_umap_10d_md0"]["dist_sample"],
        umap10_kdist=space_artifacts["06_umap_10d_md0"]["k_dist_15"],
        umap10_knee=np.array([space_artifacts["06_umap_10d_md0"]["eps_knee_15"]]),
        best_dists=space_artifacts[best_scen_id]["dist_sample"],
        best_kdist=space_artifacts[best_scen_id]["k_dist_15"],
        best_knee=np.array([space_artifacts[best_scen_id]["eps_knee_15"]]),
    )
    print(f"\nAll experiment artifacts saved to {PROC_DIR}")


if __name__ == "__main__":
    main()
