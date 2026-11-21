#!/usr/bin/env python3
"""Download and inspect the Brazilian presidential speeches dataset (1985-2022)."""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

import kagglehub
import pandas as pd
import requests

KAGGLE_HANDLE = "pascoalgoncalves/discursos-presidentes-civis-do-brasil-1985-2022"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_via_kagglehub(handle: str) -> Path:
    """Download dataset via kagglehub and copy files into data/raw."""
    print(f"Downloading via kagglehub: {handle}")
    cache_path = Path(kagglehub.dataset_download(handle))
    print(f"kagglehub cache path: {cache_path}")
    copied = 0
    for item in cache_path.rglob("*"):
        if item.is_file():
            rel = item.relative_to(cache_path)
            dest = RAW_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)
            copied += 1
    print(f"Copied {copied} file(s) to {RAW_DIR}")
    return RAW_DIR


def download_via_http(handle: str) -> Path:
    """Fallback direct HTTP download of Kaggle dataset archive."""
    url = f"https://www.kaggle.com/api/v1/datasets/download/{handle}"
    print(f"Trying direct HTTP download: {url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, stream=True, allow_redirects=True, timeout=120, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} downloading {url}: {resp.text[:300]}")

    zip_path = RAW_DIR / "dataset.zip"
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink(missing_ok=True)
    return RAW_DIR


def read_tabular_auto(path: Path) -> pd.DataFrame:
    """Read CSV/Parquet/JSON with automatic encoding and delimiter detection."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, low_memory=False)
                if df.shape[1] > 1 or sep == "\t":
                    print(f"Loaded {path.name} with encoding={enc}, sep={repr(sep)}")
                    return df
            except Exception:
                continue
    return pd.read_csv(path, low_memory=False)


def inspect_raw_files() -> dict:
    """Inspect all downloaded data files (tabular or .txt tree) and print dataset summary."""
    files = sorted(
        [
            p
            for p in RAW_DIR.rglob("*")
            if p.is_file() and p.name != "dataset_inventory.json"
        ]
    )
    print(f"\n=== RAW FILES ({len(files)} total) ===")
    ext_counts: dict[str, int] = {}
    for f in files:
        ext = f.suffix.lower() or "<no_ext>"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    print("File counts by extension:", ext_counts)
    for f in files[:25]:
        print(f" - {f.relative_to(PROJECT_ROOT)} ({f.stat().st_size / 1e3:.1f} KB)")
    if len(files) > 25:
        print(f" ... ({len(files) - 25} more files)")

    tabular_files = [
        f for f in files if f.suffix.lower() in {".csv", ".parquet", ".json", ".xlsx", ".xls", ".tsv"}
    ]
    txt_files = [f for f in files if f.suffix.lower() == ".txt"]

    inventory: dict = {
        "total_files": len(files),
        "ext_counts": ext_counts,
        "tabular_files": [str(f.relative_to(PROJECT_ROOT)) for f in tabular_files],
        "txt_count": len(txt_files),
    }

    for f in tabular_files:
        print(f"\n=== INSPECTING TABULAR: {f.relative_to(PROJECT_ROOT)} ===")
        df = read_tabular_auto(f)
        print(f"Shape: {df.shape}")
        print("Columns & dtypes:")
        for col in df.columns:
            nulls = int(df[col].isna().sum())
            print(f"  - {col}: {df[col].dtype} (nulls={nulls})")

        lower_map = {c.lower(): c for c in df.columns}
        pres_col = None
        for key in ["presidente", "president", "autor", "orador", "nome"]:
            for lc, orig in lower_map.items():
                if key in lc:
                    pres_col = orig
                    break
            if pres_col:
                break
        if pres_col is None:
            for c in df.columns:
                if df[c].dtype == object or str(df[c].dtype) == "str":
                    nu = df[c].nunique(dropna=True)
                    if 3 <= nu <= 25:
                        pres_col = c
                        break

        text_col = None
        for key in ["discurso", "texto", "speech", "text", "conteudo", "content", "transcricao"]:
            for lc, orig in lower_map.items():
                if key in lc:
                    text_col = orig
                    break
            if text_col:
                break
        if text_col is None:
            best_len = -1.0
            for c in df.columns:
                if df[c].dtype == object or str(df[c].dtype) == "str":
                    med = df[c].dropna().astype(str).str.len().median()
                    if pd.notna(med) and med > best_len:
                        best_len = float(med)
                        text_col = c

        date_col = None
        for key in ["data", "date", "ano", "year"]:
            for lc, orig in lower_map.items():
                if key in lc:
                    date_col = orig
                    break
            if date_col:
                break

        print(f"Detected columns -> president={pres_col}, text={text_col}, date={date_col}")
        if pres_col:
            print("\nCounts by president:")
            print(df[pres_col].value_counts(dropna=False).to_string())
        if date_col:
            print(
                f"\nDate range sample ({date_col}): min={df[date_col].dropna().astype(str).min()}, max={df[date_col].dropna().astype(str).max()}"
            )
        if text_col:
            wc = df[text_col].fillna("").astype(str).str.split().str.len()
            pcts = wc.quantile([0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]).to_dict()
            print("\nWord count percentiles:")
            print({round(k, 2): int(v) for k, v in pcts.items()})
            print("\nSample speeches (truncated):")
            for idx, row in df.dropna(subset=[text_col]).head(3).iterrows():
                pres_val = row[pres_col] if pres_col else "?"
                date_val = row[date_col] if date_col else "?"
                snippet = re.sub(r"\s+", " ", str(row[text_col]))[:250]
                print(f"  [{idx}] ({pres_val} | {date_val}): {snippet}...")

        inventory[f.name] = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "pres_col": pres_col,
            "text_col": text_col,
            "date_col": date_col,
        }

    if txt_files:
        print(f"\n=== INSPECTING .TXT CORPUS ({len(txt_files)} files) ===")
        folder_counts: dict[str, int] = {}
        word_counts: list[int] = []
        samples: list[tuple[str, str]] = []
        for tf in txt_files:
            rel = tf.relative_to(RAW_DIR)
            top = rel.parts[0] if len(rel.parts) > 1 else "<root>"
            folder_counts[top] = folder_counts.get(top, 0) + 1
            try:
                txt = tf.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                txt = tf.read_text(encoding="latin-1")
            word_counts.append(len(txt.split()))
            if len(samples) < 3 or (top not in [s[0].split("/")[0] for s in samples] and len(samples) < 8):
                samples.append((str(rel), re.sub(r"\s+", " ", txt)[:260]))

        print("Counts by top-level folder:")
        for k, v in sorted(folder_counts.items()):
            print(f"  - {k}: {v}")
        wc_s = pd.Series(word_counts)
        pcts = wc_s.quantile([0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]).to_dict()
        print("\nWord count percentiles across .txt files:")
        print({round(k, 2): int(v) for k, v in pcts.items()})
        print("\nSample .txt speeches:")
        for rel_str, snip in samples:
            print(f"  [{rel_str}]: {snip}...")
        inventory["txt_folder_counts"] = folder_counts

    meta_path = RAW_DIR / "dataset_inventory.json"
    meta_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved inventory to {meta_path}")
    return inventory


def main() -> None:
    existing_data = [
        p
        for p in RAW_DIR.rglob("*")
        if p.is_file() and p.name != "dataset_inventory.json"
    ]
    if existing_data:
        print(f"Found {len(existing_data)} existing raw files in {RAW_DIR}, skipping download.")
    else:
        try:
            download_via_kagglehub(KAGGLE_HANDLE)
        except Exception as exc:
            print(f"kagglehub failed ({exc}); trying direct HTTP fallback...")
            download_via_http(KAGGLE_HANDLE)

    inspect_raw_files()


if __name__ == "__main__":
    main()
