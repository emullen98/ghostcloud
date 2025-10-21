#!/usr/bin/env python3
"""
Fractal Dimension from Perimeter–Area (log–log OLS), spec-accurate and minimal.

- Strict source separation: SOURCE ∈ {"png","siteperc"} (no pooling).
- PNG filters: time_windows (list), threshold_policy ("all" | "argmax_clouds"),
               thresholds_whitelist (optional with "all"),
               on_missing_metadata ("skip_run" | "keep_all_thresholds").
- Site-perc filters: p_vals (list or None).
- Common filters: min_area, max_area, optional min_perim/max_perim.
- Fit: OLS on logP ~ m*logA + b; report D = 2m ± 2*stderr_m.
- Output: single plot + metrics.json in
          scratch/all_clouds_data/analysis/fractal_dimension/
          with filenames prefixed by SUFFIX_TAG.

Requires: pandas, numpy, pyarrow (for parquet), matplotlib
"""

from __future__ import annotations
import os, re, glob, json, math
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Resolve all key paths absolutely (fixes discovery issues) ---
def _abs(p: str) -> str:
    """Return absolute, expanded, resolved path (handles relative paths safely)."""
    return str(Path(p).expanduser().resolve())



# ===================== CONFIG (EDIT) =====================

# Choose exactly one
SOURCE = "siteperc"        # "png" or "siteperc" (NEVER "both")

# Suffix tag to distinguish runs in filenames
SUFFIX_TAG = "SITEPERC_p04074"  # e.g., "PNG_allthr_alltimes", "SITEPERC_p04074"

# Common filters
MIN_AREA = 500
MAX_AREA = 7_500_000
MIN_PERIM: Optional[int] = None
MAX_PERIM: Optional[int] = None

# Site-perc filter (None => accept all present)
P_VALS: Optional[List[float]] = None

# PNG filters
# Time windows: list of ("HH:MM","HH:MM") in 24h; empty list => accept all times
TIME_WINDOWS: List[Tuple[str, str]] = []

# Threshold handling for PNG:
THRESHOLD_POLICY = "argmax_clouds"  # "all" or "argmax_clouds"

# When THRESHOLD_POLICY == "all": optional whitelist (None => accept all thresholds present)
THRESHOLDS_WHITELIST: Optional[List[float]] = None

# When THRESHOLD_POLICY == "argmax_clouds": behavior if meta JSON is missing/malformed
ON_MISSING_METADATA = "skip_run"    # "skip_run" (default) or "keep_all_thresholds"

# Roots (your structure)
PNG_PER_CLOUD_ROOT = "scratch/all_clouds_data/threshold_autocorr_bd/per_cloud"
PNG_META_ROOT      = "scratch/all_clouds_data/threshold_autocorr_bd"
SP_PER_CLOUD_ROOT  = "scratch/all_clouds_data/sp_autocorr_bd/per_cloud"

# Output directory (files will be prefixed by SUFFIX_TAG)
OUTPUT_DIR = "scratch/all_clouds_data/analysis/fractal_dimension"

# Plot look
FIGSIZE = (9.0, 7.0)
SCATTER_ALPHA = 0.25
SCATTER_S = 2
DPI = 300

# Float compare for thresholds
ATOL = 1e-12
RTOL = 1e-10

# --- Resolve to absolute paths so cwd doesn't matter ---
PNG_PER_CLOUD_ROOT = _abs(PNG_PER_CLOUD_ROOT)
PNG_META_ROOT      = _abs(PNG_META_ROOT)
SP_PER_CLOUD_ROOT  = _abs(SP_PER_CLOUD_ROOT)
OUTPUT_DIR         = _abs(OUTPUT_DIR)


# =========================================================


def _assert_source():
    if SOURCE not in {"png", "siteperc"}:
        raise ValueError('SOURCE must be "png" or "siteperc" (never "both").')
    if SOURCE == "png" and THRESHOLD_POLICY not in {"all", "argmax_clouds"}:
        raise ValueError('For PNG, THRESHOLD_POLICY must be "all" or "argmax_clouds".')
    if SOURCE == "png" and THRESHOLD_POLICY == "argmax_clouds" and ON_MISSING_METADATA not in {"skip_run", "keep_all_thresholds"}:
        raise ValueError('ON_MISSING_METADATA must be "skip_run" or "keep_all_thresholds".')


def _ensure_outdir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _hm_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


_TIME_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})--(\d{2})-(\d{2})-(\d{2})")

def _parse_time_minutes_from_run_tag(run_tag: str) -> Optional[int]:
    """
    run_tag example: thr_autocorr_bd_2012-12-30--03-15-16--375
    Returns minutes since midnight or None if not parsable.
    """
    m = _TIME_RE.search(run_tag)
    if not m:
        return None
    HH, MM = int(m.group(4)), int(m.group(5))
    return HH*60 + MM


def _within_any_window(run_tag: str, windows: List[Tuple[str, str]]) -> bool:
    if not windows:
        return True
    minutes = _parse_time_minutes_from_run_tag(run_tag)
    if minutes is None:
        # If time windows specified but we can't parse time, exclude
        return False
    for start, end in windows:
        s = _hm_to_minutes(start)
        e = _hm_to_minutes(end)
        if s <= e:
            if s <= minutes < e:
                return True
        else:
            # Overnight window (e.g., 22:00–03:00)
            if minutes >= s or minutes < e:
                return True
    return False


def _discover_png_runs() -> List[str]:
    """
    Return list of PNG run_tags (folder names under PNG_PER_CLOUD_ROOT).
    """
    pattern = os.path.join(PNG_PER_CLOUD_ROOT, "*")
    runs = []
    for d in glob.iglob(pattern):
        if os.path.isdir(d):
            runs.append(os.path.basename(d))
    return sorted(runs)


def _discover_sp_parquets() -> List[str]:
    """
    Return list of site-perc parquet shard paths.
    """
    pattern = os.path.join(SP_PER_CLOUD_ROOT, "*", "cloud_metrics.part*.parquet")
    return sorted(glob.iglob(pattern))


def _png_meta_path(run_tag: str) -> str:
    return os.path.join(PNG_META_ROOT, f"{run_tag}_meta.json")


def _choose_argmax_threshold(run_tag: str) -> Optional[float]:
    """
    Read num_clouds_by_threshold from the meta JSON and pick argmax (tie -> smallest thr).
    Returns chosen threshold (float) or None if unavailable/malformed.
    """
    meta_path = _png_meta_path(run_tag)
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        return None
    d = meta.get("num_clouds_by_threshold")
    if not isinstance(d, dict) or not d:
        return None
    best_thr = None
    best_cnt = None
    # Cast keys to float; counts to int
    for k, v in d.items():
        try:
            thr = float(k)
            cnt = int(v)
        except Exception:
            continue
        if best_cnt is None or cnt > best_cnt or (cnt == best_cnt and thr < best_thr):
            best_cnt = cnt
            best_thr = thr
    return best_thr


def _apply_common_filters(df: pd.DataFrame) -> pd.DataFrame:
    if "area" not in df.columns or "perim" not in df.columns:
        return df.iloc[0:0]
    df = df[(df["area"] > 0) & (df["perim"] > 0)]
    df = df[(df["area"] >= MIN_AREA) & (df["area"] <= MAX_AREA)]
    if MIN_PERIM is not None:
        df = df[df["perim"] >= MIN_PERIM]
    if MAX_PERIM is not None:
        df = df[df["perim"] <= MAX_PERIM]
    return df


def _load_scalar_cols(parquet_path: str, cols: List[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(parquet_path, columns=cols)
    except Exception:
        df = pd.read_parquet(parquet_path)
        keep = [c for c in cols if c in df.columns]
        return df[keep]


def _collect_png_points() -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, object]]:
    """
    Returns (areas, perims, counts_dict, meta_used) for PNG.
    meta_used includes the chosen thresholds per run when argmax policy is used.
    """
    areas, perims = [], []
    counts = {"runs_considered": 0, "scanned_rows": 0, "kept_rows": 0, "runs_skipped_no_meta": 0}
    meta_used: Dict[str, object] = {}

    run_tags = _discover_png_runs()
    for run_tag in run_tags:
        if not _within_any_window(run_tag, TIME_WINDOWS):
            continue

        run_dir = os.path.join(PNG_PER_CLOUD_ROOT, run_tag)
        shard_paths = sorted(glob.iglob(os.path.join(run_dir, "cloud_metrics.part*.parquet")))
        if not shard_paths:
            continue

        chosen_thr: Optional[float] = None
        if THRESHOLD_POLICY == "argmax_clouds":
            chosen_thr = _choose_argmax_threshold(run_tag)
            if chosen_thr is None and ON_MISSING_METADATA == "skip_run":
                counts["runs_skipped_no_meta"] += 1
                continue
            meta_used[run_tag] = {"policy": "argmax_clouds", "chosen_thr": chosen_thr, "on_missing": ON_MISSING_METADATA}
        else:
            # "all" policy
            meta_used[run_tag] = {"policy": "all", "whitelist": THRESHOLDS_WHITELIST}

        kept_in_run = 0
        scanned_in_run = 0

        for pq in shard_paths:
            df = _load_scalar_cols(pq, ["area", "perim", "threshold"])
            scanned_in_run += len(df)
            if df.empty:
                continue

            df = _apply_common_filters(df)
            if df.empty:
                continue

            # Threshold filtering
            if THRESHOLD_POLICY == "argmax_clouds":
                if chosen_thr is not None and "threshold" in df.columns:
                    df = df[np.isclose(df["threshold"].astype(float), float(chosen_thr), rtol=RTOL, atol=ATOL)]
            else:  # "all"
                if THRESHOLDS_WHITELIST is not None and "threshold" in df.columns:
                    df = df[df["threshold"].isin(THRESHOLDS_WHITELIST)]

            if df.empty:
                continue

            areas.append(df["area"].to_numpy(dtype=float, copy=False))
            perims.append(df["perim"].to_numpy(dtype=float, copy=False))
            kept_in_run += len(df)

        if kept_in_run > 0:
            counts["runs_considered"] += 1
            counts["scanned_rows"] += scanned_in_run
            counts["kept_rows"] += kept_in_run

    if counts["kept_rows"] == 0:
        return np.array([]), np.array([]), counts, meta_used

    return np.concatenate(areas).astype(float), np.concatenate(perims).astype(float), counts, meta_used


def _collect_sp_points() -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    areas, perims = [], []
    counts = {"files_considered": 0, "scanned_rows": 0, "kept_rows": 0}

    for pq in _discover_sp_parquets():
        df = _load_scalar_cols(pq, ["area", "perim", "p_val"])
        counts["files_considered"] += 1
        counts["scanned_rows"] += len(df)
        if df.empty:
            continue
        df = _apply_common_filters(df)
        if df.empty:
            continue
        if P_VALS is not None and "p_val" in df.columns:
            df = df[df["p_val"].isin(P_VALS)]
        if df.empty:
            continue
        areas.append(df["area"].to_numpy(dtype=float, copy=False))
        perims.append(df["perim"].to_numpy(dtype=float, copy=False))
        counts["kept_rows"] += len(df)

    if counts["kept_rows"] == 0:
        return np.array([]), np.array([]), counts

    return np.concatenate(areas).astype(float), np.concatenate(perims).astype(float), counts


def _ols_fit(logA: np.ndarray, logP: np.ndarray) -> Tuple[float, float, float, float]:
    """
    OLS on y=logP, x=logA. Returns (m, b, stderr_m, R2).
    """
    n = logA.size
    if n < 3:
        raise ValueError("Not enough points for OLS.")
    xm, ym = logA.mean(), logP.mean()
    Sxx = np.sum((logA - xm) ** 2)
    Sxy = np.sum((logA - xm) * (logP - ym))
    m = Sxy / Sxx
    b = ym - m * xm
    yhat = m * logA + b
    SS_res = float(np.sum((logP - yhat) ** 2))
    SS_tot = float(np.sum((logP - ym) ** 2))
    R2 = 1.0 - SS_res / SS_tot if SS_tot > 0 else float("nan")
    sigma2 = SS_res / (n - 2)
    stderr_m = math.sqrt(sigma2 / Sxx)
    return m, b, stderr_m, R2


def _plot(area: np.ndarray, perim: np.ndarray, m: float, b: float, title: str, subtitle: str, out_png: str):
    import numpy as np, math, textwrap
    import matplotlib.pyplot as plt

    # Clean data for log scales
    mask = np.isfinite(area) & np.isfinite(perim) & (area > 0) & (perim > 0)
    area, perim = area[mask], perim[mask]
    if area.size < 3:
        raise ValueError("Not enough finite/positive points for log plot.")

    # Slightly larger figure + higher DPI
    fig, ax = plt.subplots(figsize=(9, 7))

    # Scatter and red best-fit line
    ax.scatter(area, perim, s=SCATTER_S, alpha=SCATTER_ALPHA)
    a_min, a_max = float(np.min(area)), float(np.max(area))
    a_line = np.logspace(np.log10(a_min), np.log10(a_max), 200)
    c = math.exp(b)
    p_line = c * np.power(a_line, m)
    ax.plot(a_line, p_line, linewidth=2, color="red", zorder=3)

    # Axes and formatting
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Area (A)"); ax.set_ylabel("Perimeter (P)")
    ax.set_title(title, pad=10)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

    # Prepare wrapped subtitle (avoid spillover)
    wrapped_subtitle = "\n".join(textwrap.wrap(subtitle, width=110))

    # Reserve bottom margin for the footer
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    # Caption/footer (outside axes, centered)
    fig.text(
        0.5, 0.04,
        wrapped_subtitle,
        ha="center", va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor="white", alpha=0.95, linewidth=0.6)
    )

    # Save high-res figure
    fig.savefig(out_png, dpi=300)
    plt.close(fig)




def main():
    _assert_source()
    _ensure_outdir()

    if SOURCE == "png":
        area, perim, counts, meta_used = _collect_png_points()
        mode_descr = f'png[{THRESHOLD_POLICY}]'
    else:
        area, perim, counts = _collect_sp_points()
        meta_used = {}
        mode_descr = 'siteperc'

    if area.size < 3:
        print("No points after filtering; nothing to fit.")
        print("Counts:", counts)
        if meta_used:
            print("Meta used:", meta_used)
        return

    logA, logP = np.log(area), np.log(perim)
    m, b, stderr_m, R2 = _ols_fit(logA, logP)
    D = 2.0 * m
    stderr_D = 2.0 * stderr_m

    # Compose labels
    filt_bits = [
        f"{MIN_AREA} ≤ area ≤ {MAX_AREA}",
        f"MIN_PERIM={MIN_PERIM}" if MIN_PERIM is not None else None,
        f"MAX_PERIM={MAX_PERIM}" if MAX_PERIM is not None else None,
    ]
    if SOURCE == "png":
        if TIME_WINDOWS:
            filt_bits.append(f"time_windows={TIME_WINDOWS}")
        if THRESHOLD_POLICY == "all":
            if THRESHOLDS_WHITELIST is None:
                filt_bits.append("thresholds=ALL")
            else:
                filt_bits.append(f"thresholds_whitelist={THRESHOLDS_WHITELIST}")
        else:
            filt_bits.append("threshold=argmax(num_clouds_by_threshold)")
            filt_bits.append(f"on_missing_metadata={ON_MISSING_METADATA}")
    else:
        filt_bits.append(f"p_vals={P_VALS if P_VALS is not None else 'ALL'}")

    filt_bits = [x for x in filt_bits if x]
    title = f"Perimeter vs Area (log–log) — {SOURCE.upper()}"
    subtitle = (
        f"N = {area.size}  |  slope m = {m:.4f} ± {stderr_m:.4f}  |  "
        f"D = 2m = {D:.4f} ± {stderr_D:.4f}  |  R² = {R2:.4f}\n"
        f"Mode: {mode_descr}  |  Filters: " + "; ".join(filt_bits)
    )

    # File paths
    plot_png = os.path.join(OUTPUT_DIR, f"{SUFFIX_TAG}__perim_vs_area_loglog.png")
    metrics_json = os.path.join(OUTPUT_DIR, f"{SUFFIX_TAG}__metrics.json")

    # Plot and save
    _plot(area, perim, m, b, title, subtitle, plot_png)

    metrics = {
        "source": SOURCE,
        "suffix_tag": SUFFIX_TAG,
        "mode_descr": mode_descr,
        "counts": counts,
        "filters": {
            "min_area": MIN_AREA, "max_area": MAX_AREA,
            "min_perim": MIN_PERIM, "max_perim": MAX_PERIM,
            "time_windows": TIME_WINDOWS if SOURCE == "png" else None,
            "threshold_policy": THRESHOLD_POLICY if SOURCE == "png" else None,
            "thresholds_whitelist": THRESHOLDS_WHITELIST if (SOURCE == "png" and THRESHOLD_POLICY=="all") else None,
            "on_missing_metadata": ON_MISSING_METADATA if (SOURCE == "png" and THRESHOLD_POLICY=="argmax_clouds") else None,
            "p_vals": P_VALS if SOURCE == "siteperc" else None,
        },
        "fit": {
            "method": "OLS on logP ~ m*logA + b",
            "m": float(m),
            "stderr_m": float(stderr_m),
            "D": float(D),
            "stderr_D": float(stderr_D),
            "R2": float(R2),
            "N": int(area.size),
        },
        "meta_used": meta_used if SOURCE == "png" else {},
        "outputs": {
            "plot_png": plot_png,
            "metrics_json": metrics_json
        },
        "notes": "Strict source separation; thresholds via filters; no binning; 1-sigma errors.",
    }

    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # Console summary for quick readout
    print("-" * 60)
    print(" Fractal Dimension (log–log OLS) — strict source")
    print("-" * 60)
    print(f"Saved plot:    {plot_png}")
    print(f"Saved metrics: {metrics_json}")
    print("Fit:", metrics["fit"])
    print("Counts:", counts)
    if meta_used:
        print("Meta used:", meta_used)

# -------------------- CLI overrides + entrypoint --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute fractal dimension from perimeter–area data.")
    parser.add_argument("--source", choices=["png", "siteperc"], help="Data source: png or siteperc.")
    parser.add_argument("--threshold_policy", choices=["all", "argmax_clouds"],
                        help="For PNG only: threshold selection policy.")
    parser.add_argument("--time_window",
                        help="Single time window 'HH:MM-HH:MM' or 'all' (for all times).")
    parser.add_argument("--suffix_tag", help="Suffix tag for output filenames (required).")
    parser.add_argument("--p_vals", help="Site-perc p-values, comma-separated or 'all'.")
    args = parser.parse_args()

    # Apply overrides
    if args.source:
        SOURCE = args.source
    if args.threshold_policy:
        THRESHOLD_POLICY = args.threshold_policy
    if args.time_window:
        if args.time_window.lower() == "all":
            TIME_WINDOWS = []  # no filter = all times
        else:
            start, end = args.time_window.split("-")
            TIME_WINDOWS = [(start.strip(), end.strip())]
    if args.suffix_tag:
        SUFFIX_TAG = args.suffix_tag
    if args.p_vals:
        if args.p_vals.lower() == "all":
            P_VALS = None
        else:
            P_VALS = [float(x) for x in args.p_vals.split(",")]

    main()
