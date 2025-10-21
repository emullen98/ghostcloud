#!/usr/bin/env python3
"""
autocorr_aggregate.py — Single-run Autocorrelation (CSV-only)

Behavior:
  - READ config (common schema, 3-root paths).
  - FILTER rows (area/perim; plus PNG/siteperc selectors).
  - USE ONLY NUMERATOR vectors from parquet:
      which == "all" -> row["num_all"], which == "bnd" -> row["num_bnd"]
  - BUILD DENOMINATOR INTERNALLY (IGNORE parquet den):
      agg_den[r] = (Σ area_i) * ring_counts_quadrant(R)[r]
  - WRITE ac_{TAG}.csv with columns: r, agg_num, agg_den, Cr
          ac_{TAG}_diagnostics.json
          resolved_config.yaml
          ac_{TAG}_run_info.json

Notes:
  - r is 0-based in CSV.
  - Serialized; no parallelism.
"""

from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from clouds.utils.analysis_utils import *  # type: ignore

# denom helper — external util (do not reimplement here)
try:
    from utils.autocorr_utils import ring_counts_quadrant
except Exception:
    from clouds.utils.autocorr_utils import ring_counts_quadrant  # type: ignore


def _iter_png_rows(cfg: Dict[str, Any], which: str) -> Iterable[Tuple[np.ndarray, float]]:
    """Yield (numerator_vector, area) from PNG runs after filters & threshold picking."""
    paths = cfg["paths"]; flt = cfg["filters"]
    runs = discover_png_runs(paths["png_per_cloud_root"])
    windows: List[Tuple[str, str]] = []
    if flt.get("time_windows"):
        for w in flt["time_windows"]:
            if isinstance(w, str) and "-" in w:
                s, e = [x.strip() for x in w.split("-")]
                windows.append((s, e))

    thr_policy = flt.get("threshold_policy", "argmax_clouds")
    thr_whitelist = flt.get("thresholds_whitelist")
    on_missing = flt.get("on_missing_metadata", "skip_run")

    num_col = "num_all" if which == "all" else "num_bnd"

    for run_tag in runs:
        if windows and not within_any_window(run_tag, windows):
            continue

        # choose threshold if needed
        chosen_thr = None
        if thr_policy == "argmax_clouds":
            chosen_thr = choose_argmax_threshold(paths["png_meta_root"], run_tag)
            if chosen_thr is None and on_missing == "skip_run":
                continue

        run_dir = Path(paths["png_per_cloud_root"]) / run_tag
        for pq in sorted(run_dir.glob("cloud_metrics.part*.parquet")):
            need = ["area", "perim", "threshold", num_col]
            df = read_parquet_cols(str(pq), need)
            if df.empty:
                continue
            df = apply_common_filters(
                df,
                flt.get("min_area"), flt.get("max_area"),
                flt.get("min_perim"), flt.get("max_perim"),
            )
            if df.empty:
                continue

            # threshold gating
            if "threshold" in df.columns:
                if thr_policy == "argmax_clouds" and chosen_thr is not None:
                    df = df[np.isclose(df["threshold"].astype(float), float(chosen_thr), rtol=1e-10, atol=1e-12)]
                elif thr_policy == "all" and isinstance(thr_whitelist, list) and len(thr_whitelist) > 0:
                    df = df[df["threshold"].isin(thr_whitelist)]
            if df.empty:
                continue

            for _, row in df[[num_col, "area"]].iterrows():
                arr = row[num_col]; area = row["area"]
                if arr is None or area is None:
                    continue
                v = np.asarray(arr, dtype=np.float64).ravel()
                if v.size == 0 or not np.all(np.isfinite(v)):
                    continue
                yield v, float(area)

def _iter_siteperc_rows(cfg: Dict[str, Any], which: str) -> Iterable[Tuple[np.ndarray, float]]:
    """Yield (numerator_vector, area) from site-perc shards after filters."""
    paths = cfg["paths"]; flt = cfg["filters"]
    num_col = "num_all" if which == "all" else "num_bnd"

    for pq in discover_sp_parquets(paths["siteperc_per_cloud_root"]):
        need = ["area", "perim", "p_val", num_col]
        df = read_parquet_cols(pq, need)
        if df.empty:
            continue
        df = apply_common_filters(
            df,
            flt.get("min_area"), flt.get("max_area"),
            flt.get("min_perim"), flt.get("max_perim"),
        )
        if df.empty:
            continue
        if flt.get("p_vals") is not None and "p_val" in df.columns:
            df = df[df["p_val"].isin(flt["p_vals"])]
        if df.empty:
            continue
        for _, row in df[[num_col, "area"]].iterrows():
            arr = row[num_col]; area = row["area"]
            if arr is None or area is None:
                continue
            v = np.asarray(arr, dtype=np.float64).ravel()
            if v.size == 0 or not np.all(np.isfinite(v)):
                continue
            yield v, float(area)

def _dynamic_add(total: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Grow-and-add for unknown-length vectors (preserves values)."""
    if total.size == 0:
        return v.astype(np.float64, copy=True)
    if v.size > total.size:
        out = np.zeros(v.size, dtype=np.float64)
        out[: total.size] = total
        out[: v.size] += v
        return out
    total[: v.size] += v
    return total

def main():
    ap = argparse.ArgumentParser(description="Autocorr aggregator (0-based r; denom=area_sum×ring_counts)")
    ap.add_argument("-c", "--config", required=True, help="Path to per-run config (YAML/JSON)")
    args = ap.parse_args()

    t0 = time.time()
    cfg = validate_config(load_yaml(args.config))
    tag = build_tag(cfg)
    arts = expand_artifacts(cfg, tag)

    out_dir = Path(cfg["paths"]["output_root"])
    ensure_dir(str(out_dir))
    write_resolved_config(cfg, out_dir)

    eps = float(cfg.get("autocorr", {}).get("eps_den", 1e-12))
    which = str(cfg.get("autocorr", {}).get("which", "all")).lower()
    if which not in {"all", "bnd"}:
        raise ConfigError("E_SCHEMA_VALIDATION", "autocorr.which must be 'all' or 'bnd'", "autocorr.which")

    source = cfg["source"]
    rows = _iter_png_rows(cfg, which) if source == "png" else _iter_siteperc_rows(cfg, which)

    n_scanned = 0
    n_used = 0
    area_sum = 0.0
    agg_num = np.array([], dtype=np.float64)

    for num_vec, area in rows:
        n_scanned += 1
        agg_num = _dynamic_add(agg_num, num_vec)
        area_sum += float(area)
        n_used += 1

    if agg_num.size == 0:
        diag = {
            "metric": "autocorr",
            "tag": tag,
            "n_scanned": int(n_scanned),
            "n_pass_filters": int(n_used),
            "n_used": int(n_used),
            "r_max_effective": 0,
            "eps_den": float(eps),
            "den_strategy": "area_sum * ring_counts_quadrant",
            "which": which,
            "warnings": ["No numerator arrays after filters; nothing to aggregate."],
            "provenance": {"resolved_config_path": str(out_dir / "resolved_config.yaml")},
        }
        with open(out_dir / arts["diagnostics_json"], "w") as f:
            json.dump(diag, f, indent=2, sort_keys=True)

        t1 = time.time()
        summary = {
            "n_scanned": int(n_scanned),
            "n_pass_filters": int(n_used),
            "n_used": int(n_used),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
            "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t1)),
            "duration_sec": float(t1 - t0),
        }
        write_run_info(cfg, tag, {
            "csv": arts.get("csv"),
            "diagnostics_json": arts.get("diagnostics_json"),
            "run_info_json": arts.get("run_info_json"),
        }, out_dir, status="failed", summary=summary, diagnostics_path=str(out_dir / arts["diagnostics_json"]))
        return

    # Build denominator via area_sum × integer ring counts (0..R-1)
    R = int(agg_num.size)
    ring_counts = ring_counts_quadrant(R)[:R].astype(np.float64)
    agg_den = area_sum * ring_counts
    Cr = agg_num / np.maximum(agg_den, eps)

    ac_df = pd.DataFrame({
        "r": np.arange(R, dtype=int),
        "agg_num": agg_num,
        "agg_den": agg_den,
        "Cr": Cr,
    })
    ac_df.to_csv(out_dir / arts["csv"], index=False)

    diag = {
        "metric": "autocorr",
        "tag": tag,
        "n_scanned": int(n_scanned),
        "n_pass_filters": int(n_used),
        "n_used": int(n_used),
        "r_max_effective": int(R),
        "eps_den": float(eps),
        "den_strategy": "area_sum * ring_counts_quadrant",
        "which": which,
        "provenance": {"resolved_config_path": str(out_dir / "resolved_config.yaml")},
        "warnings": [],
    }
    with open(out_dir / arts["diagnostics_json"], "w") as f:
        json.dump(diag, f, indent=2, sort_keys=True)

    t1 = time.time()
    summary = {
        "n_scanned": int(n_scanned),
        "n_pass_filters": int(n_used),
        "n_used": int(n_used),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t1)),
        "duration_sec": float(t1 - t0),
    }
    write_run_info(cfg, tag, {
        "csv": arts.get("csv"),
        "diagnostics_json": arts.get("diagnostics_json"),
        "run_info_json": arts.get("run_info_json"),
    }, out_dir, status="success", summary=summary, diagnostics_path=str(out_dir / arts["diagnostics_json"]))

if __name__ == "__main__":
    main()
