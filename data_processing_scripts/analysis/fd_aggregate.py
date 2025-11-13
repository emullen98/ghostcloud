#!/usr/bin/env python3
"""
fd_aggregate.py — Single-run Fractal Dimension (CSV + fit PNG(s) + diagnostics)

Behavior:
  - READ config (common schema, 3-root paths).
  - FILTER rows (area/perim; plus PNG/siteperc selectors incl. thresholds/p_vals).
  - COLLECT (area, perim_raw, perim_hull, perim_accessible) tuples
    (area > 0, at least one perim > 0).
  - For each perimeter metric:
      * If its series is identical to an earlier metric's series, record as alias_of
        (no extra fit/plot).
      * Otherwise, FIT OLS: log P = m * log A + b (natural logs), with optional fit_range.
  - WRITE:
      fd_{TAG}_raw_areas_perims.csv
          (area_px, perim_raw_edge, perim_hull_edge, perim_accessible_edge)
      fd_{TAG}_fit.png                (primary metric, usually perim_raw)
      fd_{TAG}_fit_{metric}.png       (additional unique metrics)
      fd_{TAG}_diagnostics.json       (with "fits" per metric and "fit" alias to primary)
      resolved_config.yaml
      fd_{TAG}_run_info.json

Notes:
  - Perimeter connectivity in diagnostics currently assumed 4 (fixed) for backward compatibility.
  - Serialized; no parallelism.
"""

from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from utils.analysis_utils import (
        ConfigError,
        load_yaml, validate_config, build_tag, expand_artifacts,
        write_resolved_config, write_run_info,
        ensure_dir, apply_common_filters, read_parquet_cols,
        discover_png_runs, discover_sp_parquets, within_any_window,
        choose_argmax_threshold, png_meta_path,
    )
except Exception:
    from clouds.utils.analysis_utils import *  # type: ignore


# ---------- Helpers ----------------------------------------------------------

def _nan_equal_series(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Compare two 1D arrays with NaNs allowed.
    Returns True if:
      - same shape
      - NaN pattern identical
      - finite entries equal within allclose tolerance.
    """
    if a.shape != b.shape:
        return False
    a_finite = np.isfinite(a)
    b_finite = np.isfinite(b)
    if not np.array_equal(a_finite, b_finite):
        return False
    if not np.any(a_finite):
        return True
    return np.allclose(a[a_finite], b[b_finite], rtol=1e-10, atol=1e-12)


# ---------- PNG / SP row iters (yield area, perim_raw, perim_hull, perim_accessible) ----------

def _iter_png_rows(cfg: Dict[str, Any]) -> Iterable[Tuple[float, float, float, float]]:
    """
    Yield (area, perim_raw, perim_hull, perim_accessible) from PNG per-cloud shards
    after filters and threshold policy.

    Assumes columns:
      - area
      - perim_raw
      - perim_hull
      - perim_accessible
      - threshold
    """
    paths = cfg["paths"]
    flt = cfg["filters"]
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

    for run_tag in runs:
        if windows and not within_any_window(run_tag, windows):
            continue

        chosen_thr: Optional[float] = None
        if thr_policy == "argmax_clouds":
            chosen_thr = choose_argmax_threshold(paths["png_meta_root"], run_tag)
            if chosen_thr is None and on_missing == "skip_run":
                continue

        run_dir = Path(paths["png_per_cloud_root"]) / run_tag
        for pq in sorted(run_dir.glob("cloud_metrics.part*.parquet")):
            need = [
                "area",
                "perim_raw",
                "perim_hull",
                "perim_accessible",
                "threshold",
            ]
            df = read_parquet_cols(str(pq), need)
            if df.empty:
                continue

            # Temporary 'perim' column for apply_common_filters: use perim_raw
            df["perim"] = df["perim_raw"]

            # basic filters (area + perim bounds on perim_raw)
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
                    df = df[np.isclose(
                        df["threshold"].astype(float),
                        float(chosen_thr),
                        rtol=1e-10,
                        atol=1e-12,
                    )]
                elif thr_policy == "all" and isinstance(thr_whitelist, list) and len(thr_whitelist) > 0:
                    df = df[df["threshold"].isin(thr_whitelist)]
            if df.empty:
                continue

            for _, row in df[["area", "perim_raw", "perim_hull", "perim_accessible"]].iterrows():
                a = row["area"]
                rp = row["perim_raw"]
                hp = row["perim_hull"]
                ap = row["perim_accessible"]

                # Area must be positive and finite
                try:
                    af = float(a)
                except Exception:
                    continue
                if not np.isfinite(af) or af <= 0:
                    continue

                vals: List[float] = []
                for v in (rp, hp, ap):
                    try:
                        vf = float(v)
                    except Exception:
                        vf = np.nan
                    if not np.isfinite(vf) or vf <= 0:
                        vf = np.nan
                    vals.append(vf)

                if all(np.isnan(v) for v in vals):
                    continue

                yield (af, vals[0], vals[1], vals[2])


def _iter_siteperc_rows(cfg: Dict[str, Any]) -> Iterable[Tuple[float, float, float, float]]:
    """
    Yield (area, perim_raw, perim_hull, perim_accessible) from site-perc per-cloud shards
    after filters (and optional p_vals).

    Assumes columns:
      - area
      - perim_raw
      - perim_hull
      - perim_accessible
      - p_val
    """
    paths = cfg["paths"]
    flt = cfg["filters"]

    # -------- normalize p_vals filter --------
    raw_p_filter = flt.get("p_vals", "all")
    p_vals_filter: Optional[List[float]]

    if raw_p_filter is None:
        p_vals_filter = None
    elif isinstance(raw_p_filter, str) and raw_p_filter.lower() == "all":
        p_vals_filter = None  # no filtering
    elif isinstance(raw_p_filter, list):
        vals: List[float] = []
        for v in raw_p_filter:
            try:
                vals.append(float(v))
            except Exception:
                continue
        p_vals_filter = vals or None
    else:
        try:
            v = float(raw_p_filter)
            p_vals_filter = [v]
        except Exception:
            vals: List[float] = []
            for tok in str(raw_p_filter).split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    vals.append(float(tok))
                except Exception:
                    continue
            p_vals_filter = vals or None

    # -------- iterate shards --------
    for pq in discover_sp_parquets(paths["siteperc_per_cloud_root"]):
        need = [
            "area",
            "perim_raw",
            "perim_hull",
            "perim_accessible",
            "p_val",
        ]
        df = read_parquet_cols(pq, need)
        if df.empty:
            continue

        # Temporary 'perim' column for apply_common_filters: use perim_raw
        df["perim"] = df["perim_raw"]

        # ---- area + perim filters (on perim_raw) ----
        df = apply_common_filters(
            df,
            flt.get("min_area"), flt.get("max_area"),
            flt.get("min_perim"), flt.get("max_perim"),
        )
        if df.empty:
            continue

        # ---- p_val filters (if any) ----
        if p_vals_filter is not None and "p_val" in df.columns:
            df = df[df["p_val"].isin(p_vals_filter)]
        if df.empty:
            continue

        for _, row in df[["area", "perim_raw", "perim_hull", "perim_accessible"]].iterrows():
            a = row["area"]
            rp = row["perim_raw"]
            hp = row["perim_hull"]
            ap = row["perim_accessible"]

            # Area must be positive and finite
            try:
                af = float(a)
            except Exception:
                continue
            if not np.isfinite(af) or af <= 0:
                continue

            vals: List[float] = []
            for v in (rp, hp, ap):
                try:
                    vf = float(v)
                except Exception:
                    vf = np.nan
                if not np.isfinite(vf) or vf <= 0:
                    vf = np.nan
                vals.append(vf)

            if all(np.isnan(v) for v in vals):
                continue

            yield (af, vals[0], vals[1], vals[2])


# ---------- Fitting helpers --------------------------------------------------

def _fit_range_mask(logA: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Return boolean mask selecting points used in the fit.
    If fit_range is null -> all True.
    If provided and mode=quantiles: keep within [q_lo, q_hi] on logA.
    If mode=absolute: keep min<=logA<=max where bounds are not None.
    """
    fd = cfg.get("fd", {})
    fr = fd.get("fit_range")
    if not fr:
        return np.ones_like(logA, dtype=bool)
    mode = fr.get("mode")
    mask = np.ones_like(logA, dtype=bool)
    if mode == "quantiles":
        q_lo, q_hi = fr.get("quantiles", [0.0, 1.0])
        lo, hi = np.quantile(logA, [float(q_lo), float(q_hi)])
        mask &= (logA >= lo) & (logA <= hi)
    elif mode == "absolute":
        bounds = fr.get("absolute_logA", {}) or {}
        lo = bounds.get("min", None)
        hi = bounds.get("max", None)
        if lo is not None:
            mask &= (logA >= float(lo))
        if hi is not None:
            mask &= (logA <= float(hi))
    return mask


def _ols_with_stderr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Unweighted OLS (y = m x + b) with stderr for m and b.
    Returns: m, b, stderr_m, stderr_b, rmse
    """
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    resid = y - yhat
    n = len(x)
    dof = max(n - 2, 1)
    s2 = float(np.sum(resid**2) / dof)
    X = np.column_stack([x, np.ones_like(x)])
    XtX_inv = np.linalg.inv(X.T @ X)
    var = s2 * XtX_inv
    stderr_m = math.sqrt(var[0, 0])
    stderr_b = math.sqrt(var[1, 1])
    rmse = math.sqrt(s2)
    return m, b, stderr_m, stderr_b, rmse


# ---------- Main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Fractal Dimension aggregator (CSV + fit PNG(s) + diagnostics)")
    ap.add_argument("-c", "--config", required=True, help="Path to per-run config (YAML/JSON)")
    args = ap.parse_args()

    t0 = time.time()
    cfg = validate_config(load_yaml(args.config))
    tag = build_tag(cfg)
    arts = expand_artifacts(cfg, tag)

    out_dir = Path(cfg["paths"]["output_root"])
    ensure_dir(str(out_dir))
    write_resolved_config(cfg, out_dir)

    source = cfg["source"]

    # ---- Collect raw tuples
    area_list: List[float] = []
    perim_raw_list: List[float] = []
    perim_hull_list: List[float] = []
    perim_accessible_list: List[float] = []
    n_scanned = 0

    row_iter = _iter_png_rows(cfg) if source == "png" else _iter_siteperc_rows(cfg)
    for a, rp, hp, ap in row_iter:
        n_scanned += 1
        area_list.append(float(a))
        perim_raw_list.append(float(rp) if np.isfinite(rp) else np.nan)
        perim_hull_list.append(float(hp) if np.isfinite(hp) else np.nan)
        perim_accessible_list.append(float(ap) if np.isfinite(ap) else np.nan)

    raw_df = pd.DataFrame({
        "area_px": area_list,
        "perim_raw_edge": perim_raw_list,
        "perim_hull_edge": perim_hull_list,
        "perim_accessible_edge": perim_accessible_list,
    })
    (out_dir / arts["raw_csv"]).write_text(raw_df.to_csv(index=False))

    diagnostics: Dict[str, Any] = {
        "metric": "fractal_dimension",
        "tag": tag,
        "data": {
            "n_pairs": int(len(raw_df)),
            "area_range_px": [
                float(raw_df["area_px"].min()) if len(raw_df) else None,
                float(raw_df["area_px"].max()) if len(raw_df) else None,
            ],
            "perim_range_edge": None,
        },
        "fit": None,
        "fits": {},
        "provenance": {"resolved_config_path": str(out_dir / "resolved_config.yaml")},
        "warnings": [],
    }

    status = "failed"
    if len(raw_df) >= 2 and (raw_df["area_px"] > 0).any():
        A = raw_df["area_px"].to_numpy(dtype=float)
        valid_area_mask = np.isfinite(A) & (A > 0)
        if not np.any(valid_area_mask):
            diagnostics["warnings"].append("No positive-finite area values after filters; skipping fit.")
        else:
            A = A[valid_area_mask]
            logA = np.log(A)

            metric_defs = [
                ("perim_raw_edge",        "perim_raw",        "Raw perimeter"),
                ("perim_hull_edge",       "perim_hull",       "Hull perimeter"),
                ("perim_accessible_edge", "perim_accessible", "Accessible perimeter"),
            ]

            perim_series: Dict[str, np.ndarray] = {}
            for col, label, _ in metric_defs:
                P_full = raw_df[col].to_numpy(dtype=float)
                P = P_full[valid_area_mask]
                perim_series[label] = P

            base_mask = _fit_range_mask(logA, cfg)

            uniques: List[Tuple[str, np.ndarray]] = []
            metric_alias_of: Dict[str, Optional[str]] = {}

            for col, label, _ in metric_defs:
                P = perim_series[label]
                alias_of: Optional[str] = None
                for u_label, u_P in uniques:
                    if _nan_equal_series(P, u_P):
                        alias_of = u_label
                        break
                if alias_of is None:
                    uniques.append((label, P))
                metric_alias_of[label] = alias_of

            fits: Dict[str, Any] = {}
            primary_fit_label: Optional[str] = None

            base_png_path = Path(out_dir / arts["fit_png"])

            for idx, (label, P) in enumerate(uniques):
                pretty_name = next((pname for col, lbl, pname in metric_defs if lbl == label), label)

                perim_mask = np.isfinite(P) & (P > 0)
                mask = base_mask & perim_mask

                if np.count_nonzero(mask) < 2:
                    fits[label] = {
                        "status": "no_fit",
                        "reason": "Fewer than 2 valid points for this perimeter metric after fit_range.",
                        "n_pairs_total": int(P.size),
                        "n_used_for_fit": int(np.count_nonzero(mask)),
                    }
                    diagnostics["warnings"].append(
                        f"No fit for {label}: fewer than 2 valid points after fit_range."
                    )
                    continue

                logA_fit = logA[mask]
                logP_fit = np.log(P[mask])

                if len(np.unique(logA_fit)) < 2:
                    fits[label] = {
                        "status": "no_fit",
                        "reason": "Degenerate fit range: fewer than 2 distinct logA values.",
                        "n_pairs_total": int(P.size),
                        "n_used_for_fit": int(logA_fit.shape[0]),
                    }
                    diagnostics["warnings"].append(
                        f"Degenerate fit range for {label}: fewer than 2 distinct logA values."
                    )
                    continue

                m, b, stderr_m, stderr_b, rmse = _ols_with_stderr(logA_fit, logP_fit)
                yhat = m * logA_fit + b
                resid = logP_fit - yhat
                ss_res = float(np.sum(resid**2))
                ss_tot = float(np.sum((logP_fit - float(np.mean(logP_fit))) ** 2))
                R2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)

                D = 2.0 * m
                D_stderr = 2.0 * stderr_m

                if idx == 0:
                    png_path = base_png_path
                    primary_fit_label = label
                else:
                    png_path = base_png_path.with_name(
                        f"{base_png_path.stem}_{label}{base_png_path.suffix}"
                    )

                plot_cfg = cfg.get("fd", {}).get("fit_plot", {})
                fig_w = plot_cfg.get("width", 1200) / 100.0
                fig_h = plot_cfg.get("height", 900) / 100.0
                dpi = plot_cfg.get("dpi", 150)

                plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
                plt.scatter(logA_fit, logP_fit, s=6, alpha=0.6, label="data")
                xline = np.linspace(float(np.min(logA_fit)), float(np.max(logA_fit)), 200)
                yline = m * xline + b
                plt.plot(xline, yline, linewidth=2.0, label="OLS fit", color="orange")
                if plot_cfg.get("annotate", True):
                    txt = (
                        f"{pretty_name}\n"
                        f"D = {D:.3f} ± {D_stderr:.3f}\n"
                        f"$R^2$ = {R2:.3f}\n"
                        f"n = {len(logA_fit)}"
                    )
                    ax = plt.gca()
                    ax.text(
                        0.02, 0.98, txt, transform=ax.transAxes,
                        ha="left", va="top",
                        bbox=dict(boxstyle="round", alpha=0.2, pad=0.3),
                    )
                plt.xlabel("log A")
                plt.ylabel("log P")
                plt.title(f"Fractal Dimension fit — {tag} — {pretty_name}")
                plt.legend(loc="best")
                plt.tight_layout()
                plt.savefig(png_path)
                plt.close()

                fits[label] = {
                    "status": "ok",
                    "method": "OLS",
                    "logs_base": "e",
                    "perimeter_connectivity": 4,
                    "perimeter_metric": label,
                    "range_rule": cfg.get("fd", {}).get("fit_range", None),
                    "slope_m": float(m),
                    "slope_stderr": float(stderr_m),
                    "intercept_b": float(b),
                    "intercept_stderr": float(stderr_b),
                    "D": float(D),
                    "D_stderr": float(D_stderr),
                    "R2": float(R2),
                    "rmse": float(rmse),
                    "n_pairs_total": int(P.size),
                    "n_used_for_fit": int(logA_fit.shape[0]),
                    "logA_min_used": float(np.min(logA_fit)),
                    "logA_max_used": float(np.max(logA_fit)),
                    "png_path": str(png_path),
                }

            for _, label, _ in metric_defs:
                alias = metric_alias_of[label]
                if alias is None:
                    continue
                if label not in fits or fits[label].get("status") != "ok":
                    fits[label] = {
                        "status": "alias",
                        "alias_of": alias,
                    }

            diagnostics["fits"] = fits

            if (
                primary_fit_label is not None
                and primary_fit_label in fits
                and fits[primary_fit_label].get("status") == "ok"
            ):
                diagnostics["fit"] = fits[primary_fit_label]
                status = "success"
            else:
                diagnostics["warnings"].append("No successful fit for any perimeter metric.")
    else:
        diagnostics["warnings"].append("No valid area/perim tuples after filters; skipping fit.")

    diag_path = out_dir / arts["diagnostics_json"]
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2, sort_keys=True)

    t1 = time.time()
    summary = {
        "n_scanned": int(n_scanned),
        "n_pass_filters": int(len(raw_df)),
        "n_used": int(len(raw_df)),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t1)),
        "duration_sec": float(t1 - t0),
    }
    write_run_info(
        cfg, tag, {
            "raw_csv": arts.get("raw_csv"),
            "diagnostics_json": arts.get("diagnostics_json"),
            "fit_png": arts.get("fit_png"),
            "run_info_json": arts.get("run_info_json"),
        },
        out_dir,
        status=status,
        summary=summary,
        diagnostics_path=str(diag_path),
    )


if __name__ == "__main__":
    main()
