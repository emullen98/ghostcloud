#!/usr/bin/env python3
"""
analysis_utils.py — tiny, shared helpers for FD / Autocorr / BDD

Key changes (3-root model):
- paths.png_per_cloud_root       → PNG per-cloud runs root (dirs per run_tag)
- paths.png_meta_root            → PNG metadata root (run_tag_meta.json files)
- paths.siteperc_per_cloud_root  → Site-perc per-cloud runs root (dirs per lattice/config)
- paths.output_root              → Output base for the metric

Everything else stays featherweight: stdlib (+pandas), deterministic TAGs, and
a single minimal validator used by all metrics.
"""

from __future__ import annotations

import os, re, glob, json, hashlib, socket
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

import pandas as pd

# ---------------------------------------------------------------------------
# Basic path/time helpers (kept small & explicit)
# ---------------------------------------------------------------------------

_TIME_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})--(\d{2})-(\d{2})-(\d{2})")

def abs_path(p: str) -> str:
    """Expand ~ and make absolute real path."""
    return str(Path(p).expanduser().resolve())

def ensure_dir(p: str) -> None:
    """Create directory (and parents) if missing."""
    Path(p).expanduser().resolve().mkdir(parents=True, exist_ok=True)

def hm_to_min(hhmm: str) -> int:
    """Convert 'HH:MM' to minutes since midnight."""
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)

def parse_time_minutes_from_run_tag(run_tag: str) -> Optional[int]:
    """
    Extract HH:MM from a run tag like 'YYYY-MM-DD--HH-MM-SS_...' and return minutes since midnight.
    Returns None if the pattern isn't present.
    """
    m = _TIME_RE.search(run_tag)
    if not m:
        return None
    HH, MM = int(m.group(4)), int(m.group(5))
    return HH * 60 + MM

def within_any_window(run_tag: str, windows: List[Tuple[str, str]]) -> bool:
    """
    Return True iff run_tag's time falls within ANY of the given windows.
    Windows can wrap midnight (e.g., '23:00'-'01:00').
    If windows is empty, return True.
    """
    if not windows:
        return True
    minutes = parse_time_minutes_from_run_tag(run_tag)
    if minutes is None:
        return False
    for start, end in windows:
        s, e = hm_to_min(start), hm_to_min(end)
        if s <= e:
            if s <= minutes < e:
                return True
        else:
            if minutes >= s or minutes < e:
                return True
    return False

def discover_png_runs(png_per_cloud_root: str) -> List[str]:
    """
    Return sorted list of run directory basenames under the PNG per-cloud root.
    Each item is a run_tag like '2025-03-02--03-00-00'.
    """
    root = abs_path(png_per_cloud_root)
    runs = [d.name for d in Path(root).glob("*") if d.is_dir()]
    return sorted(runs)

def discover_sp_parquets(sp_per_cloud_root: str) -> List[str]:
    """
    Return sorted list of site-perc parquet files under the given root.
    Pattern: <root>/*/cloud_metrics.part*.parquet
    """
    root = abs_path(sp_per_cloud_root)
    pattern = str(Path(root) / "*" / "cloud_metrics.part*.parquet")
    return sorted(glob.iglob(pattern))

def png_meta_path(png_meta_root: str, run_tag: str) -> str:
    """Build the per-run metadata JSON path (threshold stats, etc.)."""
    return str(Path(abs_path(png_meta_root)) / f"{run_tag}_meta.json")

def choose_argmax_threshold(png_meta_root: str, run_tag: str) -> Optional[float]:
    """
    From the per-image/run metadata, return the threshold with the max # clouds.
    Tie-break: choose the smaller threshold. Returns None if missing/malformed.
    """
    meta_path = png_meta_path(png_meta_root, run_tag)
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        return None
    d = meta.get("num_clouds_by_threshold")
    if not isinstance(d, dict) or not d:
        return None
    best_thr: Optional[float] = None
    best_cnt: Optional[int] = None
    for k, v in d.items():
        try:
            thr = float(k); cnt = int(v)
        except Exception:
            continue
        if best_cnt is None or cnt > best_cnt or (cnt == best_cnt and (best_thr is None or thr < best_thr)):
            best_cnt, best_thr = cnt, thr
    return best_thr

def apply_common_filters(
    df: pd.DataFrame,
    min_area: Optional[float],
    max_area: Optional[float],
    min_perim: Optional[float],
    max_perim: Optional[float],
) -> pd.DataFrame:
    """
    Apply shared area/perimeter gates. Non-positive area/perim rows are dropped first.
    If a bound is None, it is not applied.
    Expects columns 'area' and 'perim'; returns empty DF if not found.
    """
    if "area" not in df.columns or "perim" not in df.columns:
        return df.iloc[0:0]
    df = df[(df["area"] > 0) & (df["perim"] > 0)]
    if min_area is not None:
        df = df[df["area"] >= float(min_area)]
    if max_area is not None:
        df = df[df["area"] <= float(max_area)]
    if min_perim is not None:
        df = df[df["perim"] >= float(min_perim)]
    if max_perim is not None:
        df = df[df["perim"] <= float(max_perim)]
    return df

def read_parquet_cols(path: str, cols: list) -> pd.DataFrame:
    """
    Read only requested columns if possible; if not, read all and subselect those present.
    """
    try:
        return pd.read_parquet(path, columns=cols)
    except Exception:
        df = pd.read_parquet(path)
        keep = [c for c in cols if c in df.columns]
        return df[keep]

# ---------------------------------------------------------------------------
# Minimal YAML I/O (PyYAML optional)
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = str(path)
    if p.lower().endswith(".json"):
        with open(p, "r") as f:
            return json.load(f)
    try:
        import yaml
        with open(p, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        with open(p, "r") as f:
            return json.load(f)

def dump_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    p = str(path)
    try:
        import yaml
        with open(p, "w") as f:
            yaml.safe_dump(obj, f, sort_keys=True)
    except Exception:
        with open(p, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)

# ---------------------------------------------------------------------------
# Config validation & normalization (single lightweight validator)
# ---------------------------------------------------------------------------

class ConfigError(RuntimeError):
    def __init__(self, code: str, message: str, path: Optional[str] = None):
        self.code = code
        self.path = path
        msg = f"[{code}] {message}" + (f" (at '{path}')" if path else "")
        super().__init__(msg)

_ALLOWED_METRICS = {"fractal_dimension", "autocorr", "bdd"}
_ALLOWED_SOURCES = {"png", "siteperc"}
_ALLOWED_THRESH_POL = {"all", "argmax_clouds"}
_ALLOWED_RMAX_MODE = {"auto", "fixed"}
_ALLOWED_U_SPACING = {"log", "linear"}
_ALLOWED_SPLAT = {"linear", "nearest"}
_ALLOWED_BDD_WEIGHTING = {"equal", "by_counts"}

def _require(d: Dict[str, Any], key: str, code="E_SCHEMA_VALIDATION"):
    if key not in d:
        raise ConfigError(code, f"Missing required key '{key}'", key)

def _ensure_enum(value: Any, allowed: set, path: str):
    if value not in allowed:
        raise ConfigError("E_SCHEMA_VALIDATION", f"Invalid value '{value}' — expected one of {sorted(allowed)}", path)

def _normalize_all_list(v: Any) -> Optional[list]:
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() == "all":
        return None
    if isinstance(v, list):
        return v
    return [v]

def _tz_now_date_india() -> str:
    try:
        tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    except Exception:
        tz = None
    dt = datetime.now(tz) if tz else datetime.now()
    return dt.strftime("%Y-%m-%d")

def validate_config(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize a *single-run* config for FD/AC/BDD using the 3-root path model.
    """
    cfg = json.loads(json.dumps(raw_cfg))  # deep copy via JSON (stable types)

    # required top-level
    for k in ("metric", "source", "paths", "filters", "output"):
        _require(cfg, k)
    _ensure_enum(cfg["metric"], _ALLOWED_METRICS, "metric")
    _ensure_enum(cfg["source"], _ALLOWED_SOURCES, "source")

    # paths (new, clean)
    paths = cfg["paths"]
    if not isinstance(paths, dict):
        raise ConfigError("E_SCHEMA_VALIDATION", "paths must be a mapping", "paths")
    for k in ("png_per_cloud_root", "png_meta_root", "siteperc_per_cloud_root", "output_root"):
        if k not in paths or not paths[k]:
            raise ConfigError("E_SCHEMA_VALIDATION", f"paths.{k} is required", f"paths.{k}")

    # filters
    filters = cfg["filters"]
    if not isinstance(filters, dict):
        raise ConfigError("E_SCHEMA_VALIDATION", "filters must be a mapping", "filters")

    filters["time_windows"] = _normalize_all_list(filters.get("time_windows", None))
    filters["p_vals"] = _normalize_all_list(filters.get("p_vals", None))

    tp = filters.get("threshold_policy", "argmax_clouds")
    _ensure_enum(tp, _ALLOWED_THRESH_POL, "filters.threshold_policy")
    filters["threshold_policy"] = tp
    if tp == "all":
        tlist = filters.get("thresholds_whitelist", None)
        if tlist is not None and not isinstance(tlist, list):
            raise ConfigError("E_SCHEMA_VALIDATION", "thresholds_whitelist must be a list or null", "filters.thresholds_whitelist")

    for num_key in ("min_area", "max_area", "min_perim", "max_perim"):
        if num_key in filters and filters[num_key] is not None:
            try:
                float(filters[num_key])
            except Exception:
                raise ConfigError("E_SCHEMA_VALIDATION", f"{num_key} must be numeric or null", f"filters.{num_key}")

    # metric-specific
    metric = cfg["metric"]
    if metric == "fractal_dimension":
        fd = cfg.get("fd", {})
        if not isinstance(fd, dict):
            raise ConfigError("E_SCHEMA_VALIDATION", "fd must be a mapping", "fd")
        if fd.get("logs_base", "e") != "e":
            raise ConfigError("E_SCHEMA_VALIDATION", "fd.logs_base must be 'e'", "fd.logs_base")
        if fd.get("perimeter_connectivity", 4) != 4:
            raise ConfigError("E_SCHEMA_VALIDATION", "fd.perimeter_connectivity must be 4", "fd.perimeter_connectivity")
        fit_plot = fd.get("fit_plot", {})
        if not isinstance(fit_plot, dict) or not fit_plot.get("filename"):
            raise ConfigError("E_SCHEMA_VALIDATION", "fd.fit_plot.filename is required", "fd.fit_plot.filename")

    elif metric == "autocorr":
        ac = cfg.get("autocorr", {})
        if not isinstance(ac, dict):
            raise ConfigError("E_SCHEMA_VALIDATION", "autocorr must be a mapping", "autocorr")
        mode = ac.get("r_max_mode", "auto")
        _ensure_enum(mode, _ALLOWED_RMAX_MODE, "autocorr.r_max_mode")
        if mode == "fixed" and (ac.get("r_max_fixed") in (None, 0)):
            raise ConfigError("E_SCHEMA_VALIDATION", "autocorr.r_max_fixed must be >=1 when r_max_mode=='fixed'", "autocorr.r_max_fixed")
        if "eps_den" in ac:
            try:
                float(ac["eps_den"])
            except Exception:
                raise ConfigError("E_SCHEMA_VALIDATION", "autocorr.eps_den must be numeric", "autocorr.eps_den")
        # optional knob: which ∈ {"all","bnd"}
        if "which" in ac and ac["which"] not in {"all", "bnd"}:
            raise ConfigError("E_SCHEMA_VALIDATION", "autocorr.which must be 'all' or 'bnd'", "autocorr.which")

    elif metric == "bdd":
        bdd = cfg.get("bdd", {})
        if not isinstance(bdd, dict):
            raise ConfigError("E_SCHEMA_VALIDATION", "bdd must be a mapping", "bdd")
        _ensure_enum(bdd.get("U_SPACING", "log"), _ALLOWED_U_SPACING, "bdd.U_SPACING")
        _ensure_enum(bdd.get("SPLAT", "linear"), _ALLOWED_SPLAT, "bdd.SPLAT")
        _ensure_enum(bdd.get("weighting", "equal"), _ALLOWED_BDD_WEIGHTING, "bdd.weighting")
        # ints
        for key in ("U_NBINS", "MIN_CLOUDS_PER_BIN"):
            if key in bdd:
                v = bdd[key]
                if not isinstance(v, int) or v < 1:
                    raise ConfigError("E_SCHEMA_VALIDATION", f"bdd.{key} must be int >=1", f"bdd.{key}")
        # margin
        mfrac = bdd.get("U_MARGIN_FRAC", 0.05)
        try:
            mfrac = float(mfrac)
        except Exception:
            raise ConfigError("E_SCHEMA_VALIDATION", "bdd.U_MARGIN_FRAC must be numeric", "bdd.U_MARGIN_FRAC")
        if not (0.0 <= mfrac < 1.0):
            raise ConfigError("E_SCHEMA_VALIDATION", "bdd.U_MARGIN_FRAC must be in [0,1)", "bdd.U_MARGIN_FRAC")
        bdd["U_MARGIN_FRAC"] = mfrac

    # output
    output = cfg["output"]
    if not isinstance(output, dict):
        raise ConfigError("E_SCHEMA_VALIDATION", "output must be a mapping", "output")
    if not output.get("tag_format"):
        raise ConfigError("E_SCHEMA_VALIDATION", "output.tag_format is required", "output.tag_format")

    if metric == "fractal_dimension":
        for k in ("fd_raw_csv", "fd_diagnostics_json", "fd_run_info_json"):
            if not output.get(k):
                raise ConfigError("E_SCHEMA_VALIDATION", f"output.{k} is required for FD", f"output.{k}")
    elif metric == "autocorr":
        for k in ("ac_csv", "ac_diagnostics_json", "ac_run_info_json"):
            if not output.get(k):
                raise ConfigError("E_SCHEMA_VALIDATION", f"output.{k} is required for Autocorr", f"output.{k}")
    elif metric == "bdd":
        for k in ("bdd_csv", "bdd_diagnostics_json", "bdd_run_info_json"):
            if not output.get(k):
                raise ConfigError("E_SCHEMA_VALIDATION", f"output.{k} is required for BDD", f"output.{k}")

    cfg["filters"] = filters
    return cfg

# ---------------------------------------------------------------------------
# TAG building & artifact expansion
# ---------------------------------------------------------------------------

def _semantic_subset_for_hash(cfg: Dict[str, Any]) -> Dict[str, Any]:
    metric = cfg["metric"]
    keep: Dict[str, Any] = {
        "metric": metric,
        "source": cfg.get("source"),
        "profile": cfg.get("profile"),
        "tag_hint": cfg.get("tag_hint"),
        "paths": {
            # semantic inputs only (exclude output_root to keep tag stable across machines)
            "png_per_cloud_root": cfg["paths"].get("png_per_cloud_root"),
            "png_meta_root": cfg["paths"].get("png_meta_root"),
            "siteperc_per_cloud_root": cfg["paths"].get("siteperc_per_cloud_root"),
        },
        "filters": cfg.get("filters"),
        "output": {"tag_format": cfg["output"].get("tag_format")},
    }
    if metric == "fractal_dimension":
        keep["fd"] = cfg.get("fd")
        keep["output"]["fd_raw_csv"] = cfg["output"].get("fd_raw_csv")
        keep["output"]["fd_diagnostics_json"] = cfg["output"].get("fd_diagnostics_json")
        keep["output"]["fd_run_info_json"] = cfg["output"].get("fd_run_info_json")
    elif metric == "autocorr":
        keep["autocorr"] = cfg.get("autocorr")
        keep["output"]["ac_csv"] = cfg["output"].get("ac_csv")
        keep["output"]["ac_diagnostics_json"] = cfg["output"].get("ac_diagnostics_json")
        keep["output"]["ac_run_info_json"] = cfg["output"].get("ac_run_info_json")
    elif metric == "bdd":
        keep["bdd"] = cfg.get("bdd")
        keep["output"]["bdd_csv"] = cfg["output"].get("bdd_csv")
        keep["output"]["bdd_diagnostics_json"] = cfg["output"].get("bdd_diagnostics_json")
        keep["output"]["bdd_run_info_json"] = cfg["output"].get("bdd_run_info_json")
    return keep

def _json_hash(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _format_thresh(filters: Dict[str, Any]) -> str:
    pol = filters.get("threshold_policy", "argmax_clouds")
    if pol == "argmax_clouds":
        return "argmax"
    wl = filters.get("thresholds_whitelist")
    return "all[wlist]" if (isinstance(wl, list) and len(wl) > 0) else "all"

def _format_win(filters: Dict[str, Any]) -> str:
    wins = filters.get("time_windows")
    if wins is None:
        return "all"
    if isinstance(wins, list):
        if len(wins) == 0:
            return "all"
        if len(wins) == 1 and isinstance(wins[0], str) and "-" in wins[0]:
            s, e = wins[0].split("-")
            return f"{s.replace(':','')}-{e.replace(':','')}"
        return "multi"
    return "all"

def build_tag(resolved_cfg: Dict[str, Any]) -> str:
    date_str = _tz_now_date_india()
    subset = _semantic_subset_for_hash(resolved_cfg)
    hash8 = _json_hash(subset)[:8]
    fmt = resolved_cfg["output"]["tag_format"]

    source = resolved_cfg.get("source", "na")
    thresh = _format_thresh(resolved_cfg["filters"])
    win = _format_win(resolved_cfg["filters"])

    # --- NEW BEHAVIOR: siteperc tags drop the window token -------------
    if source == "siteperc":
        return fmt.format(
            date=date_str,
            profile=resolved_cfg.get("profile", "na"),
            source=source,
            thresh=thresh,
            win="",                # omit window for siteperc
            hash8=hash8,
        )
    # ------------------------------------------------------------------

    # Default (png etc.): include both threshold and window
    return fmt.format(
        date=date_str,
        profile=resolved_cfg.get("profile", "na"),
        source=source,
        thresh=thresh,
        win=win,
        hash8=hash8,
    )


def expand_artifacts(resolved_cfg: Dict[str, Any], tag: str) -> Dict[str, str]:
    metric = resolved_cfg["metric"]
    out = resolved_cfg["output"]
    arts: Dict[str, str] = {}
    if metric == "fractal_dimension":
        arts["raw_csv"] = out["fd_raw_csv"].format(TAG=tag)
        arts["diagnostics_json"] = out["fd_diagnostics_json"].format(TAG=tag)
        arts["run_info_json"] = out["fd_run_info_json"].format(TAG=tag)
        fp = resolved_cfg.get("fd", {}).get("fit_plot", {}).get("filename") or out.get("fd_fit_png")
        if not fp:
            raise ConfigError("E_SCHEMA_VALIDATION", "fd.fit_plot.filename missing", "fd.fit_plot.filename")
        arts["fit_png"] = fp.format(TAG=tag)
    elif metric == "autocorr":
        arts["csv"] = out["ac_csv"].format(TAG=tag)
        arts["diagnostics_json"] = out["ac_diagnostics_json"].format(TAG=tag)
        arts["run_info_json"] = out["ac_run_info_json"].format(TAG=tag)
    elif metric == "bdd":
        arts["csv"] = out["bdd_csv"].format(TAG=tag)
        arts["diagnostics_json"] = out["bdd_diagnostics_json"].format(TAG=tag)
        arts["run_info_json"] = out["bdd_run_info_json"].format(TAG=tag)
    else:
        raise ConfigError("E_SCHEMA_VALIDATION", f"Unknown metric '{metric}'", "metric")
    return arts

# ---------------------------------------------------------------------------
# Run-info / resolved-config writers
# ---------------------------------------------------------------------------

def write_resolved_config(resolved_cfg: Dict[str, Any], output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    ensure_dir(str(out_dir))
    if resolved_cfg.get("output", {}).get("save_resolved_config", True):
        path = out_dir / "resolved_config.yaml"
        dump_yaml(resolved_cfg, path)
        return path
    return out_dir / "resolved_config.yaml"

def write_run_info(
    resolved_cfg: Dict[str, Any],
    tag: str,
    artifacts: Dict[str, str],
    output_dir: str | Path,
    status: str,
    summary: Dict[str, Any],
    diagnostics_path: Optional[str] = None,
) -> Path:
    out_dir = Path(output_dir)
    ensure_dir(str(out_dir))

    metric = resolved_cfg["metric"]
    if metric == "fractal_dimension":
        run_info_name = resolved_cfg["output"]["fd_run_info_json"].format(TAG=tag)
    elif metric == "autocorr":
        run_info_name = resolved_cfg["output"]["ac_run_info_json"].format(TAG=tag)
    elif metric == "bdd":
        run_info_name = resolved_cfg["output"]["bdd_run_info_json"].format(TAG=tag)
    else:
        raise ConfigError("E_SCHEMA_VALIDATION", f"Unknown metric '{metric}'", "metric")

    resolved_hash = _json_hash(_semantic_subset_for_hash(resolved_cfg))
    envelope = {
        "metric": metric,
        "tag": tag,
        "output_root": resolved_cfg["paths"]["output_root"],
        "artifacts": artifacts,
        "resolved_config": resolved_cfg,
        "resolved_config_hash": resolved_hash,
        "status": status,
        "summary": summary,
        "timestamps": {
            "started": summary.get("started"),
            "ended": summary.get("ended"),
            "duration_sec": summary.get("duration_sec"),
        },
        "provenance": {
            "git": summary.get("git", None),
            "host": socket.gethostname(),
        },
        "diagnostics_path": diagnostics_path,
    }
    path = out_dir / run_info_name
    with open(path, "w") as f:
        json.dump(envelope, f, indent=2, sort_keys=True)
    return path
