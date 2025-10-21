#!/usr/bin/env python3
"""
Expand a master YAML into single-run YAML configs (source-aware, serialized).

Rules
-----
- PNG expands over:            threshold_policies × time_windows
- SitePerc expands over:       effective_p_vals (time_window forced to "all")
- `tag_hint` is HUMAN-FACING ONLY: kept in the config, not used in TAG.
- TAG uses your output.tag_format with fields:
    {date}, {profile}, {source}, {thresh}, {win}, {hash8}
  where:
    - PNG:      thresh = threshold_policy, win = actual window
    - SitePerc: thresh = f"p{p_val}",      win = "all"

Discovery (siteperc)
--------------------
If filters.p_vals == "all", we try to DISCOVER p values from
<paths.siteperc_per_cloud_root> by scanning immediate subdirs like:
  p=0.39, p0.4074, p-0.42
If none found, fallback to [0.4074] and warn.

Output
------
Writes to: <paths.output_root>/configs/<profile>/<metric>/
Filenames: <metric>__<source>__<threshTok>__<winTok>__<hash8>.yaml

Requires: PyYAML
"""

from __future__ import annotations
import os, sys, json, hashlib, itertools, datetime, re
from typing import Any, Dict, List, Tuple
try:
    import yaml  # pip install pyyaml
except Exception:
    print("ERROR: PyYAML is required. `pip install pyyaml`", file=sys.stderr)
    raise

# ------------------------ tiny helpers ------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def slug(s: Any) -> str:
    s = str(s)
    s = s.strip()
    s = s.replace("/", "_")
    s = s.replace(":", "-")
    s = s.replace(" ", "")
    s = s.replace(",", "_")
    s = s.replace("__", "_")
    return s

def hash8(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:8]

def today_yyyymmdd() -> str:
    return datetime.datetime.now().strftime("%Y%m%d")

def deep_copy(x: Any) -> Any:
    return json.loads(json.dumps(x))

# ------------------------ siteperc discovery ------------------------

_P_PAT = re.compile(r"p[=\-:]?(\d+(?:\.\d+)?)", re.IGNORECASE)

def discover_p_vals(siteperc_root: str) -> List[float]:
    vals: List[float] = []
    try:
        for name in os.listdir(siteperc_root):
            full = os.path.join(siteperc_root, name)
            if not os.path.isdir(full):
                continue
            m = _P_PAT.search(name)
            if m:
                try:
                    vals.append(float(m.group(1)))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    vals = sorted(set(vals))
    return vals

# ------------------------ TAG builder ------------------------

def build_tag(tag_format: str, *, profile: str, source: str, thresh_token: str, win_token: str, content_for_hash: Any) -> str:
    return tag_format.format(
        date=today_yyyymmdd(),
        profile=slug(profile),
        source=slug(source),
        thresh=slug(thresh_token),
        win=slug(win_token),
        hash8=hash8(content_for_hash),
    )

# ------------------------ validation ------------------------

def require(master: Dict[str, Any], key: str) -> Any:
    if key not in master:
        raise ValueError(f"Master missing required key: {key}")
    return master[key]

def normalize_master(master: Dict[str, Any]) -> Dict[str, Any]:
    require(master, "metrics")
    require(master, "sources")
    require(master, "paths")
    require(master, "filters")
    require(master, "output")

    if not isinstance(master["metrics"], list) or not master["metrics"]:
        raise ValueError("`metrics` must be a non-empty list.")
    if not isinstance(master["sources"], list) or not master["sources"]:
        raise ValueError("`sources` must be a non-empty list.")

    paths = master["paths"]
    for req in ["png_per_cloud_root", "png_meta_root", "siteperc_per_cloud_root", "output_root"]:
        if req not in paths or not isinstance(paths[req], str):
            raise ValueError(f"`paths.{req}` is required and must be a string.")

    filt = master["filters"]
    if "threshold_policies" not in filt or not isinstance(filt["threshold_policies"], list) or not filt["threshold_policies"]:
        raise ValueError("`filters.threshold_policies` must be a non-empty list.")
    if "time_windows" not in filt or not isinstance(filt["time_windows"], list) or not filt["time_windows"]:
        raise ValueError("`filters.time_windows` must be a non-empty list.")

    out = master["output"]
    if "tag_format" not in out or not isinstance(out["tag_format"], str):
        raise ValueError("`output.tag_format` must be a string.")

    return master

# ------------------------ single-run builder ------------------------

def make_single_run(
    master: Dict[str, Any],
    metric: str,
    source: str,
    thresh_token: str,
    win_token: str,
    pinned_filters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a single-run config matching the per-run schema the user supplied:
      metric/source/profile/tag_hint
      paths: {...}
      filters: {...}
      fd/autocorr/bdd
      output: {..., TAG}
    """
    m = master
    single: Dict[str, Any] = {}
    single["metric"]   = metric
    single["source"]   = source
    single["profile"]  = m.get("profile", "")
    single["tag_hint"] = m.get("tag_hint", "")  # human-facing only

    single["paths"]    = deep_copy(m["paths"])

    # --- NEW: metric-specific output_root -------------------------
    base_out = single["paths"].get("output_root")
    if base_out:
        single["paths"]["output_root"] = os.path.join(base_out, metric)  ### ADDED
    # ---------------------------------------------------------------

    # Merge base filters with pinned per-run fields (threshold_policy/time_window/p_vals)
    base_filters = deep_copy(m["filters"])
    base_filters.update(pinned_filters)
    single["filters"]  = base_filters

    # Pass through metric-specific sections verbatim
    if "fd" in m:       single["fd"]       = deep_copy(m["fd"])
    if "autocorr" in m: single["autocorr"] = deep_copy(m["autocorr"])
    if "bdd" in m:      single["bdd"]      = deep_copy(m["bdd"])

    # Output with resolved TAG
    out = deep_copy(m["output"])
    content_for_hash = {
        "metric": metric, "source": source,
        "profile": m.get("profile", ""),
        "filters": {
            "thresh_token": thresh_token,
            "win_token": win_token,
            "min_area": base_filters.get("min_area"),
            "max_area": base_filters.get("max_area"),
            "min_perim": base_filters.get("min_perim"),
            "max_perim": base_filters.get("max_perim"),
            "p_vals": base_filters.get("p_vals"),
        },
    }
    TAG = build_tag(
        out["tag_format"],
        profile=single["profile"],
        source=source,
        thresh_token=thresh_token,
        win_token=win_token,
        content_for_hash=content_for_hash,
    )
    out["TAG"] = TAG
    single["output"] = out
    return single


def file_name_for_config(sr: Dict[str, Any]) -> str:
    metric = sr["metric"]
    source = sr["source"]
    filt   = sr["filters"]
    # tokens for readability
    if source == "siteperc":
        t_tok = f"p-{str(filt['p_vals'][0]).replace('.', '_')}"
        w_tok = "all"
    else:
        t_tok = str(filt["threshold_policy"])
        w_tok = str(filt["time_windows"][0])
    stem = f"{metric}__{source}__{slug(t_tok)}__{slug(w_tok)}__{hash8({'m':metric,'s':source,'t':t_tok,'w':w_tok,'p':sr.get('profile','')})}"
    return f"{stem}.yaml"

# ------------------------ main expansion ------------------------

def main(argv: List[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("--master",):
        print("Usage: python expand_master.py --master MASTER.yaml", file=sys.stderr)
        return 2

    master_path = argv[1]
    with open(master_path, "r") as f:
        master = yaml.safe_load(f)

    master = normalize_master(master)

    metrics   = master["metrics"]
    sources   = master["sources"]
    tpols     = master["filters"]["threshold_policies"]
    windows   = master["filters"]["time_windows"]
    p_vals_in = master["filters"].get("p_vals", "all")

    out_root  = master["paths"]["output_root"]
    profile   = master.get("profile", "default")
    cfg_root  = os.path.join(out_root, "configs", slug(profile))
    ensure_dir(cfg_root)

    # Precompute effective p-vals if needed
    siteperc_root = master["paths"]["siteperc_per_cloud_root"]
    if p_vals_in == "all":
        p_vals_eff = discover_p_vals(siteperc_root)
        if not p_vals_eff:
            print(f"[WARN] No p-values discovered under {siteperc_root}; falling back to [0.4074]", file=sys.stderr)
            p_vals_eff = [0.4074]
    elif isinstance(p_vals_in, list):
        p_vals_eff = p_vals_in
    else:
        # string but not "all" → try to parse comma-separated like "0.39,0.4074"
        try:
            p_vals_eff = [float(x) for x in str(p_vals_in).split(",") if x.strip()]
        except Exception:
            raise ValueError("filters.p_vals must be 'all', a list, or a comma-separated string of numbers.")

    total = 0
    for metric in metrics:
        metric_dir = os.path.join(cfg_root, metric)
        ensure_dir(metric_dir)

        # PNG combos
        if "png" in sources:
            for tp in tpols:
                for win in windows:
                    pinned = {
                        "threshold_policy": tp,
                        "time_windows": [win],
                        "p_vals": "all",  # PNG ignores p_vals by design; keep schema uniform
                    }
                    single = make_single_run(master, metric, "png", thresh_token=tp, win_token=win, pinned_filters=pinned)
                    fpath  = os.path.join(metric_dir, file_name_for_config(single))
                    with open(fpath, "w") as w:
                        yaml.safe_dump(single, w, sort_keys=False)
                    print("WROTE", fpath)
                    total += 1

        # SitePerc combos
        if "siteperc" in sources:
            for pv in p_vals_eff:
                pinned = {
                    "p_vals": [pv],
                    "time_windows": ["all"],        # force "all" for siteperc
                    "threshold_policy": "argmax_clouds",  # irrelevant for siteperc; keep a sane token
                }
                single = make_single_run(master, metric, "siteperc", thresh_token=f"p{pv}", win_token="all", pinned_filters=pinned)
                fpath  = os.path.join(metric_dir, file_name_for_config(single))
                with open(fpath, "w") as w:
                    yaml.safe_dump(single, w, sort_keys=False)
                print("WROTE", fpath)
                total += 1

    print(f"Done. Wrote {total} configs under: {cfg_root}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
