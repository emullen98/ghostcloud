#!/usr/bin/env python3
"""
Expand a master YAML into single-run YAML configs (source-aware, serialized).

Updated behavior
----------------
- Each master config now describes exactly ONE source: either "png" or "siteperc".
- A per-dataset identifier `dataset_id` is used to scope outputs:
    paths.output_root (inside single-run configs) becomes:
        <base_output_root>/<metric>/<dataset_id>
- Config files are written under:
    <paths.output_root>/configs/<profile>/<metric>/
  with filenames:
    <metric>__<source>__<dataset_id>__<threshTok>__<winTok>__<hash8>.yaml

Rules (unchanged in spirit)
---------------------------
- PNG expands over:      threshold_policies × time_windows
- SitePerc (new behavior): one config per metric per master, no p-expansion.
  We set filters.p_vals = "all" so downstream code uses all clouds in that dataset.
- `tag_hint` is HUMAN-FACING ONLY: kept in the config, not used in TAG.
- TAG uses your output.tag_format with fields:
    {date}, {profile}, {source}, {thresh}, {win}, {hash8}
  where:
    - PNG:      thresh = threshold_policy, win = actual window
    - SitePerc: thresh = "p_all",         win = "all"

Discovery (siteperc)
--------------------
The old behavior tried to DISCOVER p values from a global siteperc_per_cloud_root.
That logic is now disabled because each siteperc_per_cloud_root is already a
single (p, cl, cf, order, ...) dataset. The old code is preserved below as
commented-out blocks for reference.

Requires: PyYAML
"""

# python -m clouds.data_processing_scripts.analysis.expand_master --master clouds/data_processing_scripts/analysis/configs/orchestrator.yaml

from __future__ import annotations
import os, sys, json, hashlib, datetime, re
from typing import Any, Dict, List
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

# ------------------------ siteperc discovery (unused now) ------------------------

_P_PAT = re.compile(r"p[=\-:]?(\d+(?:\.\d+)?)", re.IGNORECASE)

def discover_p_vals(siteperc_root: str) -> List[float]:
    """
    OLD BEHAVIOR (now unused):

    Scan a *global* siteperc_per_cloud_root for immediate subdirs with p-patterns
    like 'p=0.39', 'p0.4074', 'p-0.42' and return sorted unique p-values.

    This was used when a single master covered many p-values at once. In the new
    pipeline, each master describes a single (p, cl, cf, order, ...) dataset, so
    we do NOT discover or expand over p anymore.
    """
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

    # metrics
    if not isinstance(master["metrics"], list) or not master["metrics"]:
        raise ValueError("`metrics` must be a non-empty list.")

    # sources: enforce exactly one source
    sources = master["sources"]
    if isinstance(sources, str):
        sources = [sources]
    if not isinstance(sources, list) or not sources:
        raise ValueError("`sources` must be a non-empty list or string.")
    if len(sources) != 1:
        raise ValueError("New pipeline: `sources` must contain exactly one source ('png' or 'siteperc').")
    master["sources"] = sources  # normalized to single-element list

    paths = master["paths"]
    for req in ["png_per_cloud_root", "png_meta_root", "siteperc_per_cloud_root", "output_root"]:
        if req not in paths or not isinstance(paths[req], str):
            # NOTE: some may be irrelevant for the chosen source; we keep strictness.
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

# ------------------------ dataset_id inference ------------------------

def infer_dataset_id(paths: Dict[str, Any], source: str) -> str:
    """
    Determine a dataset identifier for scoping outputs.

    Priority:
      1. paths.dataset_id (if present and non-empty)
      2. Leaf directory name of the relevant per-cloud root.
    """
    ds = paths.get("dataset_id")
    if isinstance(ds, str) and ds.strip():
        return slug(ds)

    if source == "png":
        key = "png_per_cloud_root"
    else:
        key = "siteperc_per_cloud_root"

    root = paths.get(key, "")
    if not root:
        raise ValueError(f"Cannot infer dataset_id: `paths.{key}` is missing or empty for source={source!r}.")

    leaf = os.path.basename(os.path.normpath(root))
    if not leaf:
        raise ValueError(f"Cannot infer dataset_id from `paths.{key}={root!r}`.")
    return slug(leaf)

# ------------------------ single-run builder ------------------------

def make_single_run(
    master: Dict[str, Any],
    metric: str,
    source: str,
    thresh_token: str,
    win_token: str,
    pinned_filters: Dict[str, Any],
    dataset_id: str,
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
    single["metric"]     = metric
    single["source"]     = source
    single["profile"]    = m.get("profile", "")
    single["tag_hint"]   = m.get("tag_hint", "")  # human-facing only
    single["dataset_id"] = dataset_id

    single["paths"] = deep_copy(m["paths"])

    # metric + dataset-specific output_root for the *run*
    base_out = single["paths"].get("output_root")
    if base_out:
        single["paths"]["output_root"] = os.path.join(base_out, metric, dataset_id)

    # Merge base filters with pinned per-run fields (threshold_policy/time_window/p_vals)
    base_filters = deep_copy(m["filters"])
    base_filters.update(pinned_filters)
    single["filters"] = base_filters

    # Pass through metric-specific sections verbatim
    if "fd" in m:       single["fd"]       = deep_copy(m["fd"])
    if "autocorr" in m: single["autocorr"] = deep_copy(m["autocorr"])
    if "bdd" in m:      single["bdd"]      = deep_copy(m["bdd"])

    # Output with resolved TAG
    out = deep_copy(m["output"])
    content_for_hash = {
        "metric": metric,
        "source": source,
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
    metric     = sr["metric"]
    source     = sr["source"]
    filt       = sr["filters"]
    dataset_id = sr.get("dataset_id", "")

    # tokens for readability
    if source == "siteperc":
        # New behavior: we no longer expand over p, so we don't encode numeric p here.
        t_tok = "p_all"
        w_tok = "all"
    else:
        t_tok = str(filt["threshold_policy"])
        w_tok = str((filt.get("time_windows") or ["?"])[0])

    stem = f"{metric}__{source}__{slug(dataset_id)}__{slug(t_tok)}__{slug(w_tok)}__" + hash8({
        "m": metric,
        "s": source,
        "d": dataset_id,
        "t": t_tok,
        "w": w_tok,
        "p": sr.get("profile", ""),
    })
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
    # single source, already validated
    source    = master["sources"][0]
    tpols     = master["filters"]["threshold_policies"]
    windows   = master["filters"]["time_windows"]
    p_vals_in = master["filters"].get("p_vals", "all")

    out_root  = master["paths"]["output_root"]
    profile   = master.get("profile", "default")
    cfg_root  = os.path.join(out_root, "configs", slug(profile))
    ensure_dir(cfg_root)

    # dataset identifier for this master (dataset + source)
    dataset_id = infer_dataset_id(master["paths"], source)

    # ------------------------------------------------------------------
    # OLD SITEPERC P-VAL DISCOVERY (DISABLED)
    #
    # The previous behavior expanded over p-values discovered from a
    # *global* siteperc_per_cloud_root, e.g.:
    #
    # p_vals_eff: List[float] = []
    # if source == "siteperc":
    #     siteperc_root = master["paths"]["siteperc_per_cloud_root"]
    #     if p_vals_in == "all":
    #         p_vals_eff = discover_p_vals(siteperc_root)
    #         if not p_vals_eff:
    #             print(f"[WARN] No p-values discovered under {siteperc_root}; "
    #                   f"falling back to [0.4074]", file=sys.stderr)
    #             p_vals_eff = [0.4074]
    #     elif isinstance(p_vals_in, list):
    #         p_vals_eff = p_vals_in
    #     else:
    #         try:
    #             p_vals_eff = [float(x) for x in str(p_vals_in).split(",") if x.strip()]
    #         except Exception:
    #             raise ValueError("filters.p_vals must be 'all', a list, or a "
    #                              "comma-separated string of numbers.")
    #
    # In the new pipeline each master describes a single (p, cl, cf, order, ...)
    # dataset, so we do NOT discover or expand over p anymore. Siteperc gets
    # exactly one config per metric with filters.p_vals = "all".
    # ------------------------------------------------------------------

    total = 0
    for metric in metrics:
        metric_dir = os.path.join(cfg_root, metric)
        ensure_dir(metric_dir)

        if source == "png":
            # PNG combos: threshold_policies × time_windows
            for tp in tpols:
                for win in windows:
                    pinned = {
                        "threshold_policy": tp,
                        "time_windows": [win],
                        "p_vals": "all",  # PNG ignores p_vals by design; keep schema uniform
                    }
                    single = make_single_run(
                        master,
                        metric,
                        "png",
                        thresh_token=tp,
                        win_token=win,
                        pinned_filters=pinned,
                        dataset_id=dataset_id,
                    )
                    fpath = os.path.join(metric_dir, file_name_for_config(single))
                    with open(fpath, "w") as w:
                        yaml.safe_dump(single, w, sort_keys=False)
                    print("WROTE", fpath)
                    total += 1

        elif source == "siteperc":
            # NEW SITEPERC BEHAVIOR:
            # Per-dataset siteperc root is already a single p.
            # Do NOT filter or expand on p at all; just pass "all" through so
            # downstream code uses every cloud in this dataset.
            pinned = {
                "p_vals": "all",          # sentinel: no p filter
                "time_windows": ["all"],  # still force "all" for siteperc
                "threshold_policy": "argmax_clouds",  # irrelevant; just a token
            }
            single = make_single_run(
                master,
                metric,
                "siteperc",
                thresh_token="p_all",   # for TAG/filename readability only
                win_token="all",
                pinned_filters=pinned,
                dataset_id=dataset_id,
            )
            fpath = os.path.join(metric_dir, file_name_for_config(single))
            with open(fpath, "w") as w:
                yaml.safe_dump(single, w, sort_keys=False)
            print("WROTE", fpath)
            total += 1

            # ------------------------------------------------------------------
            # OLD SITEPERC EXPANSION (DISABLED)
            #
            # elif source == "siteperc":
            #     # SitePerc combos: discovered/effective p_vals, force time_window="all"
            #     for pv in p_vals_eff:
            #         pinned = {
            #             "p_vals": [pv],
            #             "time_windows": ["all"],        # force "all" for siteperc
            #             "threshold_policy": "argmax_clouds",  # irrelevant; keep a sane token
            #         }
            #         single = make_single_run(
            #             master,
            #             metric,
            #             "siteperc",
            #             thresh_token=f"p{pv}",
            #             win_token="all",
            #             pinned_filters=pinned,
            #             dataset_id=dataset_id,
            #         )
            #         fpath = os.path.join(metric_dir, file_name_for_config(single))
            #         with open(fpath, "w") as w:
            #             yaml.safe_dump(single, w, sort_keys=False)
            #         print("WROTE", fpath)
            #         total += 1
            # ------------------------------------------------------------------

        else:
            raise ValueError(f"Unknown source: {source!r}. Expected 'png' or 'siteperc'.")

    print(f"Done. Wrote {total} configs under: {cfg_root}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
