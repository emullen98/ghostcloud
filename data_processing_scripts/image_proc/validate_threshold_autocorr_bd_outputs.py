#!/usr/bin/env python3
"""
validate_threshold_autocorr_bd_outputs.py

QA for per-cloud Parquet shards + metadata produced by:
  threshold_autocorr_bd_from_image.py

Checks:
  - required columns present (no cr_bnd in this pipeline)
  - bd_r spacing matches Δr = bd_bin_width (within tol)
  - zero-origin alignment: first bd_r ≈ 0.5*Δr (warn-only)
  - sum(bd_counts) == bd_n
  - com length-2 finite; rg_area finite and >= 0
  - optional WK: cr present if saved; num_all/den_all lengths match if saved
  - reports unique center_method and boundary_connectivity
  - metadata bd_bin_width agrees with rows (warn-only)

Exits non-zero on hard errors if --strict is set.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq


# ---------------------------
# Helpers
# ---------------------------
def _is_uniform_spacing(arr: np.ndarray, step: float, rtol=1e-6, atol=1e-9) -> bool:
    if arr.size <= 1:
        return True
    return np.allclose(np.diff(arr), step, rtol=rtol, atol=atol)

def _read_meta(per_cloud_dir: Path) -> Optional[dict]:
    run_tag = per_cloud_dir.name
    meta_path = per_cloud_dir.parent.parent / f"{run_tag}_meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            return None
    return None

def _val(col, i):
    """Arrow-safe getter: returns Python value or None for any scalar/list column."""
    if col is None:
        return None
    cell = col[i]
    # Some Arrow scalars don't have is_null(); as_py() returns None when null.
    try:
        return cell.as_py()
    except Exception:
        # Fallback: treat as missing
        return None

def _len_or_zero(x) -> int:
    try:
        return 0 if x is None else len(x)
    except Exception:
        return 0


# ---------------------------
# Core validation
# ---------------------------
def validate(per_cloud_dir: Path, strict: bool, show_sample: int) -> int:
    errs = 0
    warns = 0

    if not per_cloud_dir.exists():
        print(f"[ERR] per_cloud dir not found: {per_cloud_dir}")
        return 1

    parts = sorted(per_cloud_dir.glob("*.parquet"))
    if not parts:
        print(f"[ERR] No parquet shards found in {per_cloud_dir}")
        return 1

    meta = _read_meta(per_cloud_dir)
    meta_bw = float(meta["bd_bin_width"]) if (meta and "bd_bin_width" in meta) else None
    if meta_bw is not None:
        print(f"[INFO] meta bd_bin_width = {meta_bw}")

    required = {
        "cloud_idx", "perim", "area",
        "bd_r", "bd_counts", "bd_bin_width", "bd_n",
        "center_method", "boundary_connectivity",
        "com", "rg_area",
        # optional: rg_bnd, cr, num_all, den_all, threshold/p_val
    }

    n_rows = 0
    areas, perims = [], []
    uniq_bw = set()
    uniq_cm = set()
    uniq_bc = set()
    saw_cr = False
    saw_numden = False

    samples: List[Tuple[int, dict]] = []
    gid = 0

    for part in parts:
        pf = pq.ParquetFile(part)
        cols = set(pf.schema_arrow.names)
        miss = required - cols
        if miss:
            print(f"[ERR] Missing columns in {part.name}: {sorted(miss)}")
            errs += 1
            if strict:
                return 1

        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg)
            n = tbl.num_rows
            n_rows += n

            def col(name): return tbl[name] if name in tbl.column_names else None

            area_c = col("area")
            perim_c = col("perim")
            bd_r_c = col("bd_r")
            bd_counts_c = col("bd_counts")
            bw_c = col("bd_bin_width")
            bdn_c = col("bd_n")
            cm_c = col("center_method")
            bc_c = col("boundary_connectivity")
            com_c = col("com")
            rgA_c = col("rg_area")
            rgB_c = col("rg_bnd") if "rg_bnd" in tbl.column_names else None

            # optional WK
            cr_c = col("cr")
            na_c = col("num_all")
            da_c = col("den_all")

            for i in range(n):
                # --- geometry
                area = int(_val(area_c, i))
                perim = int(_val(perim_c, i))
                areas.append(area); perims.append(perim)

                # --- boundary histogram arrays
                bd_r = _val(bd_r_c, i)     # list or None
                bd_counts = _val(bd_counts_c, i)  # list or None
                if bd_r is None or bd_counts is None:
                    print(f"[ERR] row {gid}: bd_r/bd_counts missing")
                    errs += 1
                else:
                    r = np.asarray(bd_r, float)
                    c = np.asarray(bd_counts, float)
                    if r.ndim != 1 or c.ndim != 1 or r.size != c.size:
                        print(f"[ERR] row {gid}: bd_r/bd_counts shape mismatch")
                        errs += 1

                # --- Δr checks
                bw = float(_val(bw_c, i))
                uniq_bw.add(round(bw, 9))
                if bd_r is not None and _len_or_zero(bd_r) > 1:
                    r = np.asarray(bd_r, float)
                    if not _is_uniform_spacing(r, bw):
                        print(f"[ERR] row {gid}: bd_r spacing != bd_bin_width ({bw})")
                        errs += 1
                    if not np.isclose(r[0], 0.5 * bw, rtol=1e-6, atol=1e-6):
                        print(f"[WARN] row {gid}: first bd_r={r[0]:.6g} not ≈ 0.5*Δr={0.5*bw:.6g}")
                        warns += 1

                # --- count sum
                bdn = int(_val(bdn_c, i))
                if bd_counts is not None:
                    c = np.asarray(bd_counts, float)
                    s = int(np.rint(np.sum(c)))
                    if s != bdn:
                        print(f"[ERR] row {gid}: sum(bd_counts)={s} != bd_n={bdn}")
                        errs += 1

                # --- COM + Rg
                com = _val(com_c, i)
                if (com is None) or (len(com) != 2) or (not np.all(np.isfinite(com))):
                    print(f"[ERR] row {gid}: invalid com={com}")
                    errs += 1

                rgA = _val(rgA_c, i)
                if (rgA is None) or (not np.isfinite(rgA)) or (rgA < 0):
                    print(f"[ERR] row {gid}: invalid rg_area={rgA}")
                    errs += 1

                if rgB_c is not None:
                    rgB = _val(rgB_c, i)
                    if rgB is not None:
                        if (not np.isfinite(rgB)) or (rgB < 0):
                            print(f"[ERR] row {gid}: invalid rg_bnd={rgB}")
                            errs += 1

                # --- tags
                cm = _val(cm_c, i)
                bc = _val(bc_c, i)
                if cm: uniq_cm.add(cm)
                if bc: uniq_bc.add(bc)

                # --- optional WK checks
                cr_val = _val(cr_c, i) if cr_c is not None else None
                if cr_val is not None:
                    saw_cr = True

                na = _val(na_c, i) if na_c is not None else None
                da = _val(da_c, i) if da_c is not None else None
                if na is not None or da is not None:
                    saw_numden = True
                    if (na is None) ^ (da is None):
                        print(f"[ERR] row {gid}: num_all/den_all one is None")
                        errs += 1
                    else:
                        if len(na) != len(da):
                            print(f"[ERR] row {gid}: len(num_all)!=len(den_all)")
                            errs += 1

                # --- sample capture
                if len(samples) < show_sample:
                    r0 = (np.asarray(bd_r, float)[0] if bd_r else None)
                    bd_sum = (int(np.rint(np.sum(np.asarray(bd_counts, float)))) if bd_counts else None)
                    samples.append((
                        gid,
                        dict(area=area, perim=perim, bd_bins=_len_or_zero(bd_r),
                             bd_sum=bd_sum, bd_n=bdn, Δr=bw, r0=r0,
                             com=com, rg_area=rgA, cm=cm, bconn=bc)
                    ))

                gid += 1

    # Summary
    print("\n=== Summary ===")
    print(f"[OK] rows checked       : {n_rows}")
    if areas:
        print(f"[OK] area stats         : min={min(areas)} max={max(areas)} median={int(np.median(areas))}")
    if perims:
        print(f"[OK] perim stats        : min={min(perims)} max={max(perims)} median={int(np.median(perims))}")
    if uniq_bw:
        srt = sorted(uniq_bw)
        print(f"[OK] bd_bin_width uniq  : {srt}")
        if (meta_bw is not None) and (len(srt) == 1):
            u = srt[0]
            if not np.isclose(u, meta_bw, rtol=1e-6, atol=1e-9):
                print(f"[WARN] row Δr={u} differs from meta Δr={meta_bw}")
                warns += 1
    if uniq_cm:
        print(f"[OK] center_method uniq : {sorted(uniq_cm)}")
    if uniq_bc:
        print(f"[OK] boundary_conn uniq : {sorted(uniq_bc)}")
    print(f"[OK] has cr(all)?       : {saw_cr}")
    print(f"[OK] has num/den(all)?  : {saw_numden}")

    if samples:
        print("\n--- Sample rows ---")
        for idx, info in samples:
            print(f"row {idx}: {info}")

    if errs:
        print(f"\n[ERR] {errs} hard error(s) found.")
        return 1 if strict else 0
    if warns:
        print(f"\n[WARN] {warns} warning(s).")

    print("\n[OK] Validation complete.")
    return 0


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Validate per-cloud Parquet outputs (no cr_bnd) from threshold_autocorr_bd_from_image.py"
    )
    ap.add_argument("--per-cloud-dir", required=True,
                    help="Path to per_cloud/<run_tag>/ containing *.parquet shards.")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero on any [ERR].")
    ap.add_argument("--show-sample", type=int, default=3,
                    help="How many sample rows to print.")
    args = ap.parse_args()

    rc = validate(Path(args.per_cloud_dir), strict=args.strict, show_sample=args.show_sample)
    sys.exit(rc)


if __name__ == "__main__":
    main()
