#!/usr/bin/env python3
"""
validate_percloud_outputs.py

Minimal checker for per-cloud Parquet outputs produced by autocorr_datagen.py.

Usage:
  python validate_percloud_outputs.py \
    --per-cloud-dir scratch/autocorr_datagen/wk_datagen_L10000_p0.4074_seed383329928 \
    [--show-sample 3]
"""

import argparse
import glob
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _expect_schema(table: pa.Table, path: str):
    """
    Validate the new schema:
      cloud_idx:int64, perim:int64, area:int64, cr:list<double>, cr_bnd:list<double> (nullable)
    """
    sch = table.schema
    colnames = [sch[i].name for i in range(len(sch))]
    expected = ["cloud_idx", "perim", "area", "cr", "cr_bnd"]
    if colnames != expected:
        raise SystemExit(
            f"[ERR] Unexpected columns in {path}:\n"
            f"  got      : {colnames}\n"
            f"  expected : {expected}\n"
            f"Note: ensure you're using the updated writer with perim+cr_bnd."
        )

    t_idx  = sch.field("cloud_idx").type
    t_per  = sch.field("perim").type
    t_area = sch.field("area").type
    t_cr   = sch.field("cr").type
    t_crb  = sch.field("cr_bnd").type

    if str(t_idx)  != "int64":   raise SystemExit(f"[ERR] cloud_idx dtype is {t_idx}, expected int64 in {path}")
    if str(t_per)  != "int64":   raise SystemExit(f"[ERR] perim dtype is {t_per}, expected int64 in {path}")
    if str(t_area) != "int64":   raise SystemExit(f"[ERR] area dtype is {t_area}, expected int64 in {path}")
    if not pa.types.is_list(t_cr)  or str(t_cr.value_type)  != "double":
        raise SystemExit(f"[ERR] cr dtype is {t_cr}, expected list<double> in {path}")
    if not pa.types.is_list(t_crb) or str(t_crb.value_type) != "double":
        raise SystemExit(f"[ERR] cr_bnd dtype is {t_crb}, expected list<double> in {path}")
    # cr_bnd is nullable per row (Arrow columns are nullable by default).


def _list_lengths_from_chunks(list_chunks):
    """Per-row lengths for a chunked ListArray column that is non-null."""
    lens = []
    for chunk in list_chunks:
        offs = chunk.offsets.to_numpy()  # int32 offsets
        lens.extend(np.diff(offs))
    return np.asarray(lens, dtype=np.int64)


def _list_lengths_with_nulls(list_chunks):
    """
    Return (lengths, valid_mask) for a chunked ListArray that may contain nulls.

    lengths[i]   = number of items for row i (0 for null lists by Arrow convention)
    valid_mask[i]= True if row i is non-null, else False
    """
    lengths = []
    valids  = []
    for chunk in list_chunks:
        offs = chunk.offsets.to_numpy()
        lens = np.diff(offs)
        if chunk.null_count:
            valid_mask = ~chunk.is_null().to_numpy()
        else:
            valid_mask = np.ones(len(lens), dtype=bool)
        lengths.extend(lens)
        valids.extend(valid_mask)
    return np.asarray(lengths, dtype=np.int64), np.asarray(valids, dtype=bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cloud-dir", required=True,
                    help="Directory containing cr.part*.parquet for ONE lattice.")
    ap.add_argument("--show-sample", type=int, default=0,
                    help="Print up to N sample rows.")
    args = ap.parse_args()

    # 1) find parts
    parts = sorted(glob.glob(os.path.join(args.per_cloud_dir, "cr.part*.parquet")))
    if not parts:
        raise SystemExit(f"[ERR] No parquet parts found in {args.per_cloud_dir}")

    # 2) read, validate each part's schema
    tables = []
    for p in parts:
        t = pq.read_table(p)
        _expect_schema(t, p)
        tables.append(t)

    # 3) concat all parts (silence FutureWarning on newer PyArrow)
    try:
        table = pa.concat_tables(tables, promote_options="default")
    except TypeError:
        table = pa.concat_tables(tables, promote=True)  # older PyArrow
    nrows = table.num_rows
    print(f"[OK] Loaded {len(parts)} parts; total rows (clouds): {nrows}")

    # 4) columns -> numpy
    idx  = table.column("cloud_idx").to_numpy()
    per  = table.column("perim").to_numpy()
    area = table.column("area").to_numpy()

    # Lengths for cr (required, non-null)
    cr_col      = table.column("cr")
    cr_lengths  = _list_lengths_from_chunks(cr_col.chunks)
    if cr_lengths.size != nrows:
        raise SystemExit(f"[ERR] length mismatch: got {cr_lengths.size} cr rows, expected {nrows}")

    # Lengths for cr_bnd (nullable)
    crb_col              = table.column("cr_bnd")
    crb_lengths, crb_ok  = _list_lengths_with_nulls(crb_col.chunks)
    if crb_lengths.size != nrows:
        raise SystemExit(f"[ERR] length mismatch: got {crb_lengths.size} cr_bnd rows, expected {nrows}")

    # 5) basic sanity checks
    if (cr_lengths <= 0).any():
        bad = np.where(cr_lengths <= 0)[0][:5]
        raise SystemExit(f"[ERR] Found row(s) with empty cr array, e.g., indices {bad.tolist()}")

    if (area <= 0).any():
        bad = np.where(area <= 0)[0][:5]
        raise SystemExit(f"[ERR] Found row(s) with non-positive area, e.g., indices {bad.tolist()}")

    if (per <= 0).any():
        bad = np.where(per <= 0)[0][:5]
        raise SystemExit(f"[ERR] Found row(s) with non-positive perim, e.g., indices {bad.tolist()}")

    # cr_bnd: allowed to be all nulls; when present, must have length > 0
    n_crb_valid = int(crb_ok.sum())
    if n_crb_valid > 0:
        if (crb_lengths[crb_ok] <= 0).any():
            bad = np.where(crb_ok & (crb_lengths <= 0))[0][:5]
            raise SystemExit(f"[ERR] Found non-null cr_bnd with empty list, e.g., indices {bad.tolist()}")

    # 6) stats
    print(f"[OK] cr length stats     -> min: {cr_lengths.min()}, max: {cr_lengths.max()}, "
          f"median: {np.median(cr_lengths):.1f}")
    print(f"[OK] area stats          -> min: {area.min()}, max: {area.max()}, "
          f"median: {np.median(area):.1f}")
    print(f"[OK] perim stats         -> min: {per.min()}, max: {per.max()}, "
          f"median: {np.median(per):.1f}")

    n_crb_null = nrows - n_crb_valid
    print(f"[OK] cr_bnd availability -> valid rows: {n_crb_valid} / {nrows} "
          f"({100.0 * n_crb_valid / max(1, nrows):.1f}%), null rows: {n_crb_null}")
    if n_crb_valid > 0:
        print(f"[OK] cr_bnd length stats -> min: {crb_lengths[crb_ok].min()}, "
              f"max: {crb_lengths[crb_ok].max()}, "
              f"median: {np.median(crb_lengths[crb_ok]):.1f}")

    # 7) optional samples
    if args.show_sample > 0:
        print("\nSample rows:")
        to_show = min(args.show_sample, nrows)
        shown = 0

        cr_chunks  = cr_col.chunks
        crb_chunks = crb_col.chunks

        # If chunk counts align, iterate chunk-wise (fast); else fall back row-wise.
        if len(cr_chunks) == len(crb_chunks):
            for cr_chunk, crb_chunk in zip(cr_chunks, crb_chunks):
                if shown >= to_show:
                    break
                m = len(cr_chunk)
                # Precompute validity mask for cr_bnd in this chunk
                if crb_chunk.null_count:
                    crb_valid_chunk = ~crb_chunk.is_null().to_numpy()
                else:
                    crb_valid_chunk = np.ones(m, dtype=bool)

                for i in range(m):
                    if shown >= to_show:
                        break

                    cr_vals = np.asarray(cr_chunk[i].values.to_numpy())

                    if crb_valid_chunk[i]:
                        crb_vals = np.asarray(crb_chunk[i].values.to_numpy())
                        crb_info = f"len(cr_bnd)={crb_vals.size}, first3={crb_vals[:3] if crb_vals.size>=3 else crb_vals}"
                    else:
                        crb_info = "cr_bnd=None"

                    print(
                        f"  cloud_idx={int(idx[shown])}, perim={int(per[shown])}, area={int(area[shown])}, "
                        f"len(cr)={cr_vals.size}, first3={cr_vals[:3] if cr_vals.size>=3 else cr_vals}, {crb_info}"
                    )
                    shown += 1
        else:
            # Row-wise fallback (slower, but OK for tiny samples)
            for row_i in range(to_show):
                cr_cell  = table.column("cr")[row_i]
                cr_vals  = np.asarray(cr_cell.values.to_numpy())

                crb_cell = table.column("cr_bnd")[row_i]
                if crb_cell is None or (hasattr(crb_cell, "is_valid") and not crb_cell.is_valid):
                    crb_info = "cr_bnd=None"
                elif isinstance(crb_cell, pa.Scalar) and not crb_cell.is_valid:
                    crb_info = "cr_bnd=None"
                else:
                    crb_vals = np.asarray(crb_cell.values.to_numpy())
                    crb_info = f"len(cr_bnd)={crb_vals.size}, first3={crb_vals[:3] if crb_vals.size>=3 else crb_vals}"

                print(
                    f"  cloud_idx={int(idx[row_i])}, perim={int(per[row_i])}, area={int(area[row_i])}, "
                    f"len(cr)={cr_vals.size}, first3={cr_vals[:3] if cr_vals.size>=3 else cr_vals}, {crb_info}"
                )

    print("\n[OK] Per-cloud Parquet outputs look well-formed.")


if __name__ == "__main__":
    main()
