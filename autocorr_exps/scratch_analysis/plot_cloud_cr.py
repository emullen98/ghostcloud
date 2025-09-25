#!/usr/bin/env python3
# plot_cloud_cr.py
import argparse
import glob
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import matplotlib.pyplot as plt

def load_percloud_dir(percloud_run_dir: str) -> pa.Table:
    parts = sorted(glob.glob(f"{percloud_run_dir}/cr.part*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts found in: {percloud_run_dir}")
    dataset = ds.dataset(parts, format="parquet")
    table = dataset.to_table()
    required = {"cloud_idx", "perim", "area", "cr"}
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"Missing required columns in parquet: {missing}")
    return table

def table_to_numpy_rows(table: pa.Table):
    cols = {name: table[name].to_pylist() for name in table.column_names}
    rows = []
    n = len(cols["cloud_idx"])
    for i in range(n):
        rows.append({
            "cloud_idx": cols["cloud_idx"][i],
            "area": cols["area"][i],
            "perim": cols["perim"][i],
            "cr": np.asarray(cols["cr"][i], dtype=float),
            "cr_bnd": None if cols.get("cr_bnd", [None]*n)[i] is None
                      else np.asarray(cols["cr_bnd"][i], dtype=float)
        })
    return rows

def pick_row(rows, cloud_idx=None, topk=5):
    if cloud_idx is not None:
        for r in rows:
            if r["cloud_idx"] == cloud_idx:
                return r
        raise ValueError(f"cloud_idx {cloud_idx} not found in dataset.")
    rows_sorted = sorted(rows, key=lambda r: r["area"], reverse=True)
    subset = rows_sorted[:max(1, topk)]
    return np.random.choice(subset)

def _safe_loglog_series(y: np.ndarray):
    r = np.arange(len(y))
    m = (y > 0) & (r > 0)  # exclude r=0 for log–log
    return r[m], y[m]

def _safe_semilogy_series(y: np.ndarray):
    r = np.arange(len(y))
    m = (y > 0)           # r=0 is fine on semi-log (x linear)
    return r[m], y[m]

def plot_cr(row, outprefix="cloud_cr", show_title=True):
    cr = np.asarray(row["cr"], dtype=float)
    cr_bnd = row["cr_bnd"]
    r = np.arange(len(cr))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ---- Linear ----
    axes[0].plot(r, cr, label="C(r)", linewidth=1.5)
    if cr_bnd is not None:
        axes[0].plot(r, cr_bnd, label="C_bnd(r)", linewidth=1.5)
    axes[0].set_xlabel("r (pixels)")
    axes[0].set_ylabel("C")
    axes[0].set_title("Linear")
    axes[0].legend()

    # ---- Log–log ----
    r1, y1 = _safe_loglog_series(cr)
    if len(y1):
        axes[1].loglog(r1, y1, label="C(r)", linewidth=1.5)
    if cr_bnd is not None:
        r2, y2 = _safe_loglog_series(cr_bnd)
        if len(y2):
            axes[1].loglog(r2, y2, label="C_bnd(r)", linewidth=1.5)
    axes[1].set_xlabel("r (pixels)")
    axes[1].set_ylabel("C")
    axes[1].set_title("Log–log")
    axes[1].legend()

    # ---- Semi-log (y) ----
    # with y axis rescaled to show 10**-4 to 10**0 well
    r3, y3 = _safe_semilogy_series(cr)
    if len(y3):
        axes[2].semilogy(r3, y3, label="C(r)", linewidth=1.5)
    if cr_bnd is not None:
        r4, y4 = _safe_semilogy_series(cr_bnd)
        if len(y4):
            axes[2].semilogy(r4, y4, label="C_bnd(r)", linewidth=1.5)
    axes[2].set_xlabel("r (pixels)")
    axes[2].set_ylabel("C")
    axes[2].set_title("Semi-log (y)")
    axes[2].legend()
    axes[2].set_ylim(1e-4, 1.1)

    if show_title:
        fig.suptitle(
            f"cloud_idx={row['cloud_idx']} | area={row['area']} | perim={row['perim']}",
            y=1.05, fontsize=11
        )
    plt.tight_layout()

    outpath = f"{outprefix}_cloud{row['cloud_idx']}.png"
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved plot -> {outpath}")

def main():
    ap = argparse.ArgumentParser(
        description="Plot C(r) and C_bnd(r) (if present) for a large cloud from a lattice's per_cloud parts."
    )
    ap.add_argument("--percloud-run-dir", required=True,
                    help="Path to per_cloud/<run_tag> (contains cr.part*.parquet)")
    ap.add_argument("--cloud-idx", type=int, default=None,
                    help="Specific cloud_idx to plot (optional)")
    ap.add_argument("--topk", type=int, default=5,
                    help="If no cloud_idx, pick randomly among top-k largest")
    ap.add_argument("--outprefix", type=str, default="cloud_cr",
                    help="Output PNG prefix")
    args = ap.parse_args()

    print(f"[INFO] Loading per-cloud parts from: {args.percloud_run_dir}")
    table = load_percloud_dir(args.percloud_run_dir)
    print(f"[INFO] Loaded rows: {table.num_rows:,}")

    rows = table_to_numpy_rows(table)
    row = pick_row(rows, cloud_idx=args.cloud_idx, topk=args.topk)
    print(f"[INFO] Picked cloud_idx={row['cloud_idx']} area={row['area']:,} perim={row['perim']:,}")
    print(f"[INFO] Series: len(C)={len(row['cr'])}"
          + (f", len(C_bnd)={len(row['cr_bnd'])}" if row['cr_bnd'] is not None else ", C_bnd missing"))

    plot_cr(row, outprefix=args.outprefix)

if __name__ == "__main__":
    main()
