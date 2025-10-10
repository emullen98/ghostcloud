#!/usr/bin/env python3
"""
plot_percloud_rprof.py

Read per-cloud boundary radial profiles from Parquet parts and make
plots for individual clouds.

Selection:
  - Choose the top-K largest clouds (by --sort-by area|perim)
  - Optional explicit selection via --cloud-ids (overrides top-k)
  - Optional min/max area filters before top-k

Metrics (per cloud):
  - Unnormalized aggregated boundary counts per bin (rp_counts vs r)
  - Normalized pdf over r (rp_pdf vs r) [∑ pdf·Δr = 1]
  - Ring-corrected f(r) = counts / (2π r Δr)  [computed even if not saved]

Each of the above is saved in:
  - linear y
  - semilogy (log y)
  - log–log (log x & log y)

Output:
  - PNGs under --outdir (default: <per_cloud_dir>/plots/<run_tag>/)
"""

import os, json, glob, argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


# ---------------------------
# IO helpers
# ---------------------------
def load_meta(path: str) -> dict:
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def load_parquet_rows(per_cloud_dir: str):
    """
    Yield rows from all *.parquet files in per_cloud_dir.

    Uses Scalar.as_py() to check nulls (portable across pyarrow versions).
    """
    import glob
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np
    import os

    paths = sorted(glob.glob(os.path.join(per_cloud_dir, "*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {per_cloud_dir}")

    wanted = ["cloud_idx","area","perim","p_val","rp_r","rp_counts","rp_pdf","rp_f_ring"]

    for path in paths:
        tbl: pa.Table = pq.read_table(path)
        cols = {name: (tbl[name] if name in tbl.column_names else None) for name in wanted}
        n = tbl.num_rows

        for i in range(n):
            def _scalar(name, cast=None):
                col = cols.get(name)
                if col is None:
                    return None
                cell = col[i]                # pyarrow Scalar
                val = cell.as_py()           # -> None if null
                if val is None:
                    return None
                return cast(val) if cast else val

            def _list(name, dtype=float):
                col = cols.get(name)
                if col is None:
                    return None
                cell = col[i]                # ListScalar
                val = cell.as_py()           # -> list[...] or None
                if val is None:
                    return None
                return np.asarray(val, dtype=dtype)

            yield {
                "cloud_idx": _scalar("cloud_idx", int),
                "area":      _scalar("area", int),
                "perim":     _scalar("perim", int),
                "p_val":     _scalar("p_val", float),
                "rp_r":      _list("rp_r"),
                "rp_counts": _list("rp_counts"),
                "rp_pdf":    _list("rp_pdf"),
                "rp_f_ring": _list("rp_f_ring"),
            }



# ---------------------------
# Plot helpers
# ---------------------------
def save_line_plot(x, y, xlabel, ylabel, title, outfile, logx=False, logy=False):
    # mask zeros/negatives for logs to avoid warnings
    m = np.ones_like(y, dtype=bool)
    if logx: m &= (x > 0)
    if logy: m &= (y > 0)
    x2, y2 = x[m], y[m]

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(x2, y2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


# ---------------------------
# Selection & plotting
# ---------------------------
def select_clouds(rows, topk, sort_by, min_area, max_area, explicit_ids):
    """Return a list of rows selected for plotting."""
    rows = [r for r in rows if (r["rp_r"] is not None and r["rp_counts"] is not None)]
    if min_area is not None:
        rows = [r for r in rows if r["area"] is not None and r["area"] >= min_area]
    if max_area is not None:
        rows = [r for r in rows if r["area"] is not None and r["area"] <= max_area]

    if explicit_ids:
        keep = {int(x) for x in explicit_ids}
        return [r for r in rows if r["cloud_idx"] in keep]

    key = (lambda r: r["area"]) if sort_by == "area" else (lambda r: r["perim"])
    rows.sort(key=lambda r: key(r) if key(r) is not None else -1, reverse=True)
    return rows[:topk]


def compute_fring(r, counts, delta_from_meta):
    """Compute ring-corrected series f(r) = counts / (2π r Δr)."""
    if delta_from_meta is None:
        # infer Δr from r-centers
        diffs = np.diff(r)
        delta = float(np.median(diffs)) if diffs.size else 1.0
    else:
        delta = float(delta_from_meta)
    ring_len = 2.0 * np.pi * r * delta
    f = np.zeros_like(counts, dtype=float)
    m = ring_len > 0
    f[m] = counts[m] / ring_len[m]
    return f, delta


def plot_one_cloud(row, outdir, tag_prefix, meta):
    cloud_id = row["cloud_idx"]
    area     = row["area"]
    perim    = row["perim"]
    r        = row["rp_r"]
    counts   = row["rp_counts"]
    pdf      = row["rp_pdf"]

    # ring-corrected: use saved if present; else compute from counts + Δr
    if row["rp_f_ring"] is not None:
        fring = row["rp_f_ring"]
        delta = meta.get("bin_width")
        # (if meta missing, it's fine; just for titles)
    else:
        fring, delta = compute_fring(r, counts, meta.get("bin_width"))

    # Title suffix
    bits = [
        f"id={cloud_id}",
        f"area={area}",
        f"perim={perim}",
        f"Δr={meta.get('bin_width', delta)}",
        f"center={meta.get('center_method','?')}",
        f"p={meta.get('p','?')}",
        f"seed={meta.get('seed','?')}",
    ]
    tag = f"{tag_prefix} | " + " ".join(str(b) for b in bits)

    # Filenames
    base = os.path.join(outdir, f"cloud{cloud_id:06d}_A{area}_P{perim}")
    outs = {
        "counts_linear":  base + "_counts_linear.png",
        "counts_semilog": base + "_counts_semilogy.png",
        "counts_loglog":  base + "_counts_loglog.png",
        "pdf_linear":     base + "_pdf_linear.png",
        "pdf_semilog":    base + "_pdf_semilogy.png",
        "pdf_loglog":     base + "_pdf_loglog.png",
        "fr_lin":         base + "_fring_linear.png",
        "fr_semilog":     base + "_fring_semilogy.png",
        "fr_loglog":      base + "_fring_loglog.png",
    }

    # Plots
    save_line_plot(r, counts, "r (pixels)", "Boundary counts per bin",
                   f"Counts vs r\n{tag}", outs["counts_linear"])
    save_line_plot(r, counts, "r (pixels)", "Boundary counts per bin",
                   f"Counts vs r (semilogy)\n{tag}", outs["counts_semilog"], logy=True)
    save_line_plot(r, counts, "r (pixels)", "Boundary counts per bin",
                   f"Counts vs r (log–log)\n{tag}", outs["counts_loglog"], logx=True, logy=True)

    save_line_plot(r, pdf, "r (pixels)", "PDF over r (∑ pdf·Δr = 1)",
                   f"PDF vs r\n{tag}", outs["pdf_linear"])
    save_line_plot(r, pdf, "r (pixels)", "PDF over r (∑ pdf·Δr = 1)",
                   f"PDF vs r (semilogy)\n{tag}", outs["pdf_semilog"], logy=True)
    save_line_plot(r, pdf, "r (pixels)", "PDF over r (∑ pdf·Δr = 1)",
                   f"PDF vs r (log–log)\n{tag}", outs["pdf_loglog"], logx=True, logy=True)

    save_line_plot(r, fring, "r (pixels)", "f(r) = counts / (2π r Δr)",
                   f"Ring-corrected f(r)\n{tag}", outs["fr_lin"])
    save_line_plot(r, fring, "r (pixels)", "f(r) = counts / (2π r Δr)",
                   f"Ring-corrected f(r) (semilogy)\n{tag}", outs["fr_semilog"], logy=True)
    save_line_plot(r, fring, "r (pixels)", "f(r) = counts / (2π r Δr)",
                   f"Ring-corrected f(r) (log–log)\n{tag}", outs["fr_loglog"], logx=True, logy=True)

    return list(outs.values())


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Per-cloud boundary radial profile plots (top-k largest).")
    ap.add_argument("--per-cloud-dir", required=True,
                    help="Directory containing per-cloud Parquet parts (e.g., .../per_cloud/<run_tag>/)")
    ap.add_argument("--run-tag", required=True,
                    help="Run tag for titles / output path labeling (e.g., expA_internal_W4000_...).")
    ap.add_argument("--meta-json", default=None,
                    help="Optional metadata JSON to annotate titles and get Δr if needed.")
    ap.add_argument("--outdir", default=None,
                    help="Output directory for plots. Default: <per_cloud_dir>/plots/<run_tag>/")
    ap.add_argument("--topk", type=int, default=12,
                    help="Number of largest clouds to plot (ignored if --cloud-ids provided).")
    ap.add_argument("--sort-by", choices=["area","perim"], default="area",
                    help="Criterion for 'largest' selection (default: area).")
    ap.add_argument("--min-area", type=int, default=None,
                    help="Optional minimum area filter before selecting top-k.")
    ap.add_argument("--max-area", type=int, default=None,
                    help="Optional maximum area filter before selecting top-k.")
    ap.add_argument("--cloud-ids", type=int, nargs="*",
                    help="Explicit cloud indices to plot; overrides top-k selection if given.")
    args = ap.parse_args()

    meta = load_meta(args.meta_json)
    tag_prefix = args.run_tag

    outdir = args.outdir or os.path.join(args.per_cloud_dir, "plots", args.run_tag)
    os.makedirs(outdir, exist_ok=True)

    # Load rows, select
    rows = list(load_parquet_rows(args.per_cloud_dir))
    selected = select_clouds(
        rows,
        topk=args.topk,
        sort_by=args.sort_by,
        min_area=args.min_area,
        max_area=args.max_area,
        explicit_ids=args.cloud_ids,
    )

    if not selected:
        raise SystemExit("No clouds selected for plotting (check filters / inputs).")

    print(f"[INFO] plotting {len(selected)} clouds -> {outdir}")
    for r in selected:
        outs = plot_one_cloud(r, outdir, tag_prefix, meta)
        for p in outs:
            print(" -", p)


if __name__ == "__main__":
    main()
