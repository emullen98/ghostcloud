#!/usr/bin/env python3
# plot_lattice_cr.py
import argparse, glob, os, sys
import numpy as np
import matplotlib.pyplot as plt

def load_series_txt(path: str) -> np.ndarray | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f]
    vals = [float(x) for x in lines if x]
    return np.array(vals, dtype=float) if len(vals) else None

def find_runs(lattice_dir: str):
    runs = {}
    for cr_path in glob.glob(os.path.join(lattice_dir, "*_Cr.txt")):
        base = os.path.basename(cr_path)
        if not base.endswith("_Cr.txt"):
            continue
        run_tag = base[:-len("_Cr.txt")]
        bnd_path = os.path.join(lattice_dir, f"{run_tag}_Cr_bnd.txt")
        runs[run_tag] = {"cr": cr_path, "cr_bnd": bnd_path if os.path.exists(bnd_path) else None}
    return runs

def _safe_loglog_xy(y: np.ndarray):
    if y is None or len(y) == 0:
        return None, None
    r = np.arange(len(y))
    m = (r > 0) & (y > 0)
    return (r[m], y[m]) if np.any(m) else (None, None)

def _safe_semilogy_xy(y: np.ndarray):
    if y is None or len(y) == 0:
        return None, None
    r = np.arange(len(y))
    m = (y > 0)
    return (r[m], y[m]) if np.any(m) else (None, None)

def plot_lattice_cr(run_tag: str, cr: np.ndarray, cr_bnd: np.ndarray | None, outprefix: str):
    r = np.arange(len(cr)) if cr is not None else None
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ---- Linear ----
    if cr is not None:
        axes[0].plot(r, cr, label="C(r)", linewidth=1.6)
    if cr_bnd is not None and len(cr_bnd):
        axes[0].plot(np.arange(len(cr_bnd)), cr_bnd, label="C_bnd(r)", linewidth=1.6)
    axes[0].set_xlabel("r (pixels)")
    axes[0].set_ylabel("C")
    axes[0].set_title("Linear")
    axes[0].legend()

    # ---- Log–log ----
    r1, y1 = _safe_loglog_xy(cr)
    if r1 is not None:
        axes[1].loglog(r1, y1, label="C(r)", linewidth=1.6)
    if cr_bnd is not None:
        r2, y2 = _safe_loglog_xy(cr_bnd)
        if r2 is not None:
            axes[1].loglog(r2, y2, label="C_bnd(r)", linewidth=1.6)
    axes[1].set_xlabel("r (pixels)")
    axes[1].set_ylabel("C")
    axes[1].set_title("Log–log")
    axes[1].legend()

    # ---- Semi-log (y) ----
    r3, y3 = _safe_semilogy_xy(cr)
    if r3 is not None:
        axes[2].semilogy(r3, y3, label="C(r)", linewidth=1.6)
    if cr_bnd is not None:
        r4, y4 = _safe_semilogy_xy(cr_bnd)
        if r4 is not None:
            axes[2].semilogy(r4, y4, label="C_bnd(r)", linewidth=1.6)
    axes[2].set_xlabel("r (pixels)")
    axes[2].set_ylabel("C")
    axes[2].set_ylim(1e-4, 1.1)
    axes[2].set_title("Semi-log (y)")
    axes[2].legend()

    fig.suptitle(run_tag, y=1.05, fontsize=11)
    plt.tight_layout()

    outpath = f"{outprefix}_{run_tag}.png"
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved plot -> {outpath}")

def main():
    ap = argparse.ArgumentParser(description="Plot lattice-level C(r) and C_bnd(r) from *_Cr.txt files.")
    ap.add_argument("--lattice-dir", required=True,
                    help="Directory containing <run_tag>_Cr.txt (and optionally <run_tag>_Cr_bnd.txt).")
    ap.add_argument("--run-tag", default=None,
                    help="Specific run_tag to plot (filename stem before _Cr.txt).")
    ap.add_argument("--outprefix", default="lattice_cr",
                    help="Output PNG filename prefix.")
    args = ap.parse_args()

    runs = find_runs(args.lattice_dir)
    if not runs:
        print(f"[ERR] No *_Cr.txt files found in {args.lattice_dir}", file=sys.stderr)
        sys.exit(1)

    chosen = args.run_tag or sorted(runs.keys())[0]
    if chosen not in runs:
        print(f"[ERR] run_tag '{args.run_tag}' not found. Available: {', '.join(sorted(runs))}", file=sys.stderr)
        sys.exit(1)

    cr = load_series_txt(runs[chosen]["cr"])
    cr_bnd = load_series_txt(runs[chosen]["cr_bnd"]) if runs[chosen]["cr_bnd"] else None

    if cr is None or len(cr) == 0:
        print(f"[ERR] {runs[chosen]['cr']} is missing or empty.", file=sys.stderr)
        sys.exit(1)

    if cr_bnd is None:
        print(f"[WARN] Boundary aggregate missing or empty for run_tag={chosen}")

    print(f"[INFO] Plotting run_tag={chosen} | len(C)={len(cr)}"
          + (f" | len(C_bnd)={len(cr_bnd)}" if cr_bnd is not None else " | C_bnd: none"))

    plot_lattice_cr(chosen, cr, cr_bnd, outprefix=args.outprefix)

if __name__ == "__main__":
    main()
