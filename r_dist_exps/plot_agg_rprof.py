#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt

"""

Sample usage:
python -m clouds.r_dist_exps.plot_agg_rprof --agg-dir scratch/expC/aggregates/expC_internal_W10000_H10000_p0.407400_seed987 --run-tag expC_internal_W10000_H10000_p0.407400_seed987

"""

def load_vector(path):
    with open(path, "r") as f:
        return np.array([float(s) for s in f.read().split() if s.strip()], dtype=float)

def load_meta(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def title_suffix(run_tag, meta):
    bits = [
        f"p={meta.get('p','?')}",
        f"seed={meta.get('seed','?')}",
        f"Δr={meta.get('bin_width','?')}",
        f"center={meta.get('center_method','?')}",
        f"N={meta.get('num_clouds','?')}",
    ]
    return f"{run_tag} | " + " ".join(bits)

def linemaker(slope: float = None, intercept: list = None, xmin: float = None, xmax: float = None, ppd: int = 40) -> tuple[np.ndarray, np.ndarray]:

    """

    Returns X and Y arrays of a power-law line
 
    Parameters

    ----------

    slope : float, optional

        Power law PDF slope

        Defautls to None, which raises an error

    intercept : list, optional

        Intercept of the line

        Formatted as [x-val, y-val]

        Defaults to None, which raises an error

    xmin : float, optional

        Minimum x-value the line will appear over

        Defaults to None, which raises an error

    xmax : float, optional

        Maximum x-value the line will appear over

        Defaults to None, which raises an error

    ppd : int, optional

        Number of log-spaced points per decade to evaluate the line at

        Defaults to 40
 
    Returns

    -------

    x_vals : np.ndarray 

        X-values of the line

    y_vals : np.ndarray 

        Y-values of the line

    """

    if slope is None or intercept is None or xmin is None or xmax is None:

        raise ValueError('Please enter slope, intercept, xmin and xmax.')
 
    log_x_intercept, log_y_intercept = np.log10(intercept[0]), np.log10(intercept[1])

    log_xmin, log_xmax = np.log10(xmin), np.log10(xmax)
 
    log_b = log_y_intercept - slope * log_x_intercept  # Calculate the y-intercept of the line on log axes
 
    x_vals = np.logspace(log_xmin, log_xmax, round(ppd * (log_xmax - log_xmin)))  # Get the x- and y-values of the line as arrays

    y_vals = (10 ** log_b) * (x_vals ** slope)
 
    return x_vals, y_vals
 

def save_line_plot(x, y, xlabel, ylabel, title, outfile,
                   logx=False, logy=False):
    # Mask zeros/negatives for log scales to avoid warnings
    m = np.ones_like(y, dtype=bool)
    if logx: m &= (x > 0)
    if logy: m &= (y > 0)
    x2, y2 = x[m], y[m]

    lin_x_1, lin_y_1 = linemaker(0.35, [1, 100], 2, 10)
    lin_x_2, lin_y_2 = linemaker(-2.7, [100, 10], 20, 100)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(x2, y2)
    ax.plot(lin_x_1, lin_y_1, linestyle='--', color='gray', label='slope=0.35')
    ax.plot(lin_x_2, lin_y_2, linestyle='--', color='red', label='slope=-2')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)



def main():
    ap = argparse.ArgumentParser(description="Plot aggregated radial profiles (counts/pdf + ring-corrected) with linear/semilogy/loglog variants.")
    ap.add_argument("--agg-dir", required=True, help="Directory containing *_rprof_{r,counts,pdf}.txt")
    ap.add_argument("--run-tag", required=True, help="Run tag prefix used in filenames")
    ap.add_argument("--meta-json", default=None, help="Optional metadata JSON (used to get Δr if computing ring-corrected)")
    args = ap.parse_args()

    # Files
    r_path      = os.path.join(args.agg_dir, f"{args.run_tag}_rprof_r.txt")
    counts_path = os.path.join(args.agg_dir, f"{args.run_tag}_rprof_counts.txt")
    pdf_path    = os.path.join(args.agg_dir, f"{args.run_tag}_rprof_pdf.txt")
    fring_path  = os.path.join(args.agg_dir, f"{args.run_tag}_rprof_fring.txt")  # may or may not exist

    # Load data
    r      = load_vector(r_path)
    counts = load_vector(counts_path)
    pdf    = load_vector(pdf_path)
    meta   = load_meta(args.meta_json)
    tag    = title_suffix(args.run_tag, meta)

    if not (len(r) == len(counts) == len(pdf)):
        print("[WARN] Length mismatch among r, counts, pdf:",
              len(r), len(counts), len(pdf))

    # If ring-corrected was not saved, compute it now from counts + Δr
    have_fring_file = os.path.exists(frings_path := fring_path)
    if have_fring_file:
        fring = load_vector(frings_path)
        if len(fring) != len(r):
            print("[WARN] ring-corrected vector length mismatch; recomputing from counts and Δr.")
            have_fring_file = False

    if not have_fring_file:
        try:
            delta = float(meta["bin_width"])
        except Exception:
            # Fallback: infer Δr from consecutive r-centers
            diffs = np.diff(r)
            delta = float(np.median(diffs)) if diffs.size else 1.0
            print(f"[INFO] bin_width missing in meta; inferred Δr={delta:g} from r-centers.")
        ring_len = 2.0 * np.pi * r * delta
        # avoid divide by zero at r≈0 (r-centers start at (k+0.5)Δr, so this should be safe)
        mask = ring_len > 0
        fring = np.zeros_like(counts, dtype=float)
        fring[mask] = counts[mask] / ring_len[mask]

    # Output filenames
    outs = {
        "counts_linear":   os.path.join(args.agg_dir, f"{args.run_tag}_counts_linear.png"),
        "counts_semilog":  os.path.join(args.agg_dir, f"{args.run_tag}_counts_semilogy.png"),
        "counts_loglog":   os.path.join(args.agg_dir, f"{args.run_tag}_counts_loglog.png"),
        "pdf_linear":      os.path.join(args.agg_dir, f"{args.run_tag}_pdf_linear.png"),
        "pdf_semilog":     os.path.join(args.agg_dir, f"{args.run_tag}_pdf_semilogy.png"),
        "pdf_loglog":      os.path.join(args.agg_dir, f"{args.run_tag}_pdf_loglog.png"),
        "fring_linear":    os.path.join(args.agg_dir, f"{args.run_tag}_fring_linear.png"),
        "fring_semilog":   os.path.join(args.agg_dir, f"{args.run_tag}_fring_semilogy.png"),
        "fring_loglog":    os.path.join(args.agg_dir, f"{args.run_tag}_fring_loglog.png"),
    }

    # Plots: COUNTS
    save_line_plot(r, counts,
                   xlabel="r (pixels)", ylabel="Aggregated boundary counts per bin",
                   title=f"Aggregated boundary counts vs r\n{tag}",
                   outfile=outs["counts_linear"], logx=False, logy=False)
    save_line_plot(r, counts,
                   xlabel="r (pixels)", ylabel="Aggregated boundary counts per bin",
                   title=f"Aggregated boundary counts vs r (semilogy)\n{tag}",
                   outfile=outs["counts_semilog"], logx=False, logy=True)
    save_line_plot(r, counts,
                   xlabel="r (pixels)", ylabel="Aggregated boundary counts per bin",
                   title=f"Aggregated boundary counts vs r (log–log)\n{tag}",
                   outfile=outs["counts_loglog"], logx=True, logy=True)

    # Plots: PDF (normalized)
    save_line_plot(r, pdf,
                   xlabel="r (pixels)", ylabel="PDF over r (∑ pdf·Δr = 1)",
                   title=f"Aggregated PDF vs r\n{tag}",
                   outfile=outs["pdf_linear"], logx=False, logy=False)
    save_line_plot(r, pdf,
                   xlabel="r (pixels)", ylabel="PDF over r (∑ pdf·Δr = 1)",
                   title=f"Aggregated PDF vs r (semilogy)\n{tag}",
                   outfile=outs["pdf_semilog"], logx=False, logy=True)
    save_line_plot(r, pdf,
                   xlabel="r (pixels)", ylabel="PDF over r (∑ pdf·Δr = 1)",
                   title=f"Aggregated PDF vs r (log–log)\n{tag}",
                   outfile=outs["pdf_loglog"], logx=True, logy=True)

    # Plots: RING-CORRECTED (f(r))
    save_line_plot(r, fring,
                   xlabel="r (pixels)", ylabel="f(r) = counts / (2π r Δr)",
                   title=f"Aggregated ring-corrected f(r)\n{tag}",
                   outfile=outs["fring_linear"], logx=False, logy=False)
    save_line_plot(r, fring,
                   xlabel="r (pixels)", ylabel="f(r) = counts / (2π r Δr)",
                   title=f"Aggregated ring-corrected f(r) (semilogy)\n{tag}",
                   outfile=outs["fring_semilog"], logx=False, logy=True)
    save_line_plot(r, fring,
                   xlabel="r (pixels)", ylabel="f(r) = counts / (2π r Δr)",
                   title=f"Aggregated ring-corrected f(r) (log–log)\n{tag}",
                   outfile=outs["fring_loglog"], logx=True, logy=True)

    print("[OK] Saved plots:")
    for k in ["counts_linear","counts_semilog","counts_loglog",
              "pdf_linear","pdf_semilog","pdf_loglog",
              "fring_linear","fring_semilog","fring_loglog"]:
        print(" -", outs[k])

if __name__ == "__main__":
    main()
    