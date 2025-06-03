"""
Created Jun 01 2025
Updated Jun 02 2025

Computes the diagonal correlation function for a 2D correlated field
"""
import numpy as np
import matplotlib.pyplot as plt
from clouds_helpers import linemaker, generate_2d_correlated_field


def compute_spatial_diagonal_correlation(field):
    L = field.shape[0]
    max_lag = L // 2
    corr = np.zeros(max_lag)

    # Subtract mean and normalize field to unit variance
    field = (field - np.mean(field)) / np.std(field)

    for lag in range(1, max_lag):
        products = []
        for i in range(L - lag):
            for j in range(L - lag):
                a = field[i, j]
                b = field[i + lag, j + lag]
                products.append(a * b)
        corr[lag] = np.mean(products)
    return corr[1:]


def estimate_correlations(L=211, gamma_list=[0.4, 0.8, 1.2, 1.6], n_samples=50):
    lag_axis = np.arange(1, L // 2)
    results = {}
    for gamma_val in gamma_list:
        print(f"Computing for γ = {gamma_val}")
        acc_corr = np.zeros_like(lag_axis, dtype=float)
        for seed in range(n_samples):
            field = generate_2d_correlated_field(L, gamma_val, unit_normalize=False, seed=seed)
            corr = compute_spatial_diagonal_correlation(field)
            acc_corr += corr
        acc_corr /= n_samples
        results[gamma_val] = acc_corr
    return lag_axis, results


def plot_diagonal_correlations(lags, corr_dict):
    plt.figure(figsize=(7, 5))
    for gamma, corr in sorted(corr_dict.items()):
        plt.plot(lags, corr, marker='.', label=f"γ = {gamma}")
    x, y = linemaker(-1.4, [5.2, 0.25], 1.5, 15)
    plt.plot(x, y, color='k', linestyle='dashed', linewidth=3, label='gamma = 1.4')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Lag ℓ (diagonal)")
    plt.ylabel("C(ℓ)")
    plt.title("Diagonal Correlation Function (Reproducing Figure 3)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()


lags, corr_results = estimate_correlations(
    L=200,
    gamma_list=[1.4],
    n_samples=25
)

plot_diagonal_correlations(lags, corr_results)
