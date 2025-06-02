import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, kv
from clouds_helpers import linemaker


def compute_2d_spectral_density(q, gamma_exp):
    beta = (gamma_exp - 2) / 2
    q = np.where(q == 0, 1e-10, q)
    prefactor = (2 * np.pi) / gamma(beta + 1)
    S_q = prefactor * (q / 2)**beta * kv(beta, q)
    return np.nan_to_num(np.real(S_q), nan=0.0, posinf=0.0, neginf=0.0)


def generate_correlated_field(L, gamma_exp, seed=None):
    if seed is not None:
        np.random.seed(seed)
    kx = np.fft.fftfreq(L).reshape(-1, 1)
    ky = np.fft.fftfreq(L).reshape(1, -1)
    q = np.sqrt(kx**2 + ky**2)
    S_q = compute_2d_spectral_density(q, gamma_exp)
    noise = np.random.normal(0, 1, (L, L)) + 1j * np.random.normal(0, 1, (L, L))
    hq = np.fft.fftn(noise) * np.sqrt(S_q)
    field = np.real(np.fft.ifftn(hq))
    return field


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
            field = generate_correlated_field(L, gamma_val, seed)
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
