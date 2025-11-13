# Reads data from a csv with raw perims and areas and plots log-log scatter and fit for the two. 
# Sources data from path provided as CLI arg 
# Saves plot to output path provided as CLI arg

import numpy as np 
from scipy import optimize, stats
from typing import Callable
import sklearn.metrics
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Sample run command:
# python -m clouds.data_processing_scripts.analysis.paper_plotters.fd_plotter -i path/to/input.csv -o path/to/output.png


#@nb.njit(parallel=True)
def _logbinning_core(unsorted_x: np.ndarray, unsorted_y: np.ndarray, num_bins: int, error_type: str = 'SEM') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Define outputs
    centers = np.zeros(num_bins)
    errs = np.zeros(num_bins)
    out = np.zeros(num_bins)

    unsorted_y = unsorted_y[unsorted_x > 0]
    unsorted_x = unsorted_x[unsorted_x > 0]

    idxs = np.argsort(unsorted_x)

    # Organize by first index
    x = unsorted_x[idxs]
    y = unsorted_y[idxs]

    logmax = np.log10(x[-1])
    logmin = np.log10(x[0])

    edges = np.logspace(logmin, logmax, num_bins + 1)
    edgeidxs = np.zeros(num_bins + 1)

    for i in range(num_bins + 1):
        tmp = np.abs(x - edges[i])
        # Find minimimum from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
        edgeidxs[i] = tmp.argmin()

    # Get centers
    dx = (logmax-logmin)/num_bins
    centers = np.logspace(logmin + dx, logmax - dx, num_bins)

    # Get means
    for i in range(num_bins):
        st = int(edgeidxs[i])
        en = int(edgeidxs[i + 1])
        
        # Add 1 to take into account when start and end are same index
        en = en + int(st == en)
        vals = y[st:en]
        out[i] = np.mean(vals)
        if error_type == 'SEM':
            # SEM = std(X) / sqrt(N). N = en - st.
            errs[i] = np.std(vals) / np.sqrt(en - st)
        else:
            # Standard error = std(X)
            errs[i] = np.std(vals)
        
    return centers, out, errs


def logbinning(unsorted_x: np.ndarray, unsorted_y: np.ndarray, num_bins: int, error_type: str = 'SEM', ci: float = 0.68) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Logarithmically bin input data
    
    Generally, prefer to use standard error of the mean (SEM) because the error bar should describe how much the average value would change when binning data together on a logarithmic bin
    
    Report 95% CI using Z-scores
    
    Parameters
    ----------
    unsorted_x : np.ndarray
        X-values of data
    unsorted_y : np.ndarray
        Y-values of data
    num_bins : int
        Number of logarithmic bins to place data into
    error_type : str, optional
        How to compute the errors in each bin
        Options:
            'SEM' (default)
                -- Standard error of the mean
            {str} 
                -- Any other string uses standard deviation
    ci : float, optional
        Confidence interval as a fraction of 100
        Defaults to 0.68, which corresponds to 68% confidence interval

    Returns
    -------
    centers : np.ndarray
        Bin X-values
    out : np.ndarray
        Bin Y-values
    errs : np.ndarray
        Errors on bin Y-values multiplied by Z-score
    """    
    centers, out, errs = _logbinning_core(unsorted_x, unsorted_y, num_bins, error_type=error_type)
    z = np.sqrt(2) * stats.norm.ppf((1 + ci) / 2)

    return centers, out, errs * z
 




def fit(xdata: list | np.ndarray, ydata: list | np.ndarray, func: Callable[[np.ndarray, float, float], np.ndarray] = lambda x, p0, p1: p0 * x + p1, xmin: float = None, xmax: float = None, sigma: float = None, ci: float = 0.95, test: str = 'r_squared', print_chi2: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Perform a least-squares fit to the input data using the specified function.

    Parameters
    ----------
    xdata : list or np.ndarray of log binned data
    ydata : list or np.ndarray of log binned data
    func : Callable, optional
        Function to do least-squares fitting to
        Should just be a simple two-parameter line
        Defaults to a linear function: lambda x, p0, p1: p0 * x + p1
    xmin : float, optional
        Defaults to None, which sets xmin to the minimum value of xdata
    xmax : float, optional
        Defaults to None, which sets xmax to the maximum value of xdata
        If xmin > xmax, raises ValueError
    sigma : float, optional
        Defaults to None, which sets sigma to an array of ones with the same length as xdata and the fit is unweighted
        If sigma is provided, it should be an array of the same length as xdata
        If sigma is provided, it is used to weight the fit
    ci : float, optional
        Confidence interval as a fraction of 1
        Defaults to 0.95, which corresponds to a 95% confidence interval
    test : str, optional
        Statistical test to perform on the fit
        Options:
            'r_squared' (DEFAULT): Calculate the R-squared value of the fit
            'adjusted_r_squared' : Calculate the adjusted R-squared value of the fit
            'chi_squared' : Calculate the chi-squared value of the fit   
            'adjusted_chi_squared' : Calculate the adjusted chi-squared value of the fit
    print_chi2 : bool, optional
        If True, prints a message about the chi-squared test result
        Defaults to True

    Returns
    -------
    p1 : np.ndarray
        Fitted parameters of the function
    err : np.ndarray
        Errors on the fitted parameters, scaled by the Z-score for the specified confidence interval
    rsq : float
        R-squared value of the fit if test is 'r_squared' or 'adjusted_r_squared'
        Chi-squared value of the fit if test is 'chi_squared' or 'adjusted_chi_squared'
    """
    if xmin is None:
        xmin = min(xdata)
    if xmax is None:
        xmax = max(xdata)

    if xmin > xmax:
        raise ValueError('xmin is greater than xmax.')

    if test not in ['r_squared', 'adjusted_r_squared', 'chi_squared', 'adjusted_chi_squared']:
        raise ValueError('Input test must be one of: r_squared, adjusted_r_squared, chi_squared, or adjusted_chi_squared.')

    xdata = np.array(xdata)
    ydata = np.array(ydata)

    filt_idxs = ~np.isnan(ydata) * ~np.isnan(xdata) * (xdata > xmin) * (xdata < xmax)
    x = xdata[filt_idxs]
    y = ydata[filt_idxs]

    z = stats.norm.ppf((1 + ci) / 2)  # Get Z-score (should be 1.96 for 95% CI)

    if sigma is not None:  # If sigma is provided, use it to weight the fit
        sigma = sigma[filt_idxs]
        p1, pcov = optimize.curve_fit(func, x, y, sigma=sigma, absolute_sigma=True)
    else:  # If sigma is not provided, use an array of ones
        sigma = np.ones(len(xdata))
        sigma = sigma[filt_idxs]
        p1, pcov = optimize.curve_fit(func, x, y)

    # Standard error on parameters is sqrt of diagonal elements.
    # Multiply by Z-score to get CIs assuming normally distributed parameters.
    err = z * np.diag(pcov) ** 0.5
    yguess = func(x, *p1)
    weights = 1 / (sigma ** 2)  # Calculate the R^2 of the fit using standard weighting 1 / Var(x)

    if test == 'r_squared' or test == 'adjusted_r_squared':
        rsq = sklearn.metrics.r2_score(y, yguess, sample_weight=weights)

        if test == 'adjusted_r_squared':  # If adjusted, adjust using the normal definition.
            k = len(p1)
            n = len(y)
            rsq = 1 - (1 - rsq) * ((n - 1) / (n - k - 1))

        return p1, err, rsq

    elif test == 'chi_squared':
        # Note that chi2 is usually not a good choice for interpreting nonlinear fits.
        # This is because a scaling factor on y can change the reported chi2.
        # See https://arxiv.org/pdf/1012.3754.pdf
        chi_squared = np.sum((y - yguess) ** 2)

        pval = 1 - stats.chi2.cdf(chi_squared, len(y) - len(p1))

        if pval <= 1 - ci:
            message = "Chi squared test suggests model does not fit data with p = %.2e less than pcrit = %.2e" % (pval, 1 - ci)
        else:
            message = "Chi squared test cannot reject null hypothesis that data are drawn from the same distribution with p = %.2e greater than pcrit = %.2e" % (pval, 1 - ci)

        if print_chi2:
            print(message)

        return p1, err, pval

    elif test == 'adjusted_chi_squared':
        # Should be not too different from 1 for a reasonable fit.
        # But, chi squared can be easily messed up by a scaling factor.
        adjusted_chi_squared = np.sum((y - yguess) ** 2 / (len(y) - len(p1)))

        return p1, err, adjusted_chi_squared

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
 
def main():
    ap = argparse.ArgumentParser(description="Plot fractal dimension data from CSV file.")
    ap.add_argument("-i", "--input_csv", type=str, help="Path to input CSV file containing perimeter and area data.")
    ap.add_argument("-o", "--output_plot", type=str, help="Path to save the output plot.")
    args = ap.parse_args()

    # Load data from CSV
    data = pd.read_csv(args.input_csv)
    xdata = data['area_px'].values 
    ydata = data['perim_px'].values 

    # log binning the data
    xdata, ydata, yerr = logbinning(xdata, ydata, num_bins=20, error_type='SEM', ci=0.95)

    xlogdata = np.log10(xdata)
    ylogdata = np.log10(ydata)

    # Call the fitting function
    p1, err, rsq = fit(xlogdata, ylogdata, xmin=3)

    # Create scatter plot on log log scale
    plt.figure(figsize=(8, 6))
    plt.scatter(xdata, ydata, label='Data', color='blue', alpha=0.5)

    # Plot the fitted line

    # make line to plot 
    x_fit, y_fit = linemaker(slope=p1[0], intercept=[1, 10**p1[1]], xmin=min(xdata), xmax=max(xdata))
    plt.loglog(x_fit, y_fit, label='Fitted line', color='red')

    # Add error data to legend
    plt.legend(title=f'Fit parameters:\nSlope: {p1[0]:.4f} ± {err[0]:.4f}\nIntercept: {p1[1]:.4f} ± {err[1]:.4f}\nR²: {rsq:.4f}')

    plt.xlabel('Perimeter')
    plt.ylabel('Area')
    plt.title('Fractal Dimension Fit')
    plt.savefig(args.output_plot)
    plt.close()

if __name__ == "__main__":
    main()