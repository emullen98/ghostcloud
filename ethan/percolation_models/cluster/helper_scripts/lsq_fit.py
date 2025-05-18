"""
Created Sep 30 2024
Updated Sep 30 2024

(IN CLUSTER)

"""
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import sklearn
import sklearn.metrics


# Updated Sep 30 2024
def fit(xdata, ydata, func=lambda x, p0, p1: p0 * x + p1, xmin=None, xmax=None, sigma=None, ci=0.95, test='rsq',
        print_chi2=True):
    """

    :param xdata:
    :param ydata:
    :param func:
    :param xmin:
    :param xmax:
    :param sigma:
    :param ci:
    :param test:
    :param print_chi2:
    :return:
    """
    if xmin is None:
        xmin = min(xdata)
    if xmax is None:
        xmax = max(xdata)

    sigma_given = True

    if sigma is None:
        sigma_given = False
        sigma = np.ones(len(xdata))

    if (xmin > xmax) or (xmax < xmin):
        print("Error! Make sure xmin < xmax.")
        return -1, -1, -1

    xdata = np.array(xdata)
    ydata = np.array(ydata)

    filt = ~np.isnan(ydata) * ~np.isnan(xdata) * (xdata > xmin) * (xdata < xmax)

    x = xdata[filt]
    y = ydata[filt]
    sigma = sigma[filt]

    # Get z score (should be 1.96 for 95% CI)
    z = stats.norm.ppf((1 + ci) / 2)

    # Use * operator to unpack xp into list of inputs for function, starting with x
    if sigma_given:
        p1, pcov = curve_fit(func, x, y, sigma=sigma, absolute_sigma=True)
    else:
        p1, pcov = curve_fit(func, x, y)
    # Standard error on parameters is sqrt of diagnoal elements.
    # Multiply by z-score to get CIs assuming normally distributed parameters.
    err = z * np.diag(pcov) ** 0.5

    # Use * operator to unpack xp into a tuple for insertion into the function
    yguess = func(x, *p1)

    # Calculate the R^2 of the fit.
    # Using standard weighting 1/Var(x)
    weights = 1 / (sigma ** 2)

    # Use sklearn since it's well-documented.
    rsq = sklearn.metrics.r2_score(y, yguess, sample_weight=weights)

    # If adjusted, adjust using the normal definition.
    if test == 'adjusted_rsq':
        k = len(p1)
        n = len(y)
        rsq = 1 - (1 - rsq) * ((n - 1) / (n - k - 1))

    return p1, err, rsq


print(fit(np.arange(5), np.arange(5))[0][0]
      )