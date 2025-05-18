"""
Created Sep 30 2024
Updated Sep 30 2024

(IN CLUSTER)

"""
import numpy as np
import numba as nb

ln = np.log


@nb.njit
def brent_findmin(x, blo=1.0, bhi=20.0, xtol=1e-12, rtol=8.881784197001252e-16, maxiter=100):
    """

    :param x:
    :param blo:
    :param bhi:
    :param xtol:
    :param rtol:
    :param maxiter:
    :return:
    """
    x = np.sort(x)
    xmin = x[0]
    xmax = x[-1]
    n = len(x)
    if np.sum(x) == x[0] * n:
        return np.nan
    S = np.sum(np.log(x))

    def f(alpha):
        # added edge case for alpha near 1
        # test for alpha = 1
        if alpha == 1:
            val = -ln(ln(xmax / xmin)) - S / n  # equation from Deluca & Corrall 2013, equation 12.
            if val < 0:  # sometimes the value of this function will be less than zero (i.e. for lognormally distributed data). In that case, just return a positive value because it's an error.
                return 100
            return val
        # large values of test_xmin lead to undefined behavior due to float imprecision, limit approaches -inf. with derivative +inf
        test_xmin = np.log10(xmin) * (-alpha + 1)
        if test_xmin > 100:
            return -10

        # if the tested alpha is very low, use a taylor approximation
        if alpha < 1 + 1e-7:
            y = alpha - 1
            beta = y * ln(xmax / xmin)
            gam = ln(xmax / xmin) - y * ln(xmax) * ln(xmax) + y * ln(xmin) * ln(xmin)
        else:
            beta = -xmax ** (-alpha + 1) + xmin ** (-alpha + 1)
            gam = xmax ** (-alpha + 1) * ln(xmax) - xmin ** (-alpha + 1) * ln(xmin)

        y = n / (alpha - 1) - S - n * (gam / beta)

        return y

    # hold previous, current, and blk (?) values
    xpre = blo  # previous estimate of the root
    xcur = bhi  # current estimate of the root
    xblk = np.nan  # holds value of x
    fpre = f(xpre)
    fcur = f(xcur)
    fblk = np.nan  # hold value of f(x) (?)

    # s values
    spre = 0  # previous step size
    scur = 0  # current step size
    sbis = 0  # bisect
    stry = 0

    # d values
    dpre = 0
    dblk = 0

    delta = 0  # hold the value of the error

    # edge case calculation
    if fpre == 0:
        return xpre
    if fcur == 0:
        return xcur

    # main loop
    for i in range(maxiter):
        # if fpre and fcur are both not zero and fpre has a different sign from fcur
        if (fpre != 0) * (fcur != 0) * (np.sign(fpre) != np.sign(fcur)):
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre  # put xpre and fpre into xblk and fblk, set spre = scur = xcur - xpre

        # if fblk is less than fcur, then move the bracket to (xpre, xblk)
        if abs(fblk) < abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre

        # check the bounds
        delta = 0.5 * (xtol + rtol * abs(xcur))
        sbis = 0.5 * (xblk - xcur)
        if (fcur == 0) + (abs(sbis) < delta):
            # print('Root found in %d iterations.' % i)
            return xcur

        if (abs(spre) > delta) * (abs(fcur) < abs(fpre)):
            if xpre == xblk:
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            # short step
            if (2 * abs(stry) < abs(spre)) * (2 * abs(stry) < 3 * abs(sbis) - delta):
                spre = scur
                scur = stry
            # otherwise bisect
            else:
                spre = sbis
                scur = sbis
        else:
            # otherwise bisect
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur
        if abs(scur) > delta:
            xcur = xcur + scur  # step xcur by scur
        else:
            xcur = xcur + np.sign(sbis) * delta

        fcur = f(xcur)  # another function call

    return xcur


# Updated Sep 30 2024
@nb.njit
def brent_findmin_discrete(x, blo=1.0, bhi=20.0, xtol=1e-12, rtol=8.881784197001252e-16, maxiter=100):
    """

    :param x:
    :param blo:
    :param bhi:
    :param xtol:
    :param rtol:
    :param maxiter:
    :return:
    """
    x = np.sort(x)
    xmin = x[0]
    xmax = x[-1]
    vals = np.arange(xmin, xmax + 1)
    n = len(x)
    if np.sum(x) == x[0] * n:
        return np.nan
    logvals = ln(vals)
    S = np.sum(ln(x))

    def f(alpha):
        # added edge case for alpha near 1
        # test for alpha = 1
        if alpha == 1:
            val = -ln(ln(xmax / xmin)) - np.sum(np.log(x)) / n  # equation from Deluca & Corrall 2013, equation 12.
            if val < 0:  # sometimes the value of this function will be less than zero (i.e. for lognormally distributed data). In that case, just return a positive value because it's an error.
                return 100
            return val
        # large values of test_xmin lead to undefined behavior due to float imprecision, limit approaches -inf. with derivative +inf
        test_xmin = np.log10(xmin) * (-alpha + 1)
        if test_xmin > 100:
            return -10

        # Add in approximation for low alpha if necessary.
        """
        #if the tested alpha is very low, use a taylor approximation
        if alpha < 1 + 1e-7:
            y = alpha-1
            beta = y*ln(xmax/xmin)
            gam = ln(xmax/xmin) - y*ln(xmax)*ln(xmax) + y*ln(xmin)*ln(xmin)
        else:
            beta = -xmax**(-alpha+1) + xmin**(-alpha+1)
            gam = xmax**(-alpha+1)*ln(xmax) - xmin**(-alpha+1)*ln(xmin)

        y = n/(alpha - 1) - S - n*(gam/beta)
        """
        y = - S + n * (np.sum(logvals * vals ** (-alpha))) / (np.sum(vals ** -alpha))

        return y

    # hold previous, current, and blk values
    xpre = blo  # previous estimate of the root
    xcur = bhi  # current estimate of the root
    xblk = np.nan  # holds value of x
    fpre = f(xpre)
    fcur = f(xcur)
    fblk = np.nan  # hold value of f(x) (?)

    # s values
    spre = 0  # previous step size
    scur = 0  # current step size
    sbis = 0  # bisect
    stry = 0

    # d values
    dpre = 0
    dblk = 0

    delta = 0  # hold the value of the error

    # edge case calculation
    if fpre == 0:
        return xpre
    if fcur == 0:
        return xcur

    # main loop
    for i in range(maxiter):
        # if fpre and fcur are both not zero and fpre has a different sign from fcur
        if (fpre != 0) * (fcur != 0) * (np.sign(fpre) != np.sign(fcur)):
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre  # put xpre and fpre into xblk and fblk, set spre = scur = xcur - xpre

        # if fblk is less than fcur, then move the bracket to (xpre, xblk)
        if abs(fblk) < abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre

        # check the bounds
        delta = 0.5 * (xtol + rtol * abs(xcur))
        sbis = 0.5 * (xblk - xcur)
        if (fcur == 0) + (abs(sbis) < delta):
            return xcur

        if (abs(spre) > delta) * (abs(fcur) < abs(fpre)):
            if xpre == xblk:
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            # short step
            if (2 * abs(stry) < abs(spre)) * (2 * abs(stry) < 3 * abs(sbis) - delta):
                spre = scur
                scur = stry
            # otherwise bisect
            else:
                spre = sbis
                scur = sbis
        else:
            # otherwise bisect
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur
        if abs(scur) > delta:
            xcur = xcur + scur  # step xcur by scur
        else:
            xcur = xcur + np.sign(sbis) * delta

        fcur = f(xcur)  # another function call

    return xcur


# Updated Sep 30 2024
@nb.njit
def find_pl(x, xmin, xmax=1e6):
    """
    Find the power law between xmin and xmax, assuming continuous variable.
    Parameters
    ----------
    x : array
        The values to search for a power law within.
    xmin : float
        The minimum of the scaling regime.
    xmax : float
        The maximum of the scaling regime. The default is 1e6.
    stepsize : float, optional
        Added to ensure cross-compatability with discrete find_pl. Does not do anything in this function. The default is None.

    Returns
    -------
    alpha : float
        The optimal power law exponent.
    ll : float
        The log-likelihood at the optimal alpha.
    """
    xc = x[(x >= xmin) * (x <= xmax)]

    alpha = brent_findmin(xc)

    # ETHAN SEP 30 2024: Don't need log likelihood
    # ll = pl_like(xc, xmin, xmax, alpha)[0]

    return alpha


# Updated Sep 30 2024
@nb.njit
def find_pl_discrete(x, xmin, xmax, stepsize=None):
    """
    Find the power law in a discrete variable.

    Parameters
    ----------
    x : float array
        Contains the data to find the power law in. Should be discrete (i.e. only multiples of some dx value)
    xmin : float
        The minimum of the scaling regime.
    xmax : float
        The maximum of the scaling regime.
    stepsize : float, optional
        The stepsize of the discrete variable, i.e. the timestep for durations. If None, then stepsize will be set to min(x)/100 (i.e. small enough that the variable is assumed continuous). The default is None.

    Returns
    -------
    alpha : float
        The optimal power law exponent.
    ll : float
        The log-likelihood at the optimal alpha.

    """
    if stepsize is None:
        stepsize = min(x) / 100  # if not specified, assume the stepsize is small enough that we don't need to worry.

    normx = np.rint(x / stepsize)
    normxmin = np.rint(xmin / stepsize)
    normxmax = np.rint(xmax / stepsize)

    # Remove from consideration all events that are rounded down to zero, have xmin be equal to 1.
    if normxmin == 0:
        normxmin = 1
    if normxmax == 0:
        normxmax = 1

    normx = normx[(normx >= normxmin) * (normx <= normxmax)]

    alpha = brent_findmin_discrete(normx)

    # ETHAN SEP 30 2024: Don't need log likelihood
    # ll = pl_like_discrete(normx, normxmin, normxmax, alpha)[0]

    return alpha
