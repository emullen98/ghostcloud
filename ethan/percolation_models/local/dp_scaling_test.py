"""
Created Apr 10 2025
Updated Apr 10 2025

-- Creates various plots for checking that my DP implementation exhibits known scaling laws.
-- Designed to be a stand-alone script, i.e., all necessary functions are present if they aren't imported from a
package.
-- First set of functions are purely helper functions.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


def timestep(slices, prob, lx, ly):
    """
    Evolves the DP lattice by one timestep.

    Parameters
    ----------
    slices : arr
        DP lattice at one timestep.
    prob : float
        Bond probability.
    lx : int
        Number of rows in lattice.
    ly : int
        Number of columns in lattice.

    Returns
    -------
    """
    prob1 = np.random.choice(2, (ly, lx), p=[1 - prob, prob]).astype('int8')
    prob2 = np.random.choice(2, (ly, lx), p=[1 - prob, prob]).astype('int8')
    prob3 = np.random.choice(2, (ly, lx), p=[1 - prob, prob]).astype('int8')
    slice2 = np.roll(slices, shift=(0, -1), axis=(0, 1)).astype('int8')
    slice3 = np.roll(slices, shift=(-1, 0), axis=(0, 1)).astype('int8')
    slice_new = prob1 * slices + prob2 * slice2 + prob3 * slice3
    slice_new = (slice_new > 0).astype('int8')

    return slice_new


def linemaker(slope, intercept, xmin, xmax, ppd=40):
    """
    Returns X and Y arrays of a power-law line.

    Parameters
    ----------
    slope : float
        Power-law PDF exponent
    intercept : list
        Intercept of the line
        Formatted as [x-val, y-val]
    xmin : float
        Minimum x-value the line will appear over
    xmax : float
        Maximum x-value the line will appear over
    ppd : int, optional
        Number of log-spaced points per decade to evaluate the line at

    Returns
    -------
    [0] x_vals : arr
        X values of the line
    [1] y_vals : arr
        Y values of the line
    """
    if not (isinstance(slope, int) or isinstance(slope, float)) or slope == 0:
        print('Please enter the slope as a nonzero int or float')
        return 'temp'
    if not isinstance(intercept, list):
        print('Please enter an intercept of the line as a list [x, y] of ints or floats')
        return 'temp'
    for coord in intercept:
        if not (isinstance(coord, int) or isinstance(coord, float)):
            print('Please enter an intercept of the line as a list [x, y] of ints or floats')
            return 'temp'
    if not (isinstance(xmin, int) or isinstance(xmin, float)):
        print('Please enter an xmin as an int or float')
        return 'temp'
    if not (isinstance(xmax, int) or isinstance(xmax, float)):
        print('Please enter an xmax as an int or float')
        return 'temp'
    if (not (isinstance(ppd, int) or isinstance(ppd, float))) or ppd <= 0:
        print('Please enter the desired points-per-decade as a positive int')
        return 'temp'

    # Take the log of all the inputs
    log_x_intercept, log_y_intercept = np.log10(intercept[0]), np.log10(intercept[1])
    log_xmin, log_xmax = np.log10(xmin), np.log10(xmax)

    # Calculate the y-intercept of the line on log axes
    log_b = log_y_intercept - slope * log_x_intercept

    # Get the x- and y-values of the line as arrays
    x_vals = np.logspace(log_xmin, log_xmax, round(ppd * (log_xmax - log_xmin)))
    y_vals = (10 ** log_b) * (x_vals ** slope)

    return x_vals, y_vals


def ccdf(data, method='exclusive'):
    """
    -- Calculates the complementary cumulative distribution function (CCDF) for some input data.
    -- CCDF defined as: P(X > x)

    Parameters
    ----------
    data : arr
    method : str, optional
        How to compute the CCDF.
        "exclusive" means P(X > x)
        "inclusive" means P(X >= x)

    Returns
    -------
    [0] histx : arr
        X-values of the CCDF.
        If either an empty dataset or an incorrect method string is passed, returns empty array.
    [1] histy : arr
        Y-values of the CCDF.
        If either an empty dataset or an incorrect method string is passed, returns empty array.
    """
    if method != 'exclusive' and method != 'inclusive':
        print('Please choose between two methods: \'exclusive\' or \'inclusive\'.')
        return np.array([]), np.array([])

    data = np.array(data)
    if len(data) == 0:
        print('Data array is empty.')
        return np.array([]), np.array([])

    # Take only positive values, non-NaNs, and non-Infs
    data = data[(data > 0) * ~np.isnan(data) * ~np.isinf(data)]

    # Get the unique values and their counts
    vals, counts = np.unique(data, return_counts=True)
    # Sort both the values and their counts the same way
    histx = vals[np.argsort(vals)]
    counts = counts[np.argsort(vals)]

    # P(X > x)
    if method == 'exclusive':
        histx = np.insert(histx, 0, 0)

        # Get cumulative counts for the unique points
        cum_counts = np.cumsum(counts)

        # Get the total number of events
        total_count = cum_counts[-1]

        # Start constructing histy by saying that 100% of the data should be greater than 0
        histy = np.ones(len(counts) + 1)
        histy[1:] = 1 - (cum_counts / total_count)

    # P(X >= x)
    else:
        cum_counts = np.cumsum(counts)
        # Now we insert a 0 at the beginning of cum_counts.
        # Since Pr(X >= x) = 1 - Pr(X < x), we can get the second term from this newly expanded cum_counts
        cum_counts = np.insert(cum_counts, 0, 0)

        total_counts = cum_counts[-1]

        histy = (1 - (cum_counts / total_counts))[:-1]

    return histx, histy


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Plotting functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


def cluster_area_dist(size=50, prob=0.381, num_runs=5, end_time=100, fig_loc=None):
    """
    Makes a plot of the in_cluster area distribution at a fixed timestep.

    Parameters
    ----------
    size : int, optional
        Linear system size
    prob : float, optional
        Bond probability.
        Default value is my best estimate for the "critical value" determined by my large-N simulations.
        Very close to the best approximations in the literature.
    num_runs : int, optional
        Number of runs to do averaging over
    end_time : int, optional
        Timestep after which to end the simulation and compute the in_cluster area distribution
    fig_loc : str, optional
        Where to store the figure

    Returns
    -------
    """
    if fig_loc is None:
        fig_loc = './'

    lx = ly = size

    areas = []
    for i in range(num_runs):
        print(f'{i + 1} / {num_runs}')

        lattice = np.ones((ly, lx))
        for j in range(end_time):
            lattice = timestep(lattice, prob, lx, ly)

        labeled_array, num_features = label(lattice)
        objects = find_objects(labeled_array)

        for j in range(len(objects)):
            temp_arr = lattice[objects[j]]
            areas.append(np.count_nonzero(temp_arr))

    histx, histy = ccdf(areas)
    max_size = max(areas)
    plt.text(10**1, 10**(-3), f'Largest in_cluster = {max_size}')
    plt.semilogy(histx, histy, '.')
    plt.title(f'Runs: {num_runs}, Max timestep: {end_time}')
    plt.xlabel('cluster size (# of sites)')
    plt.ylabel('CCDF')
    plt.savefig(f'{fig_loc}/dp_area_dist_s={size}_p={prob}_timestep={end_time}.png')

    return 'temp'


def cluster_count_vs_time(size=50, prob=0.381, num_runs=5, scale='log', end_times=None, fig_loc=None):
    """
    Plots the number of clusters at log-spaced timesteps.

    Parameters
    ----------
    size : int, optional
        Linear system size
    prob : float, optional
        Bond probability.
        Default value is my best estimate for the "critical value" determined by my large-N simulations.
        Very close to the best approximations in the literature.
    num_runs : int, optional
        Number of runs to do averaging over
    scale : str, optional
        Axes scale ('log' or 'linear')
    end_times : arr, optional
        Log-spaced timesteps after which to end the simulation and compute the in_cluster area distribution.
        Default is 50 timesteps from 1 to 1,000.
    fig_loc : str, optional
        Where to store the figure

    Returns
    -------
    """
    if scale != 'linear' and scale != 'log':
        print('Must choose either "linear" or "log" for scale parameter')
        return 'temp'
    if end_times is None:
        end_times = np.logspace(0, 3, 50).astype('int')
        end_times = np.unique(end_times)
    if fig_loc is None:
        fig_loc = './'

    lx = ly = size

    cluster_counts = []
    for j in range(num_runs):
        print(f'{j + 1} / {num_runs}')
        lattice = np.ones((ly, lx))
        temp_cluster_counts = []
        for i in range(1, end_times[-1] + 1):
            lattice = timestep(lattice, prob, lx, ly)
            if i in end_times:
                labeled_array, num_features = label(lattice)
                temp_cluster_counts.append(num_features)
        cluster_counts.append(temp_cluster_counts)

    cluster_counts = np.array(cluster_counts)
    cluster_counts_avg = np.mean(cluster_counts, axis=0)
    cluster_counts_std = np.std(cluster_counts, axis=0)

    plt.plot(end_times, cluster_counts_avg, linewidth=0.5, color='b')
    if scale == 'log':
        plt.loglog()
    plt.errorbar(end_times, cluster_counts_avg, linestyle='None', yerr=cluster_counts_std, color='k')

    x, y = linemaker(-0.46, [10 ** 2, 2 * 10 ** 4], 10 ** 1, 10 ** 3)
    plt.loglog(x, y, linestyle='dashed', color='r', label=r'$C(t) \sim t^{-0.46}$')

    plt.title(r'cluster num. C(t);' + f'\n size={size}, prob={prob}, runs={num_runs}, max time={end_times[-1].astype("int")}')
    plt.ylabel('No. of clusters (incl. boundaries)')
    plt.xlabel('Timestep')
    plt.ylim(bottom=min(cluster_counts_avg))
    plt.legend()
    plt.savefig(f'{fig_loc}/cluster_num_vs_time_s={size}_p={prob}_maxtime={end_times[-1].astype("int")}_scale={scale}.png')

    return 'temp'


def density(size=50, prob=0.381, num_runs=2, plot_slope=False, times=None, fig_loc=None):
    """
    Plots the density of occupied sites over time.

    Parameters
    ----------
    size : int, optional
        Linear system size.
    prob : float, optional
        Bond probability.
        Default value is my best estimate for the "critical value" determined by large-N simulations.
        Very close to the best approximations in the literature.
    num_runs : int, optional
        Number of runs to do averaging over.
    plot_slope : bool, optional
        Plot a reference slope for the known scaling law.
    times : arr, optional
        Log-spaced timesteps at which to look at density.
    fig_loc : str, optional
        Where to store the figure.

    Returns
    -------
    """
    if times is None:
        times = np.logspace(0, 4, 60)  # start and stop are the exponents, i.e., start = 0 means 10**0
    if fig_loc is None:
        fig_loc = './'

    lx = ly = size

    times = times.astype('int')
    times = np.unique(times)

    densities = []
    for i in range(num_runs):
        print(f'DP run {i + 1} / {num_runs}')
        slices = np.ones((ly, lx))
        temp_count = []
        for j in range(1, max(times) + 1):
            slices = timestep(slices, prob, lx, ly)
            if j in times:
                temp_count.append(len(np.where(slices == 1)[0]))
        temp_count = np.array(temp_count)
        temp_density = temp_count / (size * size)
        densities.append(temp_density)

    density_avg = np.mean(densities, axis=0)
    density_std = np.std(densities, axis=0)

    plt.loglog(times, density_avg, linewidth=0.5, color='b')
    plt.errorbar(times, density_avg, linestyle='None', yerr=density_std, color='k')
    if plot_slope:
        x, y = linemaker(-0.46, [10**2, 2*10**(-1)], 10**1, 10**4)
        plt.loglog(x, y, linestyle='dashed', color='r', label=r'$\rho(t) \sim t^{-0.46}$')

    plt.title(r'Density of active sites $\rho(t)$' + f'\n size={size}, prob={prob}, runs={num_runs}')
    plt.ylabel(r'$\rho(t)$')
    plt.xlabel('Timestep')
    plt.loglog()
    if plot_slope:
        plt.legend()
    plt.savefig(f'{fig_loc}/density_size={size}_prob={prob}_runs={num_runs}.png')

    return 'temp'


def density_compare(size=50, num_runs=2, probs=None, times=None, fig_loc=None):
    """

    Parameters
    ----------
    size
    num_runs
    plot_slope
    probs
    times
    fig_loc

    Returns
    -------

    """
    if probs is None:
        probs = [0.38216, 0.382223, 0.387]
    if times is None:
        times = np.logspace(0, 4, 60)  # start and stop are the exponents, i.e., start = 0 means 10**0
    if fig_loc is None:
        fig_loc = './'

    lx = ly = size

    times = times.astype('int')
    times = np.unique(times)

    fig, ax = plt.subplots(1, 1)
    ax.set_prop_cycle(color=['r', 'g', 'b'])

    for prob in probs:
        densities = []
        for i in range(num_runs):
            print(f'Prob {prob}, run {i + 1} / {num_runs}')
            slices = np.ones((ly, lx))
            temp_count = []
            for j in range(1, max(times) + 1):
                slices = timestep(slices, prob, lx, ly)
                if j in times:
                    temp_count.append(len(np.where(slices == 1)[0]))
            temp_count = np.array(temp_count)
            temp_density = temp_count / (size * size)
            densities.append(temp_density)

        density_avg = np.mean(densities, axis=0)
        density_std = np.std(densities, axis=0)

        ax.loglog(times, density_avg, linewidth=0.5, label=r'$p = $' + f'{prob}')
        ax.errorbar(times, density_avg, linestyle='None', yerr=density_std, color='k')

    ax.set_title(r'Density of active sites $\rho(t)$ compare' + f'\n size={size}, runs={num_runs}')
    ax.set_ylabel(r'$\rho(t)$')
    ax.set_xlabel('Timestep')
    ax.legend()
    plt.savefig(f'{fig_loc}/density_compare_s={size}_r={num_runs}.png')

    return 'temp'


def survProb(size=500, prob=0.382223, num_runs=100, fig_loc=None):
    """
    Plots a histogram of single-seeded DP survival lifetimes.

    Parameters
    ----------
    size : int, optional
        Linear system size
    prob : float, optional
        Bond probability
    num_runs : int, optional
        Number of DP simulations to accumulute survival times from
    fig_loc : str, optional
        Where to store the figure

    Returns
    -------
    """
    if fig_loc is None:
        fig_loc = './'

    lx = ly = size

    surv_times = []
    for i in range(num_runs):
        print(f'{i + 1} / {num_runs}')
        lattice = np.zeros((ly, lx))
        yco, xco = np.random.choice(lx, 2)
        lattice[yco][xco] = 1

        t = 0
        while len(np.where(lattice == 1)[0]) != 0:
            lattice = timestep(lattice, prob, lx, ly)
            t += 1
        surv_times.append(t)

    plt.hist(surv_times)
    plt.savefig(f'{fig_loc}/lifetime_hist_s={size}_p={prob}_numruns={num_runs}.png')

    return 'temp'
