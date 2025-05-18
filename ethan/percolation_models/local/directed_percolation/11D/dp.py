"""
Based on work by someone else

Started 3/22/23
"""
import mpltex
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import njit
from numba import vectorize
from PIL import Image, ImageDraw
from helper_scripts.get_cum_dist import get_cum_dist as gcd
from joblib import Parallel, delayed
from helper_scripts.logbinning import *
from helper_scripts.fit_2 import fit_2

mpl.rcParams['figure.dpi'] = 100


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Simulation functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


@vectorize(nopython=True)
def logic2(a, b, r1, p):
    # a is the current row without the rightmost element
    # b is the current row without the leftmost element
    # r1 is the array of random numbers from 0 to 1 of the same length as a & b

    if a == 1:
        if b == 1:  # CASE A A
            if r1 < (1 - (1 - p) ** 2):
                return 1
            else:
                return 0
        else:  # CASE A I
            if r1 < p:
                return 1
            else:
                return 0
    else:
        if b == 1:  # CASE A I
            if r1 < p:
                return 1
            else:
                return 0
        else:  # CASE I I
            return 0


@njit
def simula_DP(p, L0, T):
    L = np.copy(L0)
    N = len(L0)  # Number of columns (system width)
    H = np.zeros((T + 1, N))  # T+1 is the number of rows (number of timesteps)
    H[0] = np.copy(L)
    d = 1
    cont = 1
    for t in range(T):  # Loop through the timesteps
        L2 = np.zeros(len(L))  # Initialize a new row
        if d == 1:
            # MOVE TO THE RIGHT
            d = 2
            x = L[:-1]  # The first row (L0) without the rightmost element
            y = L[1:]  # The first row (L0) without the leftmost element
            r = np.random.random(len(x))  # Array of length (columns - 1) that contains random floats from 0 to 1
            L2[:-1] = logic2(x, y, r, p)  # The new row
            if L[-1] == 1:
                if np.random.random() < p:
                    L2[-1] = 1
            else:
                L2[-1] = L[-1]
            L = np.copy(L2)
        else:  # MOVE TO THE LEFT
            d = 1
            x = L[:-1]
            y = L[1:]
            r = np.random.random(len(x))
            L2[1:] = logic2(x, y, r, p)
            if L[1] == 1:
                if np.random.random() < p:
                    L2[1] = 1
            else:
                L2[1] = L[1]
            L = np.copy(L2)
        H[cont] = np.copy(L)
        cont += 1

    return H


def disegna(**kwargs):
    stampa = False

    for key, value in kwargs.items():
        if key == 'sim':
            a = value
        elif key == 'dx':
            dx = value

    bg = (255, 255, 255)
    blue = (0, 0, 255)
    green = (0, 255, 0)
    black = (0, 0, 0)
    dim = a.shape
    img = Image.new('RGB', (dx * dim[1] * 2 + dx, dx * dim[0] + dx), bg)
    draw = ImageDraw.Draw(img)

    for j in range(dim[0]):
        y = dx * j
        for i in range(dim[1]):
            if j % 2 == 0:
                x = 2 * i * dx
            else:
                x = dx + 2 * i * dx
            if a[j, i]:
                color = blue
                draw.polygon(((x + dx, y), (x + 2 * dx, y + dx), (x + dx, y + 2 * dx), (x, y + dx)), fill=color, outline=bg)
            else:
                color = black
                draw.polygon(((x + dx, y), (x + 2 * dx, y + dx), (x + dx, y + 2 * dx), (x, y + dx)), fill=color,
                             outline=bg)

    return img


# 4/6/23: Now this just plots one simulation
def sim_and_plot(**kwargs):
    # Default values of all the parameters
    N = 1001  # Linear dimension of the lattice (or at least the number of columns...)...201 worked well
    p = 0.6447  # Critical value is p = 0.6447
    dp = 'full_row'  # Can be 'full_row' or 'single_site'

    # Unpack the parameters
    for key, value in kwargs.items():
        if key == 'dp_type':
            dp = value
        elif key == 'sys_size':
            N = value
        elif key == 'bond_prob':
            p = value

    # Get the first row
    if dp == 'full_row':
        L0 = np.ones(N)
    else:
        L0 = np.zeros(N)  # First row of the lattice
        L0[int(1 + N / 2)] = 1  # Infect the central site in the first row

    # Run the simulation
    # Typical number of timesteps is int(N * 1.5)
    a = simula_DP(p, L0, int(N * 1.5))  # Third argument is (number of rows - 1)

    # Save the image of the simulation
    dx = 1
    image = disegna(sim=a, dx=dx)
    image.save(f'/Users/emullen98/Desktop/Directed-Percolation-and-DP-2-main/plots/lattices/{dp}_sim_width_{N}.png')

    return 'temp'


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Calculate mu_parallel and mu_perp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


# 4/6/23: Helper function that gets clusters of 1's and 0's from a given list
def groups(lst):
    zero_groups = []
    ones_groups = []

    start = None
    for i, num in enumerate(lst):
        if num == 0:
            if start is None:
                start = i
        else:
            if start is not None:
                zero_groups.append(lst[start:i])
                start = None
    if start is not None:
        zero_groups.append(lst[start:])

    start = None
    for i, num in enumerate(lst):
        if num == 1:
            if start is None:
                start = i
        else:
            if start is not None:
                ones_groups.append(lst[start:i])
                start = None
    if start is not None:
        ones_groups.append(lst[start:])

    len_zero_list = [len(subgroup) for subgroup in zero_groups]
    len_ones_list = [len(subgroup) for subgroup in ones_groups]

    return len_zero_list, len_ones_list


# 4/6/23: Use for finding mu_perpendicular (distribution of laminar gaps at a fixed time)
# Can adjust the time axis length (# of rows) here
# Gave me an error for T=1.5*N
def group_getter_one_time(p, L0, N):
    # Run the simulation
    a = simula_DP(p, L0, int(1.4*N))  # Third argument is (number of rows - 1)

    # Get the statistics on either the cluster widths or the gap widths
    zero_groups, ones_groups = groups(list(a[-1]))  # Get only the last row

    return zero_groups, ones_groups


# 4/6/23: Use for finding mu_parallel (distribution of laminar gaps at a fixed position)
def group_getter_one_pos(p, L0, N):
    # Run the simulation
    a = simula_DP(p, L0, int(N))  # Third argument is (number of rows - 1)

    # Rotate the array to get the laminar gaps at a fixed position to work well with helper function 'groups':
    a = np.rot90(a)

    # Get the statistics on either the cluster widths or the gap widths
    zero_groups, ones_groups = groups(list(a[int(N/2)]))  # Get only the middle-ish row

    return zero_groups, ones_groups


def one_time_avg(**kwargs):
    # Default values of all the parameters
    N = 2001  # Linear dimension of the lattice (or at least the number of columns...)...201 worked well
    p = 0.6447  # Critical value is p = 0.6447
    dp = 'full_row'  # Can be 'full_row' or 'single_site'
    num_loops = 100
    do_plots = False

    # Unpack the parameters
    for key, value in kwargs.items():
        if key == 'dp_type':
            dp = value
        elif key == 'sys_size':
            N = value
        elif key == 'bond_prob':
            p = value
        elif key == 'num_loops':
            num_loops = value
        elif key == 'plots':
            do_plots = value

    # Get the first row
    if dp == 'full_row':
        L0 = np.ones(N)
    else:
        L0 = np.zeros(N)  # First row of the lattice
        L0[int(1 + N / 2)] = 1  # Infect the central site in the first row

    # Run some number of simulations for the same lattice size and add the clusters and gaps to two huge lists
    zero_group_counts = []
    ones_group_counts = []
    res = Parallel(n_jobs=-1)(delayed(group_getter_one_time)(p, L0, N) for i in range(num_loops))
    zero_groups, ones_groups = zip(*res)
    for group in zero_groups:
        zero_group_counts.extend(group)
    for group in ones_groups:
        ones_group_counts.extend(group)

    # Go to plotting function:
    if do_plots:
        one_time_avg_plots(zero_group_counts, ones_group_counts, N, num_loops)

    return ones_group_counts, zero_group_counts


def one_pos_avg(**kwargs):
    # Default values of all the parameters
    N = 2001  # Linear dimension of the lattice (or at least the number of columns...)...201 worked well
    p = 0.6447  # Critical value is p = 0.6447
    dp = 'full_row'  # Can be 'full_row' or 'single_site'
    num_loops = 100
    do_plots = False

    # Unpack the parameters
    for key, value in kwargs.items():
        if key == 'dp_type':
            dp = value
        elif key == 'sys_size':
            N = value
        elif key == 'bond_prob':
            p = value
        elif key == 'num_loops':
            num_loops = value
        elif key == 'plots':
            do_plots = value

    # Get the first row
    if dp == 'full_row':
        L0 = np.ones(N)
    else:
        L0 = np.zeros(N)  # First row of the lattice
        L0[int(1 + N / 2)] = 1  # Infect the central site in the first row

    # Run some number of simulations for the same lattice size and add the clusters and gaps to two huge lists
    zero_group_counts = []
    ones_group_counts = []
    res = Parallel(n_jobs=-1)(delayed(group_getter_one_pos)(p, L0, N) for i in range(num_loops))
    zero_groups, ones_groups = zip(*res)
    for group in zero_groups:
        zero_group_counts.extend(group)
    for group in ones_groups:
        ones_group_counts.extend(group)

    # Go to plotting function:
    if do_plots:
        one_pos_avg_plots(zero_group_counts, ones_group_counts, N, num_loops)

    return ones_group_counts, zero_group_counts


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Plotting and or fitting laminar gap distributions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


@mpltex.aps_decorator
def one_time_avg_plots(zero_group_sizes, ones_group_sizes, N, num_loops):
    # Generate and plot the CCDFs
    options = ['clusters', 'gaps']
    for option in options:
        if option == 'clusters':
            histx, histy = gcd(ones_group_sizes)
            plt.close('all')
            plt.loglog(histx, histy)
            plt.title(f'Cluster Size CCDF for sys width {str(N)}, fixed time, {str(num_loops)} sims')
            plt.savefig(f'plots/ccdfs/{option}_ccdf_loglog_width_{N}_fixedtime.png')

            plt.close('all')
            plt.plot(histx, histy)
            plt.yscale('log')
            plt.title(f'Cluster Size CCDF for sys width {str(N)}, fixed time, {str(num_loops)} sims')
            plt.savefig(f'plots/ccdfs/{option}_ccdf_loglin_width_{N}_fixedtime.png')
        else:
            histx, histy = gcd(zero_group_sizes)
            plt.close('all')
            plt.loglog(histx, histy)
            plt.title(f'Gap Size CCDF for sys width {str(N)}, fixed time, {str(num_loops)} sims')
            plt.savefig(f'plots/ccdfs/{option}_ccdf_loglog_width_{N}_fixedtime.png')

            plt.close('all')
            plt.plot(histx, histy)
            plt.yscale('log')
            plt.title(f'Gap Size CCDF for sys width {str(N)}, fixed time, {str(num_loops)} sims')
            plt.savefig(f'plots/ccdfs/{option}_ccdf_loglin_width_{N}_fixedtime.png')

    return 'temp'


@mpltex.aps_decorator
def one_pos_avg_plots(zero_group_sizes, ones_group_sizes, N, num_loops):
    # Generate and plot the CCDFs
    options = ['clusters', 'gaps']
    for option in options:
        if option == 'clusters':
            histx, histy = gcd(ones_group_sizes)
            plt.close('all')
            plt.loglog(histx, histy)
            plt.title(f'Cluster Size CCDF for sys width {str(N)}, fixed position, {str(num_loops)} sims')
            plt.savefig(f'plots/ccdfs/{option}_ccdf_loglog_width_{N}_fixedpos.png')

            plt.close('all')
            plt.plot(histx, histy)
            plt.yscale('log')
            plt.title(f'Cluster Size CCDF for sys width {str(N)}, fixed position, {str(num_loops)} sims')
            plt.savefig(f'plots/ccdfs/{option}_ccdf_loglin_width_{N}_fixedpos.png')
        else:
            histx, histy = gcd(zero_group_sizes)
            plt.close('all')
            plt.loglog(histx, histy)
            plt.title(f'Gap Size CCDF for sys width {str(N)}, fixed position, {str(num_loops)} sims')
            plt.savefig(f'plots/ccdfs/{option}_ccdf_loglog_width_{N}_fixedpos.png')

            plt.close('all')
            plt.plot(histx, histy)
            plt.yscale('log')
            plt.title(f'Gap Size CCDF for sys width {str(N)}, fixed position, {str(num_loops)} sims')
            plt.savefig(f'plots/ccdfs/{option}_ccdf_loglin_width_{N}_fixedpos.png')

    return 'temp'


# 4/3/23: Uses fit_2 to get an estimate for the power law exponent on the gap size distribution and shows that exponent
# on the plot.
# ADJUST PLOT FILENAME WHENEVER YOU CHANGE THE TIME AXIS LENGTH (# OF ROWS)
@mpltex.aps_decorator
def one_time_avg_fit(gap_sizes, p):
    min_gap_size, max_gap_size = np.log10(2 * 10 ** 0), np.log10(3 * 10 ** 1)
    gaps_x, gaps_y = gcd(gap_sizes)
    bx, by, _ = logbinning(gaps_x, gaps_y, 100)
    gap_exp, _, _ = fit_2(np.log10(bx), np.log10(by), minval=min_gap_size, maxval=max_gap_size)
    sizes = np.linspace(min_gap_size, max_gap_size, num=100)
    sizes = 10 ** sizes
    ystuff = -gap_exp[1] * sizes ** (gap_exp[0])

    plt.loglog(gaps_x, gaps_y)
    plt.title(r'Gap CCDF, $N=8000$, ' + f'p={str(p)},' + r' $C(L) \sim L^{\tau}$')
    plt.plot(sizes, ystuff, color='r', linewidth=1, linestyle='--', alpha=1, label=rf'$\tau$ = {(gap_exp[0]):.2f}')
    plt.axvline(x=10 ** min_gap_size, color='k', alpha=0.7)
    plt.axvline(x=10 ** max_gap_size, color='k', alpha=0.7)
    plt.ylim(top=2*10**0)
    plt.xlabel('Gap sizes')
    plt.legend()
    plt.tight_layout(pad=0.6)
    plt.savefig(f'gaps_ccdf_width=8000_T=1.4N_p={str(p)}.png')

    return 'temp'


@mpltex.aps_decorator
def one_pos_avg_fit(gap_sizes):
    min_gap_size, max_gap_size = np.log10(1*10**1), np.log10(4*10**2)
    gaps_x, gaps_y = gcd(gap_sizes)
    bx, by, _ = logbinning(gaps_x, gaps_y, 100)
    gap_exp, _, _ = fit_2(np.log10(bx), np.log10(by), minval=min_gap_size, maxval=max_gap_size)
    sizes = np.linspace(min_gap_size, max_gap_size, num=100)
    sizes = 10 ** sizes
    ystuff = -gap_exp[1] * sizes ** (gap_exp[0])

    plt.loglog(gaps_x, gaps_y)
    plt.title(r'Gap CCDF, $N=8000$, fixed pos, $C(L) \sim L^{{\mu}_{||}-1}$')
    plt.plot(sizes, ystuff, color='r', linewidth=1, linestyle='--', alpha=1, label=r'${\mu}_{||}-1$ = ' + f'{(gap_exp[0]):.2f}')
    plt.axvline(x=10 ** min_gap_size, color='k', alpha=0.7)
    plt.axvline(x=10 ** max_gap_size, color='k', alpha=0.7)
    plt.ylim(top=2 * 10 ** 0)
    plt.xlabel('Gap sizes')
    plt.legend()
    plt.tight_layout(pad=0.6)
    plt.savefig(f'plots/ccdfs/with_lsqfits/gaps_ccdf_width=8000_T=N_fixedpos.png')

    return 'temp'


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Control section
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


if __name__ == '__main__':
    sim_and_plot(sys_size=201)

