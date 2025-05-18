"""
Created Jul 16 2024
Updated Jul 18 2024

(IN CLUSTER)

"""
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(f'./lattices/s100/sp_corr_func_s=100_p=0.5927_jobid=12164093_combined.txt')
dists = data[0]
means = data[1]
stdevs = data[2]

plt.title(r'Avg. corr. func. for 100 size-100 lattices')
plt.xlabel('r')
plt.ylabel(r'$\langle C(r) \rangle$')
plt.plot(dists, means, '.')
# plt.fill_between(distances, means - stdevs, means + stdevs, color='r', alpha=0.5)
# plt.ylim(bottom=10**(-1))
plt.savefig('./corr_func_combined.png', dpi=200)
