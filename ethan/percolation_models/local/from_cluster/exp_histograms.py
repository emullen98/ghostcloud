"""
Created Jun 16 2024
Updated Jun 16 2024

Plots histograms of the exponents gathered from 100 runs of the directed percolation model.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/all_exps.csv')
areas = df['area_pdf']
perims = df['perim_pdf']
pva = df['dvs']

perim_fig, perim_ax = plt.subplots(1, 1)
area_fig, area_ax = plt.subplots(1, 1)
pva_fig, pva_ax = plt.subplots(1, 1)

perim_ax.set_title('Perimeter PDF exponent histogram, 100 runs')
perim_ax.set_xlabel(r'$\kappa_{perim}$')
perim_ax.set_ylabel('Count')
perim_ax.hist(perims)
perim_fig.tight_layout()
perim_fig.savefig('../../perim_exp_hist.png', dpi=200)

area_ax.set_title('Area PDF exponent histogram, 100 runs')
area_ax.set_xlabel(r'$\kappa_{area}$')
area_ax.set_ylabel('Count')
area_ax.hist(areas)
area_fig.tight_layout()
area_fig.savefig('../../area_exp_hist.png', dpi=200)

pva_ax.set_title('Fractal dimension histogram, 100 runs')
pva_ax.set_xlabel(r'$D_f$')
pva_ax.set_ylabel('Count')
pva_ax.hist(pva)
pva_fig.tight_layout()
pva_fig.savefig('../../pva_exp_hist.png', dpi=200)
