"""
Created Jul 03 2024
Updated Jun 03 2024

(IN CLUSTER)
Show collapsed perimeter & area CCDFs for different p_c.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import common_code as cc

do_perim = True
do_area = True
do_beforepc = True
do_afterpc = False
# Kappas are the PDF exponents
# Sigmas are the difference between the integrated and non-integrated exponents
kappa_perim = 2.50
sigma_perim = 0.6
kappa_area = 2.0
sigma_area = 0.4
p_c = 0.405

folder_list = os.listdir(f'./lattices/s50000')

fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(10, 6))
fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(10, 6))
fig3, (ax31, ax32) = plt.subplots(1, 2, figsize=(10, 6))
fig4, (ax41, ax42) = plt.subplots(1, 2, figsize=(10, 6))

prob_list_beforepc = [0.34, 0.36, 0.38]
prob_list_afterpc = [0.415, 0.425, 0.435]

if do_beforepc:
    for p in prob_list_beforepc:
        data = np.load(f'./lattices/s50000/siteperc_lattice_s=50000_p={p}_job=_task=.npy')
        perims, areas = data[0], data[1]
        perims = perims[~np.isnan(perims)]
        areas = areas[~np.isnan(areas)]
        del data

        if do_perim:
            phistx, phisty = cc.ccdf(perims)
            del perims
            ax11.loglog(phistx, phisty, '.', label=f'Prob. = {p}')
            phistx_col = phistx * (np.abs((p_c - p) / p_c) ** (1 / sigma_perim))
            phisty_col = phisty * (np.abs((p_c - p) / p_c) ** (-1 * (kappa_perim - 1) / sigma_perim))
            del phistx, phisty
            ax12.loglog(phistx_col, phisty_col, '.', label=f'Prob. = {p}')
            del phistx_col, phisty_col

        if do_area:
            ahistx, ahisty = cc.ccdf(areas)
            del areas
            ax21.loglog(ahistx, ahisty, '.', label=f'Prob. = {p}')
            ahistx_col = ahistx * (np.abs((p_c - p) / p_c) ** (1 / sigma_area))
            ahisty_col = ahisty * (np.abs((p_c - p) / p_c) ** (-1 * (kappa_area - 1) / sigma_area))
            del ahistx, ahisty
            ax22.loglog(ahistx_col, ahisty_col, '.', label=f'Prob. = {p}')
            del ahistx_col, ahisty_col

    if do_perim:
        ax11.set_title(f'Perimeter CCDF uncollapsed \n System size = {50000:,}, ' + rf'$\kappa_{{{{\text{{perim}}}}}} = {kappa_perim}, \sigma_{{{{\text{{perim}}}}}} = {sigma_perim}$')
        ax11.set_xlabel(r'$P$')
        ax11.set_ylabel(r'$C(P)$')
        ax11.legend(loc='upper right', fancybox=True, shadow=True)

        ax12.set_title(f'Perimeter CCDF collapsed \n System size = {50000:,}, ' + rf'$\kappa_{{{{\text{{perim}}}}}} = {kappa_perim}, \sigma_{{{{\text{{perim}}}}}} = {sigma_perim}$')
        ax12.set_xlabel(r'$P \cdot ((p_c - p) / p_c)^{1 / \sigma_{\text{perim}}}$')
        ax12.set_ylabel(r'$C(P) \cdot ((p_c - p) / p_c)^{-(\kappa_{\text{perim}} - 1) / \sigma_{\text{perim}}}$')
        ax12.legend(loc='upper right', fancybox=True, shadow=True)

        fig1.tight_layout()
        fig1.savefig(f'./siteperc_perim_ccdf_collapse_nomoment_s=50000_kappaperim={kappa_perim}_sigmaperim={sigma_perim}_beforepc.png', dpi=200)
        plt.close(fig=fig1)

    if do_area:
        ax21.set_title(f'Area CCDF uncollapsed \n System size = {50000:,}, ' + rf'$\kappa_{{{{\text{{area}}}}}} = {kappa_area}, \sigma_{{{{\text{{area}}}}}} = {sigma_area}$')
        ax21.set_xlabel(r'$A$')
        ax21.set_ylabel(r'$C(A)$')
        ax21.legend(loc='upper right', fancybox=True, shadow=True)

        ax22.set_title(f'Area CCDF collapsed \n System size = {50000:,}, ' + rf'$\kappa_{{{{\text{{area}}}}}} = {kappa_area}, \sigma_{{{{\text{{area}}}}}} = {sigma_area}$')
        ax22.set_xlabel(r'$A \cdot ((p_c - p) / p_c)^{1 / \sigma_{\text{area}}}$')
        ax22.set_ylabel(r'$C(A) \cdot ((p_c - p) / p_c)^{-(\kappa_{\text{area}} - 1) / \sigma_{\text{area}}}$')
        ax22.legend(loc='upper right', fancybox=True, shadow=True)

        fig2.tight_layout()
        fig2.savefig(f'./siteperc_area_ccdf_collapse_nomoment_s=50000_kappaarea={kappa_area}_sigmaarea={sigma_area}_beforepc.png', dpi=200)
        plt.close(fig=fig2)

if do_afterpc:
    for p in prob_list_afterpc:
        data = np.load(f'./lattices/s50000/siteperc_lattice_s=50000_p={p}_job=_task=.npy')
        perims, areas = data[0], data[1]

        if do_perim:
            phistx, phisty = cc.ccdf(perims)
            ax31.loglog(phistx, phisty, '.', label=f'Prob. = {p}')
            phistx_col = phistx * (np.abs((p_c - p) / p_c) ** (1 / sigma_perim))
            phisty_col = phisty * (np.abs((p_c - p) / p_c) ** (-1 * (kappa_perim - 1) / sigma_perim))
            ax32.loglog(phistx_col, phisty_col, '.', label=f'Prob. = {p}')

        if do_area:
            ahistx, ahisty = cc.ccdf(areas)
            ax41.loglog(ahistx, ahisty, '.', label=f'Prob. = {p}')
            ahistx_col = ahistx * (np.abs((p_c - p) / p_c) ** (1 / sigma_area))
            ahisty_col = ahisty * (np.abs((p_c - p) / p_c) ** (-1 * (kappa_area - 1) / sigma_area))
            ax42.loglog(ahistx_col, ahisty_col, '.', label=f'Prob. = {p}')

    if do_perim:
        ax31.set_title(f'Perimeter CCDF uncollapsed \n System size = {50000:,}, ' + rf'$\kappa_{{{{\text{{perim}}}}}} = {kappa_perim}, \sigma_{{{{\text{{perim}}}}}} = {sigma_perim}$')
        ax31.set_xlabel(r'$P$')
        ax31.set_ylabel(r'$C(P)$')
        ax31.legend(loc='upper right', fancybox=True, shadow=True)

        ax32.set_title(f'Perimeter CCDF collapsed \n System size = {50000:,}, ' + rf'$\kappa_{{{{\text{{perim}}}}}} = {kappa_perim}, \sigma_{{{{\text{{perim}}}}}} = {sigma_perim}$')
        ax32.set_xlabel(r'$P \cdot (|p_c - p| / p_c)^{1 / \sigma_{\text{perim}}}$')
        ax32.set_ylabel(r'$C(P) \cdot (|p_c - p| / p_c)^{-(\kappa_{\text{perim}} - 1) / \sigma_{\text{perim}}}$')
        ax32.legend(loc='upper right', fancybox=True, shadow=True)

        fig3.tight_layout()
        fig3.savefig(f'./siteperc_perim_ccdf_collapse_nomoment_s=50000_kappaperim={kappa_perim}_sigmaperim={sigma_perim}_afterpc.png', dpi=200)

    if do_area:
        ax41.set_title(f'Area CCDF uncollapsed \n System size = {50000:,}, ' + rf'$\kappa_{{{{\text{{area}}}}}} = {kappa_area}, \sigma_{{{{\text{{area}}}}}} = {sigma_area}$')
        ax41.set_xlabel(r'$A$')
        ax41.set_ylabel(r'$C(A)$')
        ax41.legend(loc='upper right', fancybox=True, shadow=True)

        ax42.set_title(f'Area CCDF collapsed \n System size = {50000:,}, ' + rf'$\kappa_{{{{\text{{area}}}}}} = {kappa_area}, \sigma_{{{{\text{{area}}}}}} = {sigma_area}$')
        ax42.set_xlabel(r'$A \cdot (|p_c - p| / p_c)^{1 / \sigma_{\text{area}}}$')
        ax42.set_ylabel(r'$C(A) \cdot (|p_c - p| / p_c)^{-(\kappa_{\text{area}} - 1) / \sigma_{\text{area}}}$')
        ax42.legend(loc='upper right', fancybox=True, shadow=True)

        fig4.tight_layout()
        fig4.savefig(f'./siteperc_area_ccdf_collapse_nomoment_s=50000_kappaarea={kappa_area}_sigmaarea={sigma_area}_afterpc.png', dpi=200)
