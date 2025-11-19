"""
Process the PNG area & perimeter stuff
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/emullen98/Desktop')
import helper_scripts as hs



def plot_area():
    # df = pd.read_csv('/Users/emullen98/Downloads/fd_2025-11-14_png_cl4_cf8_ordLF_bbox_png_all_multi_314a4bc3_raw_areas_perims.csv')
    df = pd.read_csv('/Users/emullen98/Downloads/fd_2025-11-14_png_cl4_cf8_ordLF_bbox_png_argmax_multi_6487f80f_raw_areas_perims.csv')
    # df = pd.read_csv('/Users/emullen98/Downloads/fd_2025-11-14_png_cl4_cf4_ordLF_bbox_png_all_multi_b127c9d2_raw_areas_perims.csv')
    # df = pd.read_csv('/Users/emullen98/Downloads/fd_2025-11-14_png_cl4_cf4_ordLF_bbox_png_argmax_multi_0961b721_raw_areas_perims.csv')

    areas = df['area_px'].to_numpy()

    hx, hy = hs.ccdf(areas)

    x, y = hs.linemaker(-0.9, [10**4, 3*10**(-2)], 10**3, 10**6)

    plt.loglog(hx, hy)
    plt.loglog(x, y, '--', color='r', label='C(A) ~ A^(-0.9)')
    plt.legend()
    plt.title('Area CCDF for PNG argmax cl4_cf8_ordLF_bbox')
    plt.xlabel('Area (px)')
    plt.show()

    return 'temp'


def plot_perimeter():
    df = pd.read_csv('/Users/emullen98/Downloads/fd_2025-11-14_png_cl4_cf8_ordLF_bbox_png_all_multi_314a4bc3_raw_areas_perims.csv')
    # df = pd.read_csv('/Users/emullen98/Downloads/fd_2025-11-14_png_cl4_cf8_ordLF_bbox_png_argmax_multi_6487f80f_raw_areas_perims.csv')
    # df = pd.read_csv('/Users/emullen98/Downloads/fd_2025-11-14_png_cl4_cf4_ordLF_bbox_png_all_multi_b127c9d2_raw_areas_perims.csv')
    # df = pd.read_csv('/Users/emullen98/Downloads/fd_2025-11-14_png_cl4_cf4_ordLF_bbox_png_argmax_multi_0961b721_raw_areas_perims.csv')

    perims_hull = df['perim_hull_edge'].to_numpy()
    perims_acce = df['perim_accessible_edge'].to_numpy()

    hx, hy = hs.ccdf(perims_hull)
    ha, hb = hs.ccdf(perims_acce)

    x_hull, y_hull = hs.linemaker(-1.25, [10**4, 2*10**(-3)], 3*10**2, 10**5)
    x_acce, y_acce = hs.linemaker(-1.5, [10**4, 2*10**(-4)], 3*10**2, 10**5)

    plt.loglog(hx, hy, label='hull')
    plt.loglog(ha, hb, label='accessible')
    plt.loglog(x_hull, y_hull, '--', color='k', label='CCDF(L) ~ L^(-1.25)')
    plt.loglog(x_acce, y_acce, '--', color='r', label='CCDF(L) ~ L^(-1.5)')
    plt.legend()
    plt.title('Perimeter CCDFs for PNG all cl4_cf8_ordLF_bbox')
    plt.xlabel('Perimeter (edges)')
    plt.show()

    return 'temp'


def plot_pva():
    df = pd.read_csv('/Users/emullen98/Downloads/fd_2025-11-14_png_cl4_cf8_ordLF_bbox_png_all_multi_314a4bc3_raw_areas_perims.csv')
    # df = pd.read_csv('/Users/emullen98/Downloads/fd_2025-11-14_png_cl4_cf8_ordLF_bbox_png_argmax_multi_6487f80f_raw_areas_perims.csv')
    # df = pd.read_csv('/Users/emullen98/Downloads/fd_2025-11-14_png_cl4_cf4_ordLF_bbox_png_all_multi_b127c9d2_raw_areas_perims.csv')
    # df = pd.read_csv('/Users/emullen98/Downloads/fd_2025-11-14_png_cl4_cf4_ordLF_bbox_png_argmax_multi_0961b721_raw_areas_perims.csv')

    areas = df['area_px'].to_numpy()
    perims_hull = df['perim_hull_edge'].to_numpy()
    perims_acce = df['perim_accessible_edge'].to_numpy()

    areas_bin, perims_hull_bin, _ = hs.logbinning(areas, perims_hull, 40)
    areas_bin, perims_acce_bin, _ = hs.logbinning(areas, perims_acce, 40)

    params_hull, errs_hull, _ = hs.fit(xdata=np.log10(areas_bin), 
                                       ydata=np.log10(perims_hull_bin), 
                                       xmin=np.log10(2000))
    d_f_hull = params_hull[0]
    d_f_hull_err= errs_hull[0]
    
    params_acce, errs_acce, _ = hs.fit(xdata=np.log10(areas_bin), 
                                       ydata=np.log10(perims_acce_bin), 
                                       xmin=np.log10(2000))
    d_f_acce = params_acce[0]
    d_f_acce_err= errs_acce[0]

    print(f'D_f hull: {d_f_hull:.3f} +/- {d_f_hull_err:.3f}')
    print(f'D_f acce: {d_f_acce:.3f} +/- {d_f_acce_err:.3f}')

    # plt.scatter(areas, perims_hull, color='silver', edgecolor='none', alpha=0.4, label='hull')
    # plt.scatter(areas, perims_acce, color='k', edgecolor='none', alpha=0.4, label='accessible')
    plt.plot(areas_bin, perims_hull_bin, marker='.', linestyle='None', label=f'hull, L ~ A^{d_f_hull:.3f}')
    plt.plot(areas_bin, perims_acce_bin, marker='.', linestyle='None', label=f'accessible, L ~ A^{d_f_acce:.3f}')
    plt.legend()
    plt.loglog()
    plt.title('PvA for PNG all cl4_cf8_ordLF_bbox')
    plt.xlabel('Area (px)')
    plt.ylabel('Perimeter (edges)')
    plt.show()

    return 'temp'


def plot_pva_2():
    # df = pd.read_csv('/Users/emullen98/Downloads/fd_2025-11-14_png_cl4_cf8_ordLF_bbox_png_all_multi_314a4bc3_raw_areas_perims.csv')
    df = pd.read_csv('/Users/emullen98/Downloads/fd_2025-11-14_png_cl4_cf8_ordLF_bbox_png_argmax_multi_6487f80f_raw_areas_perims.csv')
    # df = pd.read_csv('/Users/emullen98/Downloads/fd_2025-11-14_png_cl4_cf4_ordLF_bbox_png_all_multi_b127c9d2_raw_areas_perims.csv')
    # df = pd.read_csv('/Users/emullen98/Downloads/fd_2025-11-14_png_cl4_cf4_ordLF_bbox_png_argmax_multi_0961b721_raw_areas_perims.csv')

    areas = df['area_px'].to_numpy()
    perims_hull = df['perim_hull_edge'].to_numpy()
    perims_acce = df['perim_accessible_edge'].to_numpy()

    plt.scatter(areas, perims_acce, label='accessible')
    plt.scatter(areas, perims_hull, label='hull')
    plt.legend()
    plt.loglog()
    plt.title('PvA for PNG argmax cl4_cf8_ordLF_bbox')
    plt.xlabel('Area (px)')
    plt.ylabel('Perimeter (edges)')
    plt.show()

    return 'temp'


# plot_pva_2()
plot_pva()
# plot_area()
# plot_perimeter()