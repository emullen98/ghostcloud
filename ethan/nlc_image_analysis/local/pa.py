import numpy as np
import matplotlib.pyplot as plt
import common_code as cc

file_loc = '/Users/emullen98/Downloads/image/2012-12-30--06-08-02--109_id=211_pa_fill_thresh'

"""
DVS
"""

# for i in range(94, 255):
#     perims, areas = np.load(f'{file_loc}={i}.npy')
#     if len(areas[areas > 300]) > 50:
#         plt.scatter(areas, perims)
#
# plt.loglog()
# plt.show()

"""
Area CCDF
"""

# for i in range(94, 255):
#     perims, areas = np.load(f'{file_loc}={i}.npy')
#     if len(areas[areas > 300]) > 50:
#         hx, hy = cc.ccdf(areas)
#         plt.plot(hx, hy, '.')
#
# x, y = cc.linemaker(slope=-0.65, intercept=[1000, 0.002], xmin=100, xmax=10000)
# plt.plot(x, y, color='k', linestyle='dashed', label='area ccdf exp = -0.65')
# plt.legend()
# plt.loglog()
# plt.show()

"""
Perim CCDF
"""

for i in range(94, 255):
    perims, areas = np.load(f'{file_loc}={i}.npy')
    if len(perims[perims > 100]) > 50:
        hx, hy = cc.ccdf(perims)
        plt.plot(hx, hy, '.')

xp, yp = cc.linemaker(slope=-0.95, intercept=[1000, 0.0008], xmin=100, xmax=10000)
plt.plot(xp, yp, color='k', linestyle='dashed', label='perim ccdf exp = -0.95')
plt.legend()
plt.loglog()
plt.show()
