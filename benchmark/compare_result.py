import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

from imm_handler import ImmDataset


target_dir = "/scratch/cluster_results"

with h5py.File(target_dir + '/B981_multitau_eigen.hdf', 'r') as f:
    a = f['/exchange/pixelSum'][()]
    a_mask = f['/xpcs/mask'][()].astype(np.int8)
    a = a * a_mask
    sqmap = f["/xpcs/sqmap"][()]
    sqlist = f["/xpcs/sqlist"][()]
    saxs1d_eigen = f["/exchange/partition-mean-total"][()][0]


with h5py.File(target_dir + '/B981_2_10k_star_dynamic_0p1Hz_Strain1.01mm_Ampl0.005mm_att5_Lq0_001_0001-0800.hdf', 'r') as f:
    b = f['/exchange/pixelSum'][()]
    b_mask = f['/xpcs/mask'][()].astype(np.int8)
    b = b * b_mask
    saxs1d_gpu = f["/exchange/partition-mean-total"][()][0]

# read raw data directly
imm = ImmDataset('/scratch/xpcs_data_raw/B981_2_10k_star_dynamic_0p1Hz_Strain1.01mm_Ampl0.005mm_att5_Lq0_001/B981_2_10k_star_dynamic_0p1Hz_Strain1.01mm_Ampl0.005mm_att5_Lq0_001_00001-00800.imm')
tot = np.zeros(imm.det_size, dtype=np.int64)
for n in range(imm.batch_num):
    x = imm[n].cpu().numpy().astype(np.int64).sum(axis=0)
    tot += x.reshape(imm.det_size)
tot = tot * a_mask
tot = tot / 1.0 / imm.frame_num

non_zero = np.nonzero(a_mask)
sl_v = slice(np.min(non_zero[0]), np.max(non_zero[0]) + 1)
sl_h = slice(np.min(non_zero[1]), np.max(non_zero[1]) + 1)

a = a[sl_v, sl_h]
b = b[sl_v, sl_h]
mask = a_mask[sl_v, sl_h]
tot = tot[sl_v, sl_h]

print(np.max(a), np.max(b), np.max(tot))
print(np.mean(a), np.mean(b), np.mean(tot))
sqmap = sqmap[sl_v, sl_h]

sqmap1d = sqmap.flatten()
tot1d = tot.flatten()
saxs1d_raw = np.bincount(sqmap1d, weights=tot1d)[1:]

count = np.bincount(sqmap1d)[1:]
valid_idx = count > 0
print(valid_idx.dtype, np.sum(valid_idx))
print("invalid_idx", np.nonzero(~valid_idx))
saxs1d_raw = saxs1d_raw[valid_idx] / count[valid_idx]

# print(saxs1d_eigen.shape)
# print(saxs1d_eigen[-20:])
# print(saxs1d_gpu[-20:])

xline = np.arange(saxs1d_eigen.size)
plt.semilogy(xline, saxs1d_eigen, 'rx-', ms=5, label='eigen', alpha=0.9)
plt.semilogy(xline, saxs1d_gpu, 'b+-', ms=5, label='gpu', alpha=0.9)
plt.semilogy(xline, saxs1d_raw, 'go-', alpha=0.5, ms=5, label='raw')
plt.legend()
plt.show()
plt.savefig('saxs1d_all.png', dpi=600)
plt.close()

# plt.plot(xline, saxs1d_eigen / saxs1d_raw - 1, 'rx-', label='eigen/raw - 1')
# plt.legend()
# plt.savefig('saxs1d_eigen.png', dpi=600)
# plt.close()
# 
# plt.plot(xline, saxs1d_gpu / saxs1d_raw - 1, 'bo-', label='gpu/raw - 1')
# plt.legend()
# plt.savefig('saxs1d_gpu.png', dpi=600)
# plt.close()
# saxs 2d comparison
# def find_vmin_vmax(diff):
#     vmin = np.min(diff)
#     vmax = np.max(diff)
#     max_diff = max(abs(vmin), vmax)
#     return -max_diff, max_diff
# 
# 
# diff = [a - b, tot - b, tot - a]
# # diff = [a / b - 1, np.divide tot / b - 1, tot /a - 1]
# label = ['eigen-gpu', 'raw-gpu', 'raw-eigen']
# 
# for n in range(3):
#     x = diff[n]
#     vmin, vmax = find_vmin_vmax(x)
#     plt.imshow(x, cmap=plt.get_cmap('seismic'), vmin=vmin, vmax=vmax)
#     plt.colorbar()
#     plt.title(label[n])
#     plt.savefig(label[n] + '.png', dpi=600)
#     plt.close()
# 