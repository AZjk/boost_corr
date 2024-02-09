import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse


def read_result(fname):
    keys = {
        'g2': '/exchange/norm-0-g2',
        'g2_err': '/exchange/norm-0-stderr',
        'G2': '/exchange/G2IPIF',
        'mask': '/xpcs/mask',
        'tau': '/exchange/tau'
    }

    contents = {}
    with h5py.File(fname, 'r') as f:
        for k, val in keys.items():
            contents[k] = f[val][()]

    # squeeze axis
    contents['tau'] = contents['tau'][0]

    # apply mask
    contents['G2'] = contents['G2'] * contents['mask']

    # remove mask
    contents.pop('mask')

    return contents


def get_error(x_diff):
    max_err = np.nanmax(np.abs(x_diff))
    mean_err = np.nanmean(np.abs(x_diff))
    median_err = np.nanmedian(np.abs(x_diff))
    return max_err, mean_err, median_err


def main(fname1, fname2, mode='all'):
    data1 = read_result(fname1)
    data2 = read_result(fname2)
    
    # fig, ax = plt.subplots(6, 3, figsize=(11, 8.6))
    # ax = ax.flatten()

    # for n in range(len(ax)):
    #     ax[n].errorbar(data1['tau'], data1['g2'][:, n], yerr=data1['g2_err'][:, n], alpha=0.7, fmt='o', capsize=3, label='f1')
    #     ax[n].errorbar(data2['tau'], data2['g2'][:, n], yerr=data2['g2_err'][:, n], alpha=0.7, fmt='x', capsize=3, label='f2')
    #     # ax[n].semilogx(data1['tau'], data1['g2'][:, n], 'ro')
    #     # ax[n].semilogx(data2['tau'], data2['g2'][:, n], 'bx')
    #     ax[n].set_xscale('log')
    #     # ax[n].legend()
    # plt.tight_layout()
    # plt.show()

    fields = ['g2', 'g2_err', 'G2']

    print('compare files')
    print('    file1:', fname1)
    print('    file2:', fname2)

    print('******************** data shape and type  ***********************')
    for f in fields:
        print(f'{f:<8}:', data1[f].shape, data1[f].dtype)

    print('******************** max absolute difference ********************')
    for f in fields:
        x_diff = data1[f] - data2[f]
        max_err, mean_err, median_err = get_error(x_diff)
        print(f'{f:<8}:', f'\t {max_err=:.8e}, \t {mean_err=:.8e}, \t {median_err=:.8e}')

    print('******************** max relative difference ********************')
    for f in fields:
        norm_factor = 0.5 * (data1[f] + data2[f])
        norm_factor[norm_factor == 0] = np.nan 
        x_diff = (data1[f] - data2[f]) / norm_factor

        max_err, mean_err, median_err = get_error(x_diff)
        print(f'{f:<8}:', f'\t {max_err=:.8e}, \t {mean_err=:.8e}, \t {median_err=:.8e}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('fname1', type=str, help='1st file to compare')
    ap.add_argument('fname2', type=str, help='2nd file to compare')
    # ap.add_argument('--mode', type=str, choices=['g2', 'G2', 'all'])

    args = ap.parse_args()
    kwargs = vars(args)
    main(**kwargs)
    # main('/scratch/MQICHU/aps-u/cluster_results/E017_CeramicGlass_L2Mq0_060C_att00_001_0001-1024_01.hdf',
    #      '/scratch/MQICHU/aps-u/cluster_results/E017_CeramicGlass_L2Mq0_060C_att00_001_0001-1024_06.hdf',
    #      mode='all')
    