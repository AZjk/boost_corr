import math
import numpy as np
from scipy.sparse import csr_matrix as sp_csr_matrix
import hashlib
import torch


try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
except Exception:
    # print('cannot import cupy')
    pass


def stream_simulated_data(frame_num, det_size=None, method='cupy',
                          batch_size=4096, device_id=0):

    pixel_num = det_size[0] * det_size[1]
    assert method in ['cupy', 'torch', 'numpy']
    if method == 'cupy':
        ones = cp.ones
    elif method == 'torch':
        if device_id >= 0:
            dev = torch.device('cuda:%d' % device_id)
        else:
            # if device_id < 0; then use cpu instead;
            dev = torch.device('cpu')
        ones = lambda shape: torch.ones(shape, device=dev)
    else:
        ones = np.ones

    for n in range((frame_num + batch_size - 1) // batch_size):
        sz = min(frame_num, (n + 1) * batch_size) - batch_size * n
        yield ones((sz, pixel_num))

    return


def convert_sparse(a):
    output = np.zeros(shape=(3, a.size), dtype=np.uint32)
    # index
    output[0] = ((a >> 16) & (2 ** 21 - 1)).astype(np.uint32)
    # frame
    output[1] = (a >> 40).astype(np.uint32)
    # count
    output[2] = (a & (2 ** 12 - 1)).astype(np.uint8)
    return output


def stream_sparse_data(fname, det_size=None, method=None, dtype=np.float64,
                       batch_size=4096, f_count=64*1024*1024):
    """
    stream sparse data that follows Rigaku format; instead of read the whole
    file into RAM, the function read chunks of the data and convert it to a
    sparse matrix form, till all data is read.
    """
    pixel_num = det_size[0] * det_size[1]

    buff = np.array([[], [], []], dtype=np.uint32)
    curr_index = 0

    with open(fname, 'r') as f:
        while True:
            a = np.fromfile(f, dtype=np.uint64, count=f_count)
            if a.size == 0:
                break
            new_buff = convert_sparse(a)
            buff = np.hstack([buff, new_buff])
            idx = np.searchsorted(buff[1], (curr_index + 1) * batch_size)
            if idx < buff.shape[1]:
                spm, buff, curr_index = cut_data(buff, idx, batch_size,
                                                 curr_index, pixel_num, dtype)
                yield spm

    while buff.shape[1] > 0:
        idx = np.searchsorted(buff[1], (curr_index + 1) * batch_size)
        spm, buff, curr_index = cut_data(buff, idx, batch_size, curr_index,
                                         pixel_num, dtype)
        yield spm


def cut_data(buff, idx, batch_size, curr_index, pixel_num, dtype):

    data = buff[:, :idx]

    # shift the index
    data[1] -= curr_index * batch_size
    curr_index += 1

    if idx == buff.shape[1]:
        new_sz = np.max(buff[1]) + 1
    else:
        new_sz = batch_size

    spm = sp_csr_matrix((data[2], (data[1], data[0])),
                        shape=(new_sz, pixel_num), dtype=dtype)

    buff = buff[:, idx:]
    return spm, buff, curr_index


def read_sparse_data(fname, det_size=None, method=None, dtype=np.float64):
    """
    read sparse data that follows Rigaku format
    """
    with open(fname, 'r') as f:
        a = np.fromfile(f, dtype=np.uint64)
        d = convert_sparse(a)

    index, frame, count = d[0], d[1], d[2]
    frame_num = frame[-1] + 1

    assert(method in [None, 'numpy', 'cupy'])

    if method is None:
        # return raw data;
        return index, count, frame, frame_num
    else:
        pixel_num = det_size[0] * det_size[1]
        # scipy csr matrix is much faster to initialize than cupy's
        spm = sp_csr_matrix((count, (frame, index)),
                            shape=(frame_num, pixel_num),
                            dtype=dtype)

        # convert to cupy csr matrix
        if method == 'cupy':
            spm = cp_csr_matrix(spm)

        return spm, frame_num


def gen_tau_bin(frame_num, dpl=4, max_level=60):
    """
    generate tau and fold list according to the frame number
    """
    def worker():
        for n in range(dpl):
            yield n + 1, 0
        level = 0
        while True:
            scl = 2 ** level
            for x in range(dpl + 1, dpl * 2 + 1):
                if x * scl >= frame_num // scl * scl:
                    return
                yield x * scl, level
            level += 1
            # limit the levels to max_level;
            if level > max_level:
                return

    tau_bin = []
    for x, y in worker():
        tau_bin.append([x, y])

    # make it a 2-column array
    tau_bin = np.array(tau_bin, dtype=np.int).T
    return tau_bin


def sort_tau_bin(tau_bin, frame_num):
    """
    sort the tau_bin object, so tau is relative in each level;
    """
    tau_num = tau_bin.shape[1]
    # rescale tau for each level
    tau_max = np.max(tau_bin[0] // (2 ** tau_bin[1]))

    levels = list(np.unique(tau_bin[1]))
    levels_num = len(levels)
    tau_in_level = {}

    offset = 0
    for level in levels:
        scl = 2 ** level
        tau_list = tau_bin[0][tau_bin[1] == level] // scl

        # tau_idx is used to index the result;
        tau_idx = range(offset, offset + len(tau_list))

        # avg_len is used to compute the average G2, IP, IF
        avg_len = frame_num // scl - tau_list

        tau_in_level[level] = list(zip(tau_list, tau_idx, avg_len))
        offset += len(tau_list)

    assert(tau_num == offset)
    return tau_max, levels, tau_in_level


def multi_tau(data, tau_bin, method='numpy'):
    assert(method in ['numpy', 'cupy'])
    frame_num, pixel_num = data.shape
    tau_max, levels, tau_in_level = sort_tau_bin(tau_bin, frame_num)
    tau_num = tau_bin.shape[1]
    levels_num = len(levels)

    if method == 'numpy':
        xp = np
        mul = sp_csr_matrix.multiply
    else:
        xp = cp
        mul = cp_csr_matrix.multiply

    # intensity vs t for all frames
    intt = data.mean(axis=1)

    G2 = xp.zeros([tau_num, 3, pixel_num], dtype=data.dtype)
    residue = xp.zeros((levels_num, pixel_num), dtype=data.dtype)

    for level in levels:
        for tau, tid, _ in tau_in_level[level]:
            # the [0] part is required for cupy compatibility
            G2[tid, 0] = mul(data[tau:], data[:-tau]).mean(axis=0)[0]
            G2[tid, 1] = -1 * data[0:tau].sum(axis=0)[0]
            G2[tid, 2] = -1 * data[-tau:].sum(axis=0)[0]

        eff_len = data.shape[0] // 2 * 2
        if eff_len * 2 < data.shape[0]:
            residue[level] = data[-1, :].todense()

        data = (data[0:eff_len:2] + data[1:eff_len:2]) / 2.0

        # warnning: some odd frames are dropped for saxs_2d_par; for stability
        # information, intt is a better indicator;
        if 8 < data.shape[0] <= 16:
            saxs_2d_par = data.todense()

    # accumulate flux
    tot_flux = data.sum(axis=0)
    for level in levels[::-1]:
        tot_flux = tot_flux * 2 + residue[level]
        for tau, tid, avg_len in tau_in_level[level]:
            G2[tid, 1:3] += tot_flux
            G2[tid, 1:3] /= avg_len

    # transfer data from GPU to host;
    if method == 'cupy':
        G2 = G2.get()
        tot_flux = tot_flux.get()
        intt = intt.get()
        saxs_2d_par = saxs_2d_par.get()

    saxs_2d = tot_flux

    return G2, np.squeeze(saxs_2d), saxs_2d_par, np.squeeze(intt)


def is_power_two(num):
    # return true for 1, 2, 4, 8, 16 ...
    while num > 2:
        if num % 2 == 1:
            return False
        num //= 2
    return num == 2 or num == 1


def hash_numpy(x, scale):
    x = x.flatten()
    # use ceil instead of array to avoid clipping
    x = np.ceil(x / np.max(x) * scale).astype(np.int)
    val = hashlib.sha512(x.tobytes()).hexdigest()
    return val


def compute_saxs_1d(saxs_2d, saxs_2d_par, det, output_dict=None):
    """
    convert 2-dimensional saxs data to 1D;
    """
    saxs_1d = np.zeros((1, det.ql_sta.size))
    # saxs_2d = saxs_2d # * det.mask

    stab = np.zeros((saxs_2d_par.shape[0], det.ql_sta.size))

    for n in range(det.ql_sta.size):
        qidx = (det.sqmap == n + 1)
        if np.sum(qidx) > 0:
            saxs_1d[0, n] = np.mean(saxs_2d[qidx])
            stab[:, n] = np.mean(saxs_2d_par[:, qidx], axis=1)

    if output_dict is None:
        return saxs_1d, stab
    else:
        output_dict['saxs_1d'] = saxs_1d
        output_dict['Iqp'] = stab


def nonzero_crop(img):
    """
    computes the slice in vertical and horizontal direction to crop the nonzero
        regions of the input array img.
    """
    assert isinstance(img, np.ndarray), 'img must be a numpy.ndarray'
    assert img.ndim == 2, 'img has be a two-dimensional numpy.ndarray'
    idx = np.nonzero(img)
    sl_v = slice(np.min(idx[0]), np.max(idx[0]) + 1)
    sl_h = slice(np.min(idx[1]), np.max(idx[1]) + 1)
    return sl_v, sl_h


def compute_g2(G2, det, output_dict=None, rot_img=False):
    mask = det.mask
    sl_v, sl_h = nonzero_crop(mask)
    mask = mask[sl_v, sl_h]
    G2 = G2[:, :, sl_v, sl_h]
    qmap_sta = det.sqmap[sl_v, sl_h] * mask
    qmap_dyn = det.dqmap[sl_v, sl_h] * mask

    if rot_img:
        qmap_sta = det.sqmap.T
        qmap_dyn = det.dqmap.T

    assert G2.ndim == 4, "G2 must be a 4d array"
    assert qmap_sta.ndim == 2, "qmap sta must be a 2D map"
    assert qmap_dyn.ndim == 2, "qmap dyn must be a 2D map"

    ql_sta_dim = np.max(qmap_sta)
    ql_dyn_dim = np.max(qmap_dyn)

    G2q = np.zeros([G2.shape[0], 3, ql_sta_dim])

    g2 = np.zeros([G2.shape[0], ql_dyn_dim])
    g2_err = np.zeros_like(g2)

    IP_IF = np.multiply(G2[:, 1], G2[:, 2])
    IP_IF = (np.where(IP_IF != 0, IP_IF, 10000))
    g2_per_pixel = np.divide(G2[:, 0], IP_IF)

    for ii in range(ql_sta_dim):
        idx_corr = (qmap_sta == ii+1)
        G2q[..., ii] = np.mean(G2[..., idx_corr], axis=-1)

    Isymmq = np.multiply(G2q[:, 1], G2q[:, 2])
    Isymmq = (np.where(Isymmq != 0, Isymmq, 10000))

    for ii in range(ql_dyn_dim):
        idx_corr = (qmap_dyn == ii+1)
        temp = g2_per_pixel[:, idx_corr]
        g2_err[:, ii] = np.std(temp, axis=-1) / np.sqrt(temp.shape[1])

    for ii in range(ql_dyn_dim):
        roi = qmap_sta[np.where(qmap_dyn == ii+1)]
        tmp_G2q = G2q[:, 0, (np.min(roi) - 1): (np.amax(roi) - 1)]
        tmpIsymmq = Isymmq[:, (np.min(roi) - 1): (np.max(roi) - 1)]
        g2[:, ii] = np.mean(np.divide(tmp_G2q, tmpIsymmq), axis=1)

    if output_dict is None:
        return g2, g2_err
    else:
        output_dict['g2'] = g2
        output_dict['g2_err'] = g2_err
        return output_dict
