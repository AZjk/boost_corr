import h5py
import logging
import numpy as np
import torch
import sys
import os


logger = logging.getLogger(__name__)


key_map = {
    "dqmap": "/data/dynamicMap",
    "sqmap": "/data/staticMap",
    "mask": "/data/mask",
    "sphilist": "/data/sphival",
    "dphilist": "/data/dphival"
}


def find_bin_count(qmap, minlength=None):
    count = np.bincount(qmap.ravel(), minlength=minlength)[1:]
    count = count.astype(np.int32)
    nan_idx = (count == 0)
    count[nan_idx] = -1
    return count, nan_idx


def average(img, qmap, size=None, count=None):
    qmap = qmap.view(-1)

    if count is None:
        count = torch.bincount(qmap, minlength=size)[1:]
        count[count == 0] = 1

    if img.ndim == 1:
        img = torch.unsqueeze(img, dim=0)

    # the first dimensions
    orignal_shape = img.shape[0:-1]
    img = img.reshape(-1, img.shape[-1])

    avg_list = []
    for n in range(img.shape[0]):
        sum_value = torch.bincount(qmap, weights=img[n], minlength=size)[1:]
        avg_list.append(sum_value)
    avg = torch.vstack(avg_list).reshape(*orignal_shape, -1).float() / count
    return avg


class XpcsQPartitionMap(object):
    def __init__(self, qmap_fname, flag_fix=True, flag_sort=False,
                 dq_selection=None, device='cpu') -> None:
        super().__init__()
        self.fname = qmap_fname
        self.device = device
        self.masked_ratio = 1.0
        self.masked_pixels = 0
        self.dq_dim = 0
        self.sq_dim = 0
        self.dqmap = None
        self.sqmap = None
        self.mask = None
        self.det_size = None
        self.qinfo = None
        self.flag_sort = flag_sort
        self.load(flag_fix)
        self.group_qmap(dq_selection)

    def group_qmap(self, dq_selection=None):
        scount, snan_idx = find_bin_count(self.sqmap, self.sq_dim + 1)
        if self.flag_sort:
            # select and sort the qmap; good for twotime and multitau
            mask_idx_1d, qinfo = self.select_sort(dq_selection)
        else:
            # don't sort the qmap; good for multitau
            mask_idx_1d = np.where(self.mask.reshape(-1) == 1)[0]
            qinfo = None

        info_np = {
            "mask_idx_1d": mask_idx_1d,
            "sqmap_full": self.sqmap.reshape(-1),
            "dqmap_crop": self.dqmap.reshape(-1)[mask_idx_1d],
            "sqmap_crop": self.sqmap.reshape(-1)[mask_idx_1d],
            "scount": scount,
            "snan_idx": snan_idx,
            # "dcount": dcount,
            # "dnan_idx": dnan_idx,
        }

        info = {}
        for k, v in info_np.items():
            info[k] = torch.tensor(v).to(self.device)

        self.info_np = info_np
        self.info = info
        self.qinfo = qinfo

    def load(self, flag_fix=False):
        values = {}
        with h5py.File(self.fname, "r") as f:
            for key, real_key in key_map.items():
                values[key] = np.squeeze(f[real_key][()])

        self.__dict__.update(values)
        self.mask = self.mask.astype(bool)
        self.dqmap = self.dqmap.astype(np.int32)
        self.sqmap = self.sqmap.astype(np.int32)
        if flag_fix:
            self.check_fix_qmap()
        self.det_size = self.mask.shape
        self.dq_dim = self.dphilist.size
        self.sq_dim = self.sphilist.size
        self.masked_pixels = int(np.sum(self.mask == 1))
        self.masked_ratio = self.masked_pixels / self.mask.size

    def update_file(self):
        logger.warning(f'update fixed qmap in file: [{self.fname}]')
        # update the qmap after fixing sqmap/dqmap inconsistency
        with h5py.File(self.fname, "a") as f:
            del f[key_map['dqmap']]
            f[key_map['dqmap']] = self.dqmap.astype(np.uint32)

    def update_rotation(self, det_size):
        if self.dqmap.shape != det_size:
            logger.warning(
                "qmap is rotated 90 deg to match the detctor oriention.")
            self.dqmap = np.swapaxes(self.dqmap, 0, 1)
            self.sqmap = np.swapaxes(self.sqmap, 0, 1)
            self.mask = np.swapaxes(self.mask, 0, 1)
            self.det_size = self.mask.shape
            assert self.det_size == det_size, \
                    "The shape of QMap does not match the raw data shape"
            # need redo the preprocessing
            self.group_qmap()
            return True
        return False

    def get_mask_crop(self):
        return self.info['mask_idx_1d']

    def check_fix_qmap(self):
        flag = True
        sq_list = np.sort(np.unique(self.sqmap[self.sqmap > 0]))
        for sq in sq_list:
            sq_roi = self.sqmap == sq
            dq = np.sort(np.unique(self.dqmap[sq_roi]))
            assert len(dq) in (1, 2), 'only support sqmap belongs up to 2 bins'
            if len(dq) > 1:  # only works for 2
                flag = False
                idx_0 = np.logical_and(self.dqmap == dq[0], sq_roi)
                logger.warning(f'qmap inconsistent at sq: {sq} <-> {dq}: dq')
                if np.sum(idx_0) < len(sq_roi) // 2:
                    self.dqmap[idx_0] = dq[1]
                else:
                    idx_1 = np.logical_and(self.dqmap == dq[1], sq_roi)
                    self.dqmap[idx_1] = dq[0]
        if flag:
            logger.info('sqmap/dqmap are consistent.')
        return flag

    def select_sort(self, dq_selection=None):
        """
        group the sqmap and dqmap so that the pixels are continuous in memory
        """
        dqmap = self.dqmap.ravel()
        sqmap = self.sqmap.ravel()

        if dq_selection is None:
            dq_selection = np.unique(dqmap[dqmap > 0])
        dq_selection.sort()

        info = {"dq_slc": [], "sq_slc": [], "dq_idx": [], "sq_idx": [],
                "dq_sq_map": {}}

        q_roi = []
        dq_start = 0
        for dq in dq_selection:
            # nonzero return a tuple; (np.ndarray)
            roi = np.nonzero(dqmap == dq)[0]
            if np.sum(roi) == 0:
                continue
            sq_list = np.sort(np.unique(sqmap[roi]))
            info['dq_sq_map'][dq] = []

            sq_start = dq_start
            for sq in sq_list:
                sq_roi = np.nonzero(sqmap == sq)[0]
                if np.sum(sq_roi) == 0:
                    continue
                q_roi.append(sq_roi)
                info['sq_slc'].append(slice(sq_start, sq_start + len(sq_roi)))
                info['sq_idx'].append(sq)
                info['dq_sq_map'][dq].append(sq)
                sq_start += len(sq_roi)

            info['dq_idx'].append(dq)
            info['dq_slc'].append(slice(dq_start, dq_start + len(roi)))
            dq_start += len(roi)
            assert dq_start == sq_start, "dqmap and sqmap must be consistent"

        mask_idx_1d_sort = np.hstack(q_roi)
        dq_idx = info['dq_idx']
        total_dq, dq_min, dq_max = len(dq_idx), min(dq_idx), max(dq_idx)
        logger.info(f'total dq: {total_dq}. min - max: {dq_min} - {dq_max}')
        return mask_idx_1d_sort, info

    def normalize_sqmap(self, img, flag_crop, apply_nan=True):
        scount = self.info['scount']
        if flag_crop:
            sqmap = self.info['sqmap_crop']
        else:
            sqmap = self.info['sqmap_full']
        result = average(img, sqmap, self.sq_dim + 1, scount)

        if apply_nan:
            snan_idx = self.info['snan_idx']
            result = result[..., ~snan_idx]
        # result = result.cpu().numpy()
        return result

    def recover_dimension(self, saxs2d, flag_crop):
        if flag_crop:
            pixel_num = self.det_size[0] * self.det_size[1]
            full_img = torch.zeros(pixel_num,
                                   dtype=torch.float32,
                                   device=saxs2d.device)
            full_img[self.info['mask_idx_1d']] = saxs2d
        else:
            full_img = saxs2d

        full_img = full_img.reshape(1, *self.det_size)
        return full_img

    def normalize_data(self, res, save_G2=False):
        flag_crop = res['mask_crop'] is not None
        saxs1d = self.normalize_sqmap(res["saxs2d"], flag_crop)
        # make saxs2d has ndim of 2 instead of 3.
        saxs2d = self.recover_dimension(res['saxs2d'], flag_crop)[0]
        saxs1d_par = self.normalize_sqmap(res["saxs2d_par"], flag_crop)
        g2, g2_err = self.compute_g2(res["G2"], flag_crop)
        output_dir = {
            'saxs_2d': saxs2d,
            'saxs_1d': saxs1d,
            'Iqp': saxs1d_par,
            'Int_t': res['intt'],
            'tau': res['tau'],
            'g2': g2,
            'g2_err': g2_err
        }

        if save_G2:
            G2 = res["G2"].reshape(-1, res["G2"].shape[-1])
            if flag_crop:
                pixel_num = self.det_size[0] * self.det_size[1]
                value = torch.zeros(G2.shape[0], pixel_num,
                                    dtype=torch.float32,
                                    device=G2.device)
                value[:, self.info['mask_idx_1d']] = G2
            else:
                value = G2

            # the final shape of G2IPIF is (num_tau, 3, det_row, det_col)
            output_dir['G2IPIF'] = value.reshape(-1, 3, *self.det_size)

        for k, v in output_dir.items():
            if isinstance(v, torch.Tensor):
                output_dir[k] = v.float().cpu().numpy()
        return output_dir

    def normalize_saxs(self, res):
        flag_crop = res['mask_crop'] is not None
        if res["saxs2d"].shape == self.det_size:
            flag_crop = False
        saxs1d = self.normalize_sqmap(res["saxs2d"], flag_crop)
        saxs2d = self.recover_dimension(res['saxs2d'], flag_crop)
        saxs1d_par = self.normalize_sqmap(res["saxs2d_par"], flag_crop)
        output_dir = {
            'saxs_2d': saxs2d,
            'saxs_1d': saxs1d,
            'Iqp': saxs1d_par,
        }
        for k, v in output_dir.items():
            if isinstance(v, torch.Tensor):
                output_dir[k] = v.float().cpu().numpy()
        return output_dir

    def compute_g2(self, G2, flag_crop):
        if not flag_crop:
            G2 = G2[..., self.get_mask_crop()]
        flag_crop = True

        IP_IF = G2[:, 1] * G2[:, 2]
        IP_IF[IP_IF == 0] = 1.0e8
        g2_pixel = G2[:, 0] / IP_IF

        G2_sq = self.normalize_sqmap(G2, flag_crop, apply_nan=False)
        IP_IF_sq = G2_sq[:, 1] * G2_sq[:, 2]
        IP_IF_sq[IP_IF_sq == 0] = 1.0e8
        g2_sq = G2_sq[:, 0] / IP_IF_sq

        g2 = torch.zeros(G2.shape[0], self.dq_dim, device=G2.device)
        g2_err = torch.zeros_like(g2)

        dqmap = self.info['dqmap_crop']
        sqmap = self.info['sqmap_crop']

        for idx in range(1, self.dq_dim + 1):
            roi_dq = (dqmap == idx)
            if torch.sum(roi_dq) == 0:
                continue
            temp = g2_pixel[:, roi_dq]
            g2_err[:,
                   idx - 1] = torch.std(temp, axis=1) / np.sqrt(temp.shape[1])
            # the sqmap index - 1 gives the index in the G2q
            # roi = sqmap[idx_corr].long().unique() - 1
            # [1, 2, 3, 4]
            roi_sq = sqmap[roi_dq].long().unique() - 1
            g2[:, idx - 1] = torch.mean(g2_sq[:, roi_sq], axis=1)

        return g2, g2_err


def check_and_fix_qmap(fname):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    qpm = XpcsQPartitionMap(fname, flag_fix=False)
    flag = qpm.check_fix_qmap()
    if not flag:
        choise = input('update the qmap file? [Y/n] ')
        if choise == 'Y':
            qpm.update_file()
    else:
        _, info = qpm.select_sort()
        for k, v in info['dq_sq_map'].items():
            logger.info(f'dq: {k} <-> {v}: sq')
        logger.info(f'[{fname}] is consistent')


def test():
    import time
    t0 = time.perf_counter()
    # "/scratch/xpcs_data_raw/qmap/foster202110_qmap_RubberDonut_Lq0_S180_18_D18_18.h5"
    xpm = XpcsQPartitionMap(
        "/scratch/xpcs_data_raw/qmap/qzhang20191006_qmap_Rigaku_S270_D27_log.h5",
        device='cuda:0')
    print('time: ', round(time.perf_counter() - t0, 3))
    print(xpm.sqmap.shape)
    # print(np.sum(xpm.dcount))
    print(xpm.mask.dtype)
    print(xpm.masked_ratio)
    print(xpm.masked_pixels)
    import matplotlib.pyplot as plt
    plt.imshow(xpm.sqmap)
    plt.show()
    # xpm.update_rotation((1556, 516))
    # print(xpm.det_size)
    # for k, v in xpm.info_torch.items():
    #     print(k, v.shape)


if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        check_and_fix_qmap(sys.argv[1])
    else:
        print('Usage: check_fix_qmap some_qmap.hdf')
