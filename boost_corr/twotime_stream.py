import torch
import logging
import numpy as np
logger = logging.getLogger(__name__)


def create_diagonal_index(N=512):
    m = torch.zeros((N, N), dtype=torch.int64)
    x = torch.hstack([torch.arange(N - 1, 0, -1), torch.arange(N)])

    for n in range(N):
        sl = slice(n, n + N)
        m[n] = x[sl]
    m = torch.flipud(m)
    return m


def compute_diagonal_average(index, weights):
    # index and weights must be both one d;
    tot = torch.bincount(index, weights=weights)
    pts = torch.bincount(index)
    return tot / pts


class TwotimeCorrelator():
    def __init__(self,
                 qinfo,
                 frame_num,
                 det_size=(1024, 512),
                 window=1024,
                 mask_crop=None,
                 device='cpu',
                 method='normal',
                 smooth='dqmap',
                 dtype=torch.float32) -> None:

        self.dq_idx = qinfo['dq_idx']
        self.dq_slc = qinfo['dq_slc']
        self.sq_idx = qinfo['sq_idx']
        self.sq_slc = qinfo['sq_slc']
        # self.dq_sq_map = qinfo['dq_sq_map']

        self.det_size = det_size
        self.pixel_num = self.det_size[0] * self.det_size[1]

        self.device = device
        self.cache = []
        self.frame_num = frame_num
        self.method = method
        self.smooth = smooth

        if mask_crop is None:
            mask_crop = torch.ones(det_size, device=self.device,
                                   dtype=torch.bool)
            mask_crop = mask_crop.reshape(-1)
        self.mask_crop = mask_crop

        arr_size = self.dq_slc[-1].stop
        self.arr_size = arr_size

        num_dq = len(self.dq_slc)
        if self.method == 'normal':
            shape = (frame_num, arr_size)
        elif self.method == 'window':
            shape = (window * 2, arr_size)
            self.c2 = torch.zeros((num_dq, frame_num, window), device=device)
            self.c2_ptr = 0
            self.c2_ptr_used = 0
            # self.cache_sum = torch.zeros((window * 2, num_q), device=device)

        self.cache = torch.zeros(shape, device=device, dtype=dtype)
        self.window = window
        self.cache_ptr = self.window

        self.frame_sum = None
        self.pixel_sum = None
        self.pixel_sum_curr = 0
        self.g2full = []
        self.g2partial = []
        # smooth data
        self.sdata = None
        self.c2_idx = 1
        self.current_frame = 0
        self.intt = torch.zeros(self.frame_num, device=self.device)
        self.curr_img = torch.zeros(self.det_size)

    def describe(self):
        logger.info('detector shape: ' + str(self.det_size))
        logger.info('frame number: ' + str(self.frame_num))
        logger.info('device: ' + str(self.device))
        logger.info('mask_crop: ' + str(self.mask_crop.shape))

    def reset(self):
        self.frame_sum = None
        self.pixel_sum = None
        self.pixel_sum_curr = 0
        self.g2full = []
        self.g2partial = []
        self.c2_idx = 1
        self.c2_ptr = 0
        self.c2_ptr_used = 0
        self.cache_ptr = self.window
        self.cache.zero_()
        self.current_frame = 0
        self.intt = torch.zeros(self.frame_num, device=self.device)

    def process(self, x):
        if self.method == 'normal':
            sz = x.shape[0]
            self.cache[self.cache_ptr:self.cache_ptr + sz] = x
            self.cache_ptr += sz
            # if self.cache_ptr == self.num_frames:
            #     self.calc_normal_twotime()
        elif self.method == 'window':
            self.process_window(x)

    def compute_average(self, num_part=10):
        # saxs
        self.pixel_sum_roi = torch.mean(self.cache, axis=0)
        self.pixel_sum = torch.zeros(self.pixel_num,
                                     dtype=torch.float32,
                                     device=self.device)
        self.pixel_sum[self.mask_crop] = self.pixel_sum_roi
        self.pixel_sum = self.pixel_sum.reshape(*self.det_size)

        pixel_sum_par = torch.zeros(num_part,
                                    self.pixel_num,
                                    dtype=torch.float32,
                                    device=self.device)

        frame_len = self.frame_num // num_part
        for n in range(num_part):
            sl = slice(n * frame_len, (n + 1) * frame_len)
            avg = torch.mean(self.cache[sl], axis=0)
            pixel_sum_par[n, self.mask_crop] = avg

        self.pixel_sum_par = pixel_sum_par.reshape(num_part, *self.det_size)

        frame_sum = torch.mean(self.cache, axis=1)
        taxis = torch.arange(self.frame_num, device=self.device)
        self.frame_sum = torch.stack([taxis, frame_sum])

    def compute_smooth_data(self, data):
        if self.smooth == 'sqmap':
            slc = self.sq_slc
        elif self.smooth == 'dqmap':
            slc = self.dq_slc
        else:
            raise ValueError('smooth method not supported')

        if data.ndim == 1:
            data = data.reshape(1, -1)

        data = data.T

        for n in range(len(slc)):
            avg = data[slc[n]].mean(dim=0)
            # get rid of the zeros
            avg[avg <= 0] = 1
            data[slc[n]] /= avg

        return data.T

    def process_window(self, x):

        if self.cache_ptr >= 2 * self.window:
            self.cache_ptr = self.window
            self.cache = torch.roll(self.cache, shifts=-self.window, dims=0)

        x0 = x[0][self.mask_crop]
        self.curr_img = x0
        self.pixel_sum_curr = self.pixel_sum_curr + x0.long()

        x0 = x0.float()
        self.intt[self.current_frame] = torch.sum(x0)
        x0_norm = self.compute_smooth_data(x0)[0]

        self.cache[self.cache_ptr] = x0_norm
        self.cache_ptr += 1
        self.current_frame += 1

        # if self.cache_ptr >= self.window:
        if True:
            sl = slice(self.cache_ptr - self.window, self.cache_ptr)
            wf = self.cache[sl]
            # wf = self.compute_smooth_data(wf)
            for idx, roi in enumerate(self.dq_slc):
                a = wf[:, roi]
                b = x0_norm[roi]
                down = torch.sum(a, dim=1) * torch.sum(b)
                down[down == 0] = 1
                # print(a.shape, b.shape, a_sum.shape, b_sum.shape)
                corr = torch.matmul(a, b) * len(b) / down 
                self.c2[idx, self.c2_ptr] = corr
            self.c2_ptr += 1

        if x.shape[0] <= 1:
            return
        else:
            self.process_window(x[1:])
            return

    def post_processing(self, **kwargs):
        self.cache = self.cache.float()
        # saxs2d
        self.compute_average()
        # smooth data
        self.cache = self.compute_smooth_data(self.cache)

    def get_results(self):
        tline = torch.arange(self.intt.shape[0], device=self.intt.device)
        intt = torch.vstack([tline, self.intt.float()])

        # rescale the average and add total flux to IP and IF
        self.saxs_2d = torch.unsqueeze(self.pixel_sum_curr, 0)
        self.saxs_2d_par = torch.unsqueeze(self.saxs_2d, 0)

        sl = slice(self.c2_ptr_used, self.c2_ptr)

        output = {
            'intt': intt.float()[:, 0:self.current_frame],
            'saxs2d': self.saxs_2d.float(),
            'saxs2d_par': self.saxs_2d_par.float(),
            'C2': self.c2[:, sl].float(),
            'mask_crop': self.mask_crop,
            'saxs_2d_single': self.curr_img.float(),
            'shape': np.array(self.c2[:, sl].shape),
            'slice': np.array([self.c2_ptr_used, self.c2_ptr])
        }
        self.c2_ptr_used = self.c2_ptr

        return output

    def get_twotime_result(self, **kwargs):
        for c2 in self.calc_normal_twotime(**kwargs):
            yield {f'/exchange/C2T_all/g2_{self.c2_idx:05d}': c2}
            self.c2_idx += 1

        # get the average and g2full/partials
        self.g2full = torch.stack(self.g2full).swapaxes(0, 1)
        self.g2partial = torch.stack(self.g2partial).permute(2, 1, 0)
        results = {
            '/exchange/frameSum': self.frame_sum,
            '/exchange/g2full': self.g2full,
            '/exchange/g2partials': self.g2partial,
            '/exchange/pixelSum': self.pixel_sum,
            '/xpcs/qphi_bin_to_process': self.dq_idx
        }

        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                results[k] = v.cpu().numpy()

        yield results

    def get_saxs(self):
        num_par = self.pixel_sum_par.shape[0]
        saxs = {
            'saxs2d':
            self.pixel_sum.reshape(-1)[self.mask_crop],
            'saxs2d_par':
            self.pixel_sum_par.reshape(num_par, -1)[:, self.mask_crop],
            'mask_crop':
            self.mask_crop,
        }
        return saxs

    def calc_normal_twotime(self, num_partials=5):
        diag_mat = create_diagonal_index(self.frame_num).to(self.device)
        diag_mat_1d = diag_mat.reshape(-1)
        partial_len = self.frame_num // num_partials
        diag_mat_1d_p = diag_mat[0:partial_len, 0:partial_len].reshape(-1)

        # triu_idx = torch.tril_indices(self.num_frames, self.num_frames)

        for n, q in enumerate(self.dq_idx):
            wf = self.cache[:, self.dq_slc[n]]
            if self.sdata is not None:
                wf = wf / self.sdata[self.dq_slc[n]]

            aa = torch.sum(wf, dim=1)
            aa[aa <= 0] = 1
            aa_inverse = 1.0 / aa

            ab = torch.matmul(wf, wf.T)
            c2 = ab * aa_inverse
            c2 = c2.T * aa_inverse * wf.shape[1]

            g2full = compute_diagonal_average(diag_mat_1d, c2.reshape(-1))
            self.g2full.append(g2full)

            g2partial = []

            for idx in range(num_partials):
                sl = slice(idx * partial_len, (idx + 1) * partial_len)
                temp = compute_diagonal_average(diag_mat_1d_p,
                                                c2[sl, sl].reshape(-1))
                g2partial.append(temp)
            g2partial = torch.stack(g2partial)
            self.g2partial.append(g2partial)

            yield (torch.triu(c2)).cpu().numpy()
            # yield c2[triu_idx[0], triu_idx[1]].cpu().numpy()


if __name__ == '__main__':
    pass
