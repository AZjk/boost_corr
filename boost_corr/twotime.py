import torch
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
import logging


logger = logging.getLogger(__name__)


def create_diagonal_index(N=512, device='cpu'):
    """
    create a matrix of indices for diagonal averaging
    :param N: size of the matrix
    :return: NxN matrix with indices for diagonal averaging
    Example:
    >>> create_diagonal_index(5)
    tensor([[0, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0]])
    """
    indices = torch.arange(N, device=device)
    diag_mat = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
    return diag_mat


def compute_diagonal_average(index, weights):
    """
    compute the diagonal average of a matrix using precomputed indices and weights
    :param index:  matrix with indices for diagonal averaging
    :param weights: NxN matrix with weights for diagonal averaging
    """
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
                 dtype=torch.float32) -> None:

        self.dq_idx = qinfo['dq_idx']
        self.dq_slc = qinfo['dq_slc']
        self.sq_idx = qinfo['sq_idx']
        self.sq_slc = qinfo['sq_slc']
        # self.dq_sq_map = qinfo['dq_sq_map']

        self.det_size = det_size
        self.pixel_num = self.det_size[0] * self.det_size[1]

        self.device = device
        self.frame_num = frame_num
        self.method = method

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
            self.c2 = torch.zeros((num_dq, frame_num - window + 1, window),
                                  device=device)
            self.c2_ptr = 0
            # self.cache_sum = torch.zeros((window * 2, num_q), device=device)

        self.cache = torch.zeros(shape, device=device, dtype=dtype)
        self.cache_ptr = 0

        self.window = window

        self.frame_sum = None
        self.pixel_sum = None
        self.g2full = []
        self.g2partial = []
        # smooth data
        self.sdata = None
        self.c2_idx = 1

    def process(self, x):
        if self.method == 'normal':
            sz = x.shape[0]
            self.cache[self.cache_ptr:self.cache_ptr + sz] = x
            self.cache_ptr += sz
            # if self.cache_ptr == self.num_frames:
            #     self.calc_normal_twotime()
        elif self.method == 'window':
            self.process_window(x)
    
    def process_dataset(self,
                        ds,
                        verbose=True,
                        use_loader=False,
                        num_workers=16):
        if not use_loader:
            xrange = trange if verbose else range
            for n in xrange(len(ds)):
                self.process(ds[n])
        else:
            # only pin memory if using GPU for computing
            pin_memory = (self.device != 'cpu')
            dl = DataLoader(ds,
                            batch_size=None,
                            pin_memory=pin_memory,
                            num_workers=num_workers,
                            prefetch_factor=4)
            logger.info(f'using {num_workers} workers to load data')
            
            if verbose:
                container = tqdm(dl, total=len(ds), desc='progress')
            else:
                container = dl

            for x in container:
                x = x.to(self.device, non_blocking=True)
                self.process(x)

    def compute_average(self, num_part=10):
        pixel_sum_par = torch.zeros(num_part,
                                    self.pixel_num,
                                    dtype=torch.float32,
                                    device=self.device)
        self.pixel_sum = torch.zeros_like(pixel_sum_par[0])

        frame_len = self.frame_num // num_part
        for n in range(num_part):
            if n == num_part - 1:
                sl = slice(n * frame_len, self.frame_num)  # last part
            else:
                sl = slice(n * frame_len, (n + 1) * frame_len)
            part_sum = torch.sum(self.cache[sl], axis=0)
            self.pixel_sum[self.mask_crop] += part_sum
            pixel_sum_par[n, self.mask_crop] = part_sum / (sl.stop - sl.start)

        self.pixel_sum /= self.frame_num
        self.pixel_sum = self.pixel_sum.reshape(*self.det_size)
        self.pixel_sum_par = pixel_sum_par.reshape(num_part, *self.det_size)

        frame_sum_val = torch.mean(self.cache, axis=1)
        # self.cache /= frame_sum_val.reshape(-1, 1)

        frame_sum_xax = torch.arange(self.frame_num, device=self.device)
        self.frame_sum = torch.stack([frame_sum_xax, frame_sum_val])

    def compute_smooth_data(self, mode='sqmap'):
        if mode == 'sqmap':
            slc = self.sq_slc
        elif mode == 'dqmap':
            slc = self.dq_slc
        else:
            raise ValueError('smooth method not supported')

        cache = self.cache.T
        for n in range(len(slc)):
            avg = cache[slc[n]].mean(dim=0)
            # get rid of the zeros
            avg[avg <= 0] = 1
            cache[slc[n]] /= avg

        self.cache = cache.T
        return

    def process_window(self, x):
        if self.cache_ptr >= 2 * self.window:
            self.cache_ptr = self.window
            self.cache = torch.roll(self.cache, shifts=-self.window, dims=0)

        x0 = x[0]
        self.cache[self.cache_ptr] = x0
        self.cache_ptr += 1

        if self.cache_ptr >= self.window:
            sl = slice(self.cache_ptr - self.window, self.cache_ptr)
            wf = self.cache[sl]
            for idx, roi in enumerate(self.dq_slc):
                a = wf[:, roi]
                b = x0[roi]
                a_sum = torch.sum(a, dim=1)
                b_sum = torch.sum(b)
                corr = torch.matmul(a, b.T) * len(b) / b_sum / a_sum
                self.c2[idx, self.c2_ptr] = corr
            self.c2_ptr += 1

        if x.shape[0] <= 1:
            return
        else:
            self.process_window(x[1:])
            return

    def post_processing(self, smooth_method=None, **kwargs):
        self.cache = self.cache.float()
        # saxs2d
        self.compute_average()
        # smooth data
        self.compute_smooth_data(smooth_method)

    def get_twotime_generator(self, **kwargs):
        for c2 in self.calc_normal_twotime(**kwargs):
            yield {f'correlation_map/c2_{self.c2_idx:05d}': c2}
            self.c2_idx += 1

        self.g2full = torch.stack(self.g2full).swapaxes(0, 1)
        self.g2partial = torch.stack(self.g2partial).permute(2, 1, 0)
        results = {
            'c2_g2': self.g2full,
            'c2_g2_segments': self.g2partial,
            'processed_bins': self.dq_idx
        }

        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                results[k] = v.cpu().numpy()
        yield results

    def get_scattering(self):
        num_par = self.pixel_sum_par.shape[0]
        saxs = {
            'saxs_2d': self.pixel_sum.reshape(-1)[self.mask_crop],
            'saxs_2d_par': self.pixel_sum_par.reshape(num_par, -1)[:, self.mask_crop],
            'mask_crop': self.mask_crop,
            'intensity_vs_time': self.frame_sum,
        }
        return saxs

    def calc_normal_twotime(self, num_partials=5):
        diag_mat = create_diagonal_index(self.frame_num, device=self.device)
        diag_mat_1d = diag_mat.reshape(-1)
        partial_len = self.frame_num // num_partials
        diag_mat_1d_par = diag_mat[0:partial_len, 0:partial_len].reshape(-1)

        # triu_idx = torch.tril_indices(self.num_frames, self.num_frames)
        for n, _ in enumerate(self.dq_idx):
            time_series = self.cache[:, self.dq_slc[n]]
            if self.sdata is not None:
                time_series = time_series / self.sdata[self.dq_slc[n]]

            norm_factor = torch.sum(time_series, dim=1)
            norm_factor[norm_factor <= 0] = 1
            norm_factor = 1.0 / norm_factor

            matmul_prod = torch.matmul(time_series, time_series.T)
            c2 = matmul_prod * norm_factor * norm_factor.reshape(-1, 1) * time_series.shape[1]

            g2full = compute_diagonal_average(diag_mat_1d, c2.reshape(-1))
            self.g2full.append(g2full)

            g2partial = []
            for idx in range(num_partials):
                sl = slice(idx * partial_len, (idx + 1) * partial_len)
                temp = compute_diagonal_average(diag_mat_1d_par,
                                                c2[sl, sl].reshape(-1))
                g2partial.append(temp)
            g2partial = torch.stack(g2partial)
            self.g2partial.append(g2partial)

            yield (torch.triu(c2)).cpu().numpy()


if __name__ == '__main__':
    pass
