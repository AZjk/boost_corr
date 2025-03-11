"""Module for two-time correlation analysis.

This module provides functions and a TwotimeCorrelator class for computing two-time
correlation analysis, including functionalities like average computation, smoothing,
and correlation matrix calculation.
"""

import torch
from typing import Optional, Iterator, Any


def create_diagonal_index(N: int = 512) -> torch.Tensor:
    """Create a diagonal index matrix.

    Parameters:
        N (int): Size of the square matrix. Defaults to 512.

    Returns:
        torch.Tensor: A tensor with diagonal indexing.
    """
    m = torch.zeros((N, N), dtype=torch.int64)
    x = torch.hstack([torch.arange(N - 1, 0, -1), torch.arange(N)])

    for n in range(N):
        sl = slice(n, n + N)
        m[n] = x[sl]
    m = torch.flipud(m)
    return m


def compute_diagonal_average(index: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute the diagonal average of the given weights using the provided index.
    
    Parameters:
        index (torch.Tensor): A 1D tensor of indices.
        weights (torch.Tensor): A 1D tensor of weights corresponding to the indices.
    
    Returns:
        torch.Tensor: The computed diagonal average as a tensor.
    """
    tot = torch.bincount(index, weights=weights)
    pts = torch.bincount(index)
    return tot / pts


# def convert_to():
#     if mode != 'diag':
#     c2v = np.zeros((c2.shape[0], c2.shape[1] * 2), np.float32)
#     for n in range(c2.shape[0]):
#         sl = slice(c2.shape[1] - n, c2.shape[1] * 2 - n)
#         c2v[n, sl] = c2[n]
#     c2v[c2v == 0] = np.min(c2)
#     c2 = c2v


class TwotimeCorrelator:
    def __init__(self,
                 qinfo: dict,
                 frame_num: int,
                 det_size: tuple = (1024, 512),
                 window: int = 1024,
                 mask_crop: 'Optional[torch.Tensor]' = None,
                 device: str = 'cpu',
                 method: str = 'normal',
                 dtype: torch.dtype = torch.float32) -> None:
        """Initialize a TwotimeCorrelator for two-time correlation analysis.
        
        Parameters:
            qinfo (dict): Dictionary with keys 'dq_idx', 'dq_slc', 'sq_idx', 'sq_slc'.
            frame_num (int): Number of frames to process.
            det_size (tuple, optional): Detector dimensions. Default is (1024, 512).
            window (int, optional): Window size for processing. Default is 1024.
            mask_crop (Optional[torch.Tensor], optional): Crop mask; if None, a default mask is created.
            device (str, optional): 'cpu' or 'cuda:0'. Default is 'cpu'.
            method (str, optional): Processing method, e.g. 'normal' or 'window'.
            dtype (torch.dtype, optional): Data type for tensor computations. Default is torch.float32.
        """
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

    def process(self, x: torch.Tensor) -> None:
        """Process a new frame or batch of frames for correlation analysis.
        
        Parameters:
            x (torch.Tensor): Input tensor representing one or more frames.
        """
        if self.method == 'normal':
            sz = x.shape[0]
            self.cache[self.cache_ptr:self.cache_ptr + sz] = x
            self.cache_ptr += sz
            # if self.cache_ptr == self.num_frames:
            #     self.calc_normal_twotime()
        elif self.method == 'window':
            self.process_window(x)

    def compute_average(self, num_part: int = 10) -> None:
        """Compute average values for SAXS data over partitions.
        
        Parameters:
            num_part (int): Number of partitions to divide the frames for averaging.
        """
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

    def compute_smooth_data(self, mode: str = 'sqmap') -> None:
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

    def process_window(self, x: torch.Tensor) -> None:
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

    def post_processing(self, smooth_method: str = None, **kwargs) -> None:
        """Perform post-processing on cached data by converting to float,
        computing SAXS averages, and applying smoothing.

        Parameters:
            smooth_method (str, optional): Smoothing method to apply (e.g., 'sqmap', 'dqmap').
            **kwargs: Additional keyword arguments.
        """
        self.cache = self.cache.float()
        # saxs2d
        self.compute_average()
        # smooth data
        self.compute_smooth_data(smooth_method)

    def get_twotime_result(self, **kwargs) -> Iterator[dict]:
        """Retrieve two-time correlation results.
 
        This generator yields dictionaries containing two-time correlation data.
        First, it yields individual correlation arrays for each q index, then yields a final
        dictionary with aggregated statistics.
 
        Yields:
            dict: A dictionary mapping string keys to numpy arrays representing correlation data.
        """
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

    def get_saxs(self) -> dict:
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

    def calc_normal_twotime(self, num_partials: int = 5) -> Iterator[Any]:
        """Calculate the normal two-time correlation.
 
        This generator computes the two-time correlation for each index in dq_idx and yields
        an upper-triangular correlation matrix in numpy array format. It also accumulates global
        and partial g2 statistics.
 
        Parameters:
            num_partials (int, optional): Number of partitions to compute partial correlations. Default is 5.
 
        Yields:
            Any: Upper-triangular correlation matrix as a numpy array.
        """
        diag_mat = create_diagonal_index(self.frame_num).to(self.device)
        diag_mat_1d = diag_mat.reshape(-1)
        partial_len = self.frame_num // num_partials
        diag_mat_1d_p = diag_mat[0:partial_len, 0:partial_len].reshape(-1)

        # triu_idx = torch.tril_indices(self.frame_num, self.frame_num)

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
                temp = compute_diagonal_average(diag_mat_1d_p, c2[sl, sl].reshape(-1))
                g2partial.append(temp)
            g2partial = torch.stack(g2partial)
            self.g2partial.append(g2partial)

            yield (torch.triu(c2)).cpu().numpy()
            # yield c2[triu_idx[0], triu_idx[1]].cpu().numpy()


if __name__ == '__main__':
    pass
