"""Module for multitau correlation analysis.
This module provides functions and classes for performing multitau correlation analysis.
TODO: Add detailed documentation.
"""

import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from boost_corr.help_functions import gen_tau_bin, is_power_two, sort_tau_bin

# import matplotlib.pyplot as plt
# import skimage.io as skio

logger = logging.getLogger(__name__)


def compute_dtype(max_count, levels):
    """Compute the appropriate data type for accumulation based on max_count and levels.

    Parameters:
        max_count (int): Maximum photon count.
        levels: Levels configuration.

    Returns:
        Appropriate data type.
    """
    # get the dtype that can hold the accumulation of photons without losing
    # precision
    dtype_list = [
        [1 / (2**7 - 1), torch.bfloat16, 2],  # bfloat16 has  bits
        [1 / (2**23 - 1), torch.float32, 4],  # float32 has 23 bits
        [1 / (2**52 - 1), torch.float64, 8],  # float64
    ]

    dtype_ret = []
    for level in levels:
        accu_count = max_count * (2**level)
        min_percent = 1.0 / accu_count
        for n in range(len(dtype_list)):
            if min_percent > dtype_list[n][0]:
                dtype_ret.append(dtype_list[n][1:])
                break

    return dtype_ret


def compute_queue(levels_num, queue_size, queue_level):
    """Compute the queue parameters based on levels, queue size, and queue level.

    Parameters:
        levels_num (int): Number of levels.
        queue_size (int): Size of the queue.
        queue_level (int): Queue level.

    Returns:
        Tuple: Computed queue size and level.
    """
    x = np.arange(levels_num, dtype=np.int64)
    y = queue_size // (2 ** (x // queue_level))
    y[y < 8] = 8
    return y


class MultitauCorrelator(object):
    """Class for performing multitau correlation analysis."""

    def __init__(
        self,
        det_size,
        frame_num,
        device="cpu",
        queue_size: int = 64,
        queue_level: int = 4,
        auto_queue=True,
        mask_crop=None,
        max_memory=36.0,
        max_count=7,
    ) -> None:
        """Initialize the MultitauCorrelator.

        Parameters:
            det_size: Detector size.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.det_size = det_size
        self.is_sparse = False
        self.max_memory = max_memory

        self.mask_crop = mask_crop
        if self.mask_crop is not None:
            self.pixel_num = mask_crop.shape[0]
        else:
            self.pixel_num = det_size[0] * det_size[1]

        self.frame_num = frame_num

        assert isinstance(queue_size, int)
        assert is_power_two(queue_size)
        self.queue_size = queue_size

        tau_bin = gen_tau_bin(self.frame_num)
        self.tau_bin = tau_bin

        tau_max, levels, tau_in_level = sort_tau_bin(tau_bin, frame_num)
        self.levels_num = len(levels)
        self.tau_num = tau_bin.shape[1]
        levels_num = len(levels)

        self.tau_rev = tau_max
        self.dtype_list = compute_dtype(max_count, levels)

        # self.device = torch.device(device_type)
        self.device = device

        self.ct = [None for n in range(levels_num)]
        self.tau_all = []
        self.tau_in_level = tau_in_level

        # over write queue_size and queue_level
        if auto_queue:
            queue_size, queue_level = self.auto_set_queue(queue_size)
        queue = compute_queue(self.levels_num, queue_size, queue_level)
        self.queue_info = (auto_queue, queue_size, queue_level)

        for n in range(levels_num):
            curr_tau_all = self.tau_rev + queue[n]
            self.ct[n] = torch.zeros(
                size=(curr_tau_all, self.pixel_num),
                dtype=self.dtype_list[n][0],
                device=self.device,
            )
            self.tau_all.append(curr_tau_all)
            # print(self.dtype_list[n])

        # account for the overflow photons
        self.ct_overflow = torch.zeros(size=(self.pixel_num,), device=self.device)

        # G2, IP and IF
        self.g2 = torch.zeros(
            size=(self.tau_num, 3, self.pixel_num),
            dtype=torch.float32,
            device=self.device,
        )
        # pointer to the data enty in each level
        self.pt = [self.tau_rev] * levels_num

        # advanced flag
        self.ad = [False] * levels_num

        self.par_level = 0
        while self.frame_num // (2**self.par_level) > 16:
            self.par_level += 1

        # self.intt = torch.zeros(size=(frame_num, 2), dtype=torch.float32)
        self.intt = []
        self.saxs_2d = torch.zeros_like(self.ct_overflow)
        self.saxs_2d_par = []

        self.levels = levels
        self.tau_in_level = tau_in_level
        self.current_frame = 0

    def describe(self):
        """Log or print the dataset description."""
        logger.info(f"queue information (auto, size, level): {self.queue_info}")
        dlist = [str(self.dtype_list[n][0]) for n in range(len(self.dtype_list))]
        dlist_uniq = list(set(dlist))
        dlist_uniq.sort()
        for x in dlist_uniq:
            logger.info(f"stage @ {dlist.count(x):02d}: {x}")
        logger.info(f"max_memory is {self.max_memory}GB")

    def reset(self):
        """Reset the internal state of the correlator."""
        for n in range(self.levels_num):
            self.ct[n].zero_()

        # account for the overflow photons
        self.ct_overflow = 0

        # G2, IP and IF
        self.g2.zero_()
        # pointers for each level
        self.pt = [self.tau_rev] * self.levels_num

        # advanced flag
        self.ad = [False] * self.levels_num
        self.intt = []
        self.saxs_2d = None
        self.saxs_2d_par = []
        self.current_frame = 0

    def auto_set_queue(self, min_queue_size, overhead=0.25):
        """Automatically set the queue parameters.

        Parameters:
            min_queue_size: Minimum queue size.
            overhead (float): Overhead factor.
        """
        # the size of each frame in all levels;
        sz = np.zeros(self.levels_num)
        for n in range(self.levels_num):
            sz[n] = self.dtype_list[n][1] * self.pixel_num / (1024**3)
        # RAM reservation for the G2/IP/IF array
        g2_sz = 4 * self.pixel_num / (1024**3) * self.tau_num * 3

        max_memory = self.max_memory / (1 + overhead)
        max_memory -= np.sum(sz) * self.tau_rev + g2_sz

        # choose a starting level not equal to 1 so the flush of different
        # levels don't happen at the same time.
        queue_level = 3
        queue_size = min_queue_size

        def get_size(*args):
            return np.sum(compute_queue(*args) * sz)

        while get_size(self.levels_num, queue_size * 2, queue_level) <= max_memory:
            queue_size *= 2
            if queue_size > self.frame_num:
                break

        while get_size(self.levels_num, queue_size, queue_level + 1) <= max_memory:
            queue_level += 1
            if queue_level >= self.levels_num:
                break

        return queue_size, queue_level

    def process_numpy(self, *args):
        """Process numpy input data."""
        if not self.is_sparse:
            y = torch.from_numpy(args[0]).to(self.device)
            self.__process_dense__(y)
        else:
            index, frame, count, size = args
            index = torch.from_numpy(index).to(self.device)
            frame = torch.from_numpy(frame).to(self.device)
            count = torch.from_numpy(count).to(self.device)
            self.__process_sparse__(index, frame, count, size)

    def process(self, *args):
        """Process input data."""
        if not self.is_sparse:
            self.__process_dense__(*args)
        else:
            self.__process_sparse__(*args)

    def __process_sparse__(self, index, frame, count, size):
        """Internal method to process sparse input data."""
        x = torch.zeros((size, self.pixel_num), count.dtype, device=self.device)
        x[frame.long(), index.long()] = count
        self.__process_dense__(x)

    def __process_dense__(self, x=None):
        """Internal method to process dense input data."""
        # convert raw input (likely uint16) to float
        # prefer to pass an input with size of 1/N queue size
        # x = x.float()

        self.current_frame += x.shape[0]
        last_flag = self.current_frame == self.frame_num

        for level in self.levels:
            if x is not None:
                pt0 = self.pt[level]
                self.ct[level][pt0 : pt0 + x.shape[0]] = x
                self.pt[level] += x.shape[0]

            if self.pt[level] != self.tau_all[level] and not last_flag:
                # most likely case
                return
            else:
                x = self.flush_level(level, last_flag)

                # accumulate the overflow photons; most likely to be zero due
                # to the tau/fold list setting; last level; must return
                if level == self.levels[-1] and x is not None:
                    self.ct_overflow += torch.sum(x, dim=0)
                    return

                # if there are no left events and it's not the final batch
                if x is None and not last_flag:
                    return

    def flush_level(self, level, last_flag):
        """Flush the data for a given level."""
        beg = self.tau_rev
        end = self.pt[level]
        # when the channel has nothing;
        if end - beg <= 0:
            return None

        if level == 0:
            # sl = slice(self.current_frame - end + beg, self.current_frame)
            intt = torch.mean(self.ct[level][beg:end].float(), dim=1)
            self.intt.append(intt)
            self.saxs_2d += torch.sum(self.ct[level][beg:end].float(), dim=0)
            self.ct[level][beg:end] /= intt.reshape(end - beg, 1)

        for tau, tid, _ in self.tau_in_level[level]:
            sl1 = slice(beg - tau, end - tau)
            sl2 = slice(beg, end)
            xy = torch.sum(self.ct[level][sl1] * self.ct[level][sl2], dim=0)
            self.g2[tid, 0] += xy

        # deal with IP; it may happen when the channel advances or it's the
        # last batch and needs flush;
        if not self.ad[level]:
            for tau, tid, _ in self.tau_in_level[level]:
                sl = slice(beg, beg + tau)
                self.g2[tid, 1] -= torch.sum(self.ct[level][sl], dim=0)
            self.ad[level] = True

        # deal with IF; it only happends for the final batch
        if last_flag:
            for tau, tid, _ in self.tau_in_level[level]:
                sl = slice(end - tau, end)
                self.g2[tid, 2] -= torch.sum(self.ct[level][sl], dim=0)

        if level == self.par_level:
            self.saxs_2d_par.append(self.ct[level][beg:end].clone().detach())

        # the number of elements should be an even number for all; but the
        # last call can be an odd number; advance the event; the left element
        # will be used to calcuate the sum
        avg_len = (end - beg) // 2 * 2
        self.pt[level] -= avg_len
        avg_sl = slice(beg, beg + avg_len)

        x = torch.sum(
            self.ct[level][avg_sl].view(avg_len // 2, 2, self.pixel_num), dim=1
        )

        # copy the ending to buffer
        tmp = self.ct[level][avg_len:end].clone().detach()
        self.ct[level][0 : end - avg_len] = tmp

        return x

    def post_process(self):
        """Perform post-processing on the computed data."""
        # rescale the average and add total flux to IP and IF
        tot = self.ct_overflow
        for level in np.flip(self.levels):
            sl = slice(self.tau_rev, self.pt[level])
            tot = tot + torch.sum(self.ct[level][sl], dim=0)
            # normalization
            # G2 = (x / scl) * (y / scl) / eff_len  -> eff_len * scl ^ 2
            # IP = (x / scl)             / eff_len  -> eff_len * scl
            # IF = (y / scl)             / eff_len  -> eff_len * scl
            for _, tid, eff_length in self.tau_in_level[level]:
                self.g2[tid, 0] /= eff_length * 2 ** (level * 2)
                self.g2[tid, 1:3] += tot
                self.g2[tid, 1:3] /= eff_length * 2**level

        # get saxs2d
        # self.saxs_2d = torch.unsqueeze(tot / self.frame_num, 0)
        self.saxs_2d /= self.frame_num
        if len(self.saxs_2d_par) > 0:
            self.saxs_2d_par = torch.vstack(self.saxs_2d_par) / self.frame_num
        else:
            self.saxs_2d_par = torch.unsqueeze(self.saxs_2d, 0)
        return

    def process_dataset(self, ds, verbose=True, use_loader=False, num_workers=16):
        """Process the dataset with optional verbosity and concurrency."""
        if not use_loader:
            xrange = trange if verbose else range
            for n in xrange(len(ds)):
                self.process(ds[n])
        else:
            # only pin memory if using GPU for computing
            pin_memory = self.device != "cpu"
            dl = DataLoader(
                ds,
                batch_size=None,
                pin_memory=pin_memory,
                num_workers=num_workers,
                prefetch_factor=4,
            )
            logger.info(f"using {num_workers} workers to load data")

            if verbose:
                container = tqdm(dl, total=len(ds), desc="progress")
            else:
                container = dl

            for x in container:
                x = x.to(self.device, non_blocking=True)
                self.process(x)

        self.post_process()

    def get_results(self):
        """Retrieve the results of the correlation computation."""
        intt = torch.hstack(self.intt).float()
        tline = torch.arange(intt.shape[0], device=intt.device)
        intt = torch.vstack([tline, intt])

        output_scattering = {
            "saxs_2d": self.saxs_2d.float(),
            "saxs_2d_par": self.saxs_2d_par.float(),
            "intensity_vs_time": intt.float(),
            "mask_crop": self.mask_crop,
        }

        output_multitau = {
            "G2": self.g2.float(),
            "mask_crop": self.mask_crop,
            "tau": self.tau_bin[0, :],
        }
        return output_scattering, output_multitau


def read_data(
    det_size, fname="../xpcs_data_simulation/simulation_0010k_sparse_0.005.bin"
):
    """Read data from a binary file.

    Parameters:
        det_size: Detector size.
        fname (str): File name.

    Returns:
        Parsed data.
    """
    det_size_1d = det_size[0] * det_size[1]
    data = np.fromfile(fname, dtype=np.uint16)
    data = data.astype(np.int16)
    data = data.reshape(-1, det_size_1d)
    return data, data.shape[0]


def example(
    queue_size=512,
    fname="simulation_0010k_sparse_0.005.hdf",
    data_dir="../xpcs_data_simulation",
):
    """Example function demonstrating multitau correlation process.

    Parameters:
        queue_size (int): Size of the queue.
        fname (str): File name.
        data_dir (str): Directory of the data.

    Returns:
        None
    """
    batch_size = queue_size
    # det_size = (1024, 512)
    det_size = (1813, 1558)
    frame_num = 100000

    logger.info("frame_num = %d", frame_num)
    logger.info("queue_size = %d", queue_size)
    logger.info("det_size = %s", det_size)
    xb = MultitauCorrelator(
        det_size=det_size, frame_num=frame_num, queue_size=queue_size, device="cuda:1"
    )
    xb.debug()
    stime = time.perf_counter()
    for n in tqdm(range(frame_num // batch_size + 1), colour="green"):
        sz = min(frame_num, (n + 1) * batch_size) - batch_size * n
        x = torch.ones(
            (sz, det_size[0] * det_size[1]), device=xb.device, dtype=torch.bfloat16
        )
        xb.process(x)

    etime = time.perf_counter()
    logger.info("processing frequency is %.4f" % (frame_num / (etime - stime)))

    xb.post_process()
    return


def test_dtype():
    """Test the data type computation functionality."""
    levels = list(np.arange(32))
    max_count = 1
    compute_dtype(max_count, levels)


if __name__ == "__main__":
    # for queue_size in 2 ** np.arange(3, 15):
    #     example(queue_size=queue_size)
    example(queue_size=512)
    # test_dtype()
