import numpy as np
import torch
from torch import nn
from tqdm import trange
import time
import logging
from torch.linalg import vecdot
from help_functions import gen_tau_bin


logger = logging.getLogger(__name__)


def build_delay_bin_list(num_frames):
    delay_absolute, levels = gen_tau_bin(num_frames)
    delay = delay_absolute // (2 ** levels)

    layer_args = []
    for level in np.sort(np.unique(levels)):
        delay_in_level = np.sort(delay[levels == level])
        layer_args.append([level, delay_in_level.copy()])
        # logger.info(f'{level=}: {delay_in_level=}')
    return layer_args


class CorrLayerNumpy():
    def __init__(self, queue_size=1024, num_channels=512, delay_list=(1, 2, 3),
                 avg_window=2, absolute_scale=1, use_ipf=True,
                 dtype=np.float32):
        assert queue_size % avg_window == 0

        self.num_channels = num_channels
        self.reserve_size = np.max(delay_list)
        self.total_size = queue_size + self.reserve_size
        self.queue = np.zeros((self.total_size, num_channels), dtype=dtype)
        self.bound = (self.reserve_size, self.total_size)
        self.corr_result = np.zeros((len(delay_list), num_channels),
                                    dtype=dtype)

        self.delay_list = np.array(delay_list)
        self.delay_list_absolute = self.delay_list * absolute_scale
        self.avg_window = avg_window
        self.ptr = self.reserve_size
        self.corr_length = 0
        self.signal_accumulated = np.zeros(num_channels)
        self.avg_length = 0
        self.use_ipf = use_ipf
    
    def process(self, x, **kwargs):
        return self.forward(x, **kwargs)
    
    def show_status(self):
        result = self.calc_g2()
        print(f'{self.avg_length=}, {self.corr_length=}')
        print(result)
    
    def get_signal_sum(self):
        signal = self.signal_accumulated
        return signal
    
    def calc_g2(self):
        eff_len = self.corr_length - self.delay_list
        eff_len = np.clip(eff_len, a_min=1, a_max=None)
        avg_length = max(1, self.avg_length)

        up = self.corr_result / eff_len.reshape(-1, 1)
        dn = self.signal_accumulated / avg_length 
        dn = np.clip(dn, a_min=1e-8, a_max=None)
        g2 = up / (dn ** 2) 
        return g2
    
    def get_result(self):
        eff_len = self.corr_length - self.delay_list
        eff_len = np.clip(eff_len, a_min=1, a_max=None)
        avg_length = max(1, self.avg_length)

        with torch.no_grad():
            G2 = self.corr_result / eff_len.reshape(-1, 1)
            IP_t = self.signal_accumulated / avg_length
            IP = np.ones_like(G2) * IP_t

        return G2, IP
    
    def forward(self, x, hard_flush=False):
        if x is not None:
            assert x.shape[-1] == self.queue.shape[-1]
            signal_size = x.shape[0]

            if self.ptr + signal_size <= self.bound[1]:
                sl = slice(self.ptr, self.ptr + signal_size)
                self.queue[sl] = x
                self.ptr += signal_size
            else:
                logger.warn('memory not aligned. This may affect performance.')
                target_size = self.bound[1] - self.ptr
                # x1 won't be None
                x1 = self.forward(x[0:target_size])
                x2 = self.forward(x[target_size:])
                if x2 is not None:
                    return np.cat([x1, x2])
                else:
                    return x1

        if self.ptr == self.bound[1] or hard_flush:
            return self.flush(hard_flush)
        else:
            return None

    def flush(self, hard_flush=False):
        # compute average value for the next correlation layer
        avg_length = (self.ptr - self.bound[0]) // self.avg_window * self.avg_window

        avg_shape = (avg_length // self.avg_window, self.avg_window, 
                     self.num_channels)

        if not hard_flush:
            corr_length = avg_length
        else:
            corr_length = self.ptr - self.bound[0] 

        end_ptr = self.bound[0] + corr_length

        avg_sl = slice(self.bound[0], self.bound[0] + avg_length)
        avg_value = self.queue[avg_sl].reshape(*avg_shape).sum(axis=1)

        self.signal_accumulated += avg_value.sum(axis=0)

        if corr_length != avg_length:
            left_sl = slice(self.bound[0] + avg_length, end_ptr)
            self.signal_accumulated += np.sum(self.queue[left_sl], 0)

        self.avg_length += corr_length
        self.corr_length += corr_length 

        # compute correlation
        for n, delay in enumerate(self.delay_list):
            sl_a = slice(self.bound[0] - delay, end_ptr - delay)
            sl_b = slice(self.bound[0], end_ptr)
            self.corr_result[n] += np.sum(self.queue[sl_a] * self.queue[sl_b], axis=0)

            # self.corr_result[n] += vecdot(self.queue[sl_a],
            #                               self.queue[sl_b],
            #                               dim=0)

        # copy data to reserved space
        # if self.bound[1] > self.bound[0] * 4:
        if True:
            source_sl = slice(end_ptr - self.reserve_size, self.ptr)
            target_sl = slice(0, self.ptr - corr_length)
            self.queue[target_sl] = self.queue[source_sl]
        # else:
        #     self.queue = torch.roll(self.queue, -corr_length, 0) 

        # reset pointer
        self.ptr = self.ptr - corr_length 

        return avg_value
    

class CorrModelNumpy():
    def __init__(self, num_frames, queue_size=1024, num_channels=512) -> None:
        super().__init__()
        layers = []
        layer_args = build_delay_bin_list(num_frames)
        for level, delay_in_level in layer_args:
            queue_size_t = max(16, queue_size // (2 ** level))
            logger.info(f'{level=:2d}: {queue_size_t=:4d}, {delay_in_level=}')
            layer = CorrLayerNumpy(queue_size=queue_size_t, num_channels=num_channels,
                                   delay_list=delay_in_level, avg_window=2,
                                   absolute_scale=2**level)
            layers.append(layer)
        self.model = layers
    
    def get_corr_result(self):
        delays = []
        g2 = []
        ip = []
        for layer in self.model:
            g2_t, ip_t = layer.get_result()
            g2.append(g2_t)
            ip.append(ip_t)
            delays.append(layer.delay_list_absolute)

        delays = np.hstack(delays)
        g2 = np.vstack(g2).astype(np.float32)
        ip = np.vstack(ip).astype(np.float32)
        print(g2.shape, ip.shape, delays.shape)
        result = {"t_el": delays, "G2": g2, "IP": ip}
        np.savetxt('G2.txt', g2)
        np.savetxt('IP.txt', ip)
        return
    
    def forward(self, x):
        for layer in self.model:
            x = layer.forward(x)
        return 
    
    def hard_flush(self):
        x = None
        for layer in self.model:
            x = layer.forward(x, hard_flush=True)
        return


def test_01():
    device = 'cpu'
    num_frames = 16 
    batch_size = num_frames
    num_channels = 2 
    queue_size = batch_size 

    # num_batch = num_frames // batch_size 
    num_batch = 2 
    # delay_list = (1, 2, 3, 4, 5, 6, 7, 8)
    delay_list = (1, 2)

    cl_1 = CorrLayerNumpy(queue_size, num_channels, delay_list)
    cl_2 = CorrLayerNumpy(queue_size // 2, num_channels, delay_list)

    t0 = time.perf_counter()
    # x = torch.rand(batch_size, num_channels, device=device)
    with torch.no_grad():
        for n in trange(num_batch):
            # x = torch.rand(batch_size, num_channels, device=device)
            x = np.ones((batch_size, num_channels))
            x = cl_1.forward(x)
            x = cl_2.forward(x)

    t1 = time.perf_counter()

    freq = num_batch * batch_size / (t1 - t0)
    print(f'frequency is {freq:.2f} Hz, t_diff: {t1 - t0: .2f}s')
    print('cl_1 corr result')
    cl_1.show_status()
    print('cl_2 corr result')
    cl_2.show_status()


def test_02_hard_flush():
    num_frames = 31
    num_channels = 2
    queue_size = 16

    # num_batch = num_frames // batch_size 
    num_batch = 2 
    batch_list = [16, 15]
    # delay_list = (1, 2, 3, 4, 5, 6, 7, 8)
    delay_list = (1, 2)

    cl_1 = CorrLayerNumpy(queue_size, num_channels, delay_list)
    cl_2 = CorrLayerNumpy(queue_size // 2, num_channels, delay_list)

    t0 = time.perf_counter()
    # x = torch.rand(batch_size, num_channels, device=device)
    for n in trange(num_batch):
        batch_size = batch_list[n]
        # x = torch.rand(batch_size, num_channels, device=device)
        x = torch.ones(batch_size, num_channels)
        hard_flush = (n == num_batch - 1)

        x = cl_1.forward(x, hard_flush)
        x = cl_2.forward(x, hard_flush)
    
    t1 = time.perf_counter()

    freq = num_batch * batch_size / (t1 - t0)
    print(f'frequency is {freq:.2f} Hz, t_diff: {t1 - t0: .2f}s')
    cl_1.show_status()
    cl_2.show_status()

    print(cl_1.get_signal_sum())
    print(cl_2.get_signal_sum())


def test_03():
    device = 'cuda:0'
    num_frames = 16384 * 16
    batch_size = 4096 
    num_channels = 1024 * 1024 
    queue_size = 4096 
    num_batch = num_frames // batch_size 
    delay_list = (1, 2, 3, 4, 5, 6, 7, 8)

    cl_1 = CorrLayer(queue_size, num_channels, delay_list)
    cl_1.to(device)

    t0 = time.perf_counter()
    x = torch.rand(batch_size, num_channels, device=device)
    for n in trange(num_batch):
        # x = torch.rand(batch_size, num_channels, device=device)
        cl_1.process(x)
    t1 = time.perf_counter()

    freq = num_batch * batch_size / (t1 - t0)
    print(f'frequency is {freq:.2f} Hz, t_diff: {t1 - t0: .2f}s')


if __name__ == '__main__':
    # test_01()
    test_02_hard_flush()
