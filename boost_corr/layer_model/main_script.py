import numpy as np
from layer_model import CorrModel
import logging
import torch
import time
from tqdm import trange
import tqdm
# torch.set_float32_matmul_precision('medium')
# torch.backends.cuda.matmul.allow_tf32 = True


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(filename)s]: %(message)s',
    datefmt='%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)


def main_with_simulated_data(
        device='cuda:0',
        num_frames=100000,
        batch_size=4096,
        num_channels=1024 * 512,
        queue_size=4096):

    model = CorrModel(num_frames, queue_size=queue_size,
                        num_channels=num_channels)

    # get_corr_result(model)
    # return
    # print(model)
    # print(model[0].get_result())
    # return

    model = model.to(device)
    logger.info(f'model sent to {device=}')
    num_batch = num_frames // batch_size

    with torch.no_grad():
        t0 = time.perf_counter()
        x = torch.rand(batch_size, num_channels, device=device)
        for n in trange(num_batch):
            x = torch.rand(batch_size, num_channels, device=device)
            model(x)
        t1 = time.perf_counter()

    freq = num_batch * batch_size / (t1 - t0)
    print(f'frequency is {freq:.2f} Hz, t_diff: {t1 - t0: .2f}s')
    return model


def main_with_real_data(
        device='cuda:0',
        batch_size=2048):

    from rigaku_handler import RigakuDataset
    dset = RigakuDataset('../../tests/raw_data/O018_Silica_D100_att0_Rq0_00005_nexus/O018_Silica_D100_att0_Rq0_00001.bin',
                         batch_size=batch_size, device=device)
    from torch.utils.data import DataLoader

    queue_size = batch_size
    num_channels = dset.det_size[0] * dset.det_size[1]
    logger.info(f'{num_channels=}: {dset.det_size=}')
    logger.info(f'{dset.frame_num=}')
    logger.info(f'{dset.batch_size=}')

    model = CorrModel(dset.frame_num, queue_size=queue_size,
                      num_channels=num_channels)
    # model = torch.compile(model)
    model.describe()
    model.to(device)
    model.to(torch.bfloat16)
    model.describe()
    
    with torch.no_grad():
        t0 = time.perf_counter()
        for x in tqdm.tqdm(DataLoader(dset)):
            model.forward(x[0])

    # with torch.no_grad():
    #     t0 = time.perf_counter()
    #     for x in tqdm.tqdm(DataLoader(dset)):
    #         model.forward(x[0])

    model.hard_flush()

    t1 = time.perf_counter()
    freq = dset.frame_num / (t1 - t0)
    logger.info(f'frequency is {freq:.2f} Hz, t_diff: {t1 - t0: .2f}s')

    # result = model.get_corr_result()
    # np.savez('corr_result_for_O18.npz', **result)


if __name__ == '__main__':
    # model = main_with_simulated_data()
    main_with_real_data()
