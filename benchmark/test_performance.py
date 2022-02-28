import logging
import time
import torch
from boost_corr import MultitauCorrelator 


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s T+%(relativeCreated)05dms [%(filename)s]: %(message)s",
    datefmt="%m-%d %H:%M:%S")

logger = logging.getLogger(__name__)


device = 'cuda:0'
    

def test(queue_size=128):
    batch_size = queue_size
    # det_size = (1024, 512)
    det_size = (516, 1556)
    frame_num = 100 * 1000

    logger.info('frame_num = %d', frame_num)
    logger.info('queue_size = %d', queue_size)
    logger.info('det_size = %s', det_size)

    xb = MultitauCorrelator(det_size=det_size,
                   frame_num=frame_num,
                   queue_size=512,
                   device='cuda:1')
    # xb.debug()
    stime = time.perf_counter()
    for n in range(frame_num // batch_size + 1):
        sz = min(frame_num, (n + 1) * batch_size) - batch_size * n
        x = torch.ones((sz, det_size[0] * det_size[1]),
                       device=xb.device,
                       dtype=torch.bfloat16)
        xb.process(x)

    etime = time.perf_counter()
    logger.info('processing frequency is %.4f' % (frame_num / (etime - stime)))

    # xb.post_process()
    return


def test_host_gpu(queue_size=1):
    batch_size = queue_size
    # det_size = (1024, 512)
    # det_size = (516, 1556)
    det_size = (2048, 2048)
    frame_num = 100 * 1000

    logger.info('frame_num = %d', frame_num)
    logger.info('queue_size = %d', queue_size)
    logger.info('det_size = %s', det_size)


    stime = time.perf_counter()
    for n in range(frame_num // batch_size + 1):
        sz = min(frame_num, (n + 1) * batch_size) - batch_size * n
        # x = torch.ones((sz, det_size[0] * det_size[1]), device='cpu',
        #         dtype=torch.uint8, pin_memory=True)
        x = torch.ones((sz, det_size[0] * det_size[1]), device='cpu',
                dtype=torch.uint8, pin_memory=True)
        x = x.to(device)

    etime = time.perf_counter()
    logger.info('processing frequency is %.4f' % (frame_num / (etime - stime)))


if __name__ == '__main__':
    test_host_gpu()

