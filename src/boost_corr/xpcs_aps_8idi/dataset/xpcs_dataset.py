import numpy as np
import torch
import logging
import os
import math


np_torch_map = {
    np.uint8: torch.uint8,
    np.uint16: torch.int16,
    np.int16: torch.int16,
    np.uint32: torch.int32,
}

logger = logging.getLogger(__name__)


class XpcsDataset(object):
    """ """

    def __init__(
        self,
        fname,
        begin_frame=0,
        end_frame=-1,
        stride_frame=1,
        avg_frame=1,
        batch_size=128,
        det_size=(512, 1024),
        device="cuda:0",
        use_loader=False,
        dtype=np.uint8,
        mask_crop=None,
        bin_time_s=1,  # bin time in seconds, placeholder for non-timepix detectors 
        run_config_path=None,      # run config path, placeholder for non-timepix detectors
    ):
        self.fname = fname
        self.raw_size = os.path.getsize(fname) / (1024**2)
        self.det_size = det_size
        self.pixel_num = self.det_size[0] * self.det_size[1]
        self.is_sparse = True

        self.mask_crop = mask_crop

        self.device = device

        self.batch_size = batch_size
        self.batch_num = 0
        self.frame_num = 0
        self.frame_num_raw = 0
        self.begin_frame = begin_frame
        self.end_frame = end_frame
        self.avg_frame = avg_frame
        self.stride = stride_frame

        self.use_loader = use_loader
        self.dtype = dtype
        self.dataset_type = None
        self.dtype_raw = None
        self.bin_time_s = bin_time_s

    def update_batch_info(self, frame_num: int) -> None:
        """
        Update batching settings.
        """

        def _resolve_index(idx: int, n: int) -> int:
            """Python-slice-style resolution with clamping to [0, n]."""
            if idx < 0:
                idx = n + idx
            return 0 if idx < 0 else (n if idx > n else idx)

        # --- inputs & basic checks ---
        self.frame_num_raw = frame_num
        if frame_num < 0:
            raise ValueError("frame_num must be non-negative")

        if getattr(self, "avg_frame", 1) <= 0 or getattr(self, "stride", 1) <= 0:
            raise ValueError("avg_frame and stride must be positive")

        if getattr(self, "batch_size", 1) <= 0:
            raise ValueError("batch_size must be positive")

        # note: if end_frame is <= 0, it will use all frames after begin_frame
        end = self.end_frame if self.end_frame > 0 else frame_num
        end = min(frame_num, end)

        # --- resolve indices like Python slices ---
        begin = _resolve_index(self.begin_frame, frame_num)

        if end < begin:
            raise ValueError(f"end_frame [{end}] must be >= begin_frame [{begin}]")

        # --- effective rounding to correlation block size ---
        eff_len = int(self.avg_frame) * int(self.stride)
        tot = end - begin
        assert tot > 0, "No frames to process after applying begin/end_frame."
        tot_rounded = (tot // eff_len) * eff_len
        rounded_end = begin + tot_rounded

        if rounded_end != end:
            logger.info(
                "end_frame rounded down from %d to %d to align with eff_len=%d.",
                end,
                rounded_end,
                eff_len,
            )
            end = rounded_end

        # --- finalize & batch math ---
        self.begin_frame = begin
        self.end_frame = end

        self.frame_num = (end - begin) // eff_len
        self.batch_num = (self.frame_num + self.batch_size - 1) // self.batch_size

    def update_mask_crop(self, new_mask):
        # some times the qmap's orientation is different from the dataset;
        # after rorating qmap, explicitly apply the new mask; if the original
        # mask is None (ie. not croping), then abort
        if self.mask_crop is None:
            return
        else:
            self.mask_crop = new_mask

    def update_det_size(self, det_size):
        self.det_size = tuple(det_size)
        self.pixel_num = self.det_size[0] * self.det_size[1]

    def get_raw_index(self, idx):
        # get the raw index list
        eff_len = self.stride * self.avg_frame * self.batch_size
        beg = self.begin_frame + eff_len * idx
        end = min(self.end_frame, beg + eff_len)
        size = (end - beg) // self.stride
        return beg, end, size

    def get_sparsity(self):
        x = self.__getitem__(0)[0]
        y = (x > 0).sum() / self.pixel_num
        self.dtype_raw = x.dtype
        self.__reset__()
        return y

    def get_description(self):
        result = {}
        for key in [
            "fname",
            "frame_num_raw",
            "is_sparse",
            "frame_num",
            "det_size",
            "device",
            "batch_size",
            "batch_num",
            "dataset_type",
        ]:
            result[key] = self.__dict__[key]
        result["frame_info"] = (
            "(begin, end, stride, avg) = ("
            f"{self.begin_frame}, {self.end_frame}, "
            f"{self.stride}, {self.avg_frame})"
        )
        return result

    def describe(self):
        for key, val in self.get_description().items():
            logger.info(f"{key}: {val}")

        if self.mask_crop is not None:
            valid_size = self.mask_crop.shape[0]
        else:
            valid_size = self.pixel_num
        logger.info(f"dtype: {self.dtype}")
        logger.info(f"valid_size: {valid_size}")
        logger.info(f"sparsity: {self.get_sparsity():.4f}")
        logger.info(f"raw dataset file size: {self.raw_size:.2f} MB")

    def __len__(self):
        return self.batch_num

    def __reset__(self):
        return

    def __getbatch__(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        x = self.__getbatch__(idx)
        if self.avg_frame > 1:
            x = x.reshape(-1, self.avg_frame, x.shape[-1])
            x = torch.mean(x.float(), axis=1)
        return x

    def to_rigaku_bin(self, fname):
        """
        convert to rigaku binary format
        """
        offset = 0
        fid = open(fname, "a")
        for n in range(self.batch_num):
            x = self.__getitem__(n)
            idx = np.nonzero(x)
            count = x[idx]
            frame = idx[0] + offset
            index = idx[1].astype(np.int64)
            offset += x.shape[0]

            y = np.zeros_like(count, dtype=np.int64)
            y += frame.astype(np.int64) << 40
            y += index << 16
            y = y + count
            y.tofile(fid, sep="")

        fid.close()

    def sparse_to_dense(self, index, frame, count, size):
        if isinstance(index, np.ndarray):
            # using numpy array; will be sent to device by dataloader
            x = np.zeros((size, self.pixel_num), dtype=self.dtype)
            x[frame, index] = count
        else:
            x = torch.zeros(
                (size, self.pixel_num),
                dtype=np_torch_map[self.dtype],
                device=index.device,
            )
            x[frame.long(), index.long()] = count

        return x
