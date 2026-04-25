"""Helpers shared between solve_multitau and solve_twotime."""

import logging

import torch

import boost_corr.xpcs_aps_8idi.exceptions as exc

from ..xpcs_qpartitionmap import XpcsQPartitionMap

logger = logging.getLogger(__name__)


# Keys surfaced in exception notes when a single job fails inside a batch.
# Callers add analysis-specific keys via attach_debug_note(extra_keys=...).
_BASE_DEBUG_KEYS = frozenset(
    {
        "raw",
        "qmap",
        "output",
        "meta_fname",
        "gpu_id",
        "begin_frame",
        "end_frame",
        "avg_frame",
        "stride_frame",
    }
)


def create_qpm(qmap, device, crop_ratio_threshold, dq_selection=None, flag_sort=False):
    try:
        return XpcsQPartitionMap(
            qmap,
            device=device,
            flag_sort=flag_sort,
            crop_ratio_threshold=crop_ratio_threshold,
            dq_selection=dq_selection,
        )
    except Exception as e:
        raise exc.QMapError from e


def empty_gpu_cache(device: str):
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device.startswith("xpu"):
        torch.xpu.empty_cache()


def load_raw_list(raw):
    """Return raw as a flat list of file paths.

    If the input is a single .txt file, read paths from it (one per non-empty line).
    """
    if len(raw) == 1 and raw[0].endswith(".txt"):
        logger.info(f"raw input is a text file, loading raw file list from {raw[0]}")
        with open(raw[0], "r") as f:
            raw = [line.strip() for line in f if line.strip()]
    return raw


def attach_debug_note(e, analysis_kwargs, raw_fname, extra_keys=()):
    """Attach an analysis_kwargs note to e for batch-failure diagnostics."""
    keys = _BASE_DEBUG_KEYS | set(extra_keys)
    debug_info = {k: analysis_kwargs[k] for k in keys if k in analysis_kwargs}
    debug_info["raw"] = raw_fname
    e.add_note(f"analysis_kwargs: {debug_info}")
