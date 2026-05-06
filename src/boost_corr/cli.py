import sys
import json
import traceback
import argparse
import logging
import boost_corr.xpcs_aps_8idi.exceptions as exc
from boost_corr import __version__
from boost_corr import log_timer


_LOG_FORMAT = "%(asctime)s T+%(job_elapsed)s [%(filename)s]: %(message)s"
_LOG_DATEFMT = "%m-%d %H:%M:%S"


class _JobTimingFormatter(logging.Formatter):
    def format(self, record):
        record.job_elapsed = f"{log_timer.elapsed_s():.3f}s"
        return super().format(record)


class _MaxLevelFilter(logging.Filter):
    """Pass only records strictly below max_level (keeps stdout free of warnings)."""

    def __init__(self, max_level):
        self.max_level = max_level

    def filter(self, record):
        return record.levelno < self.max_level


def _setup_logging():
    formatter = _JobTimingFormatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)

    # INFO and below → stdout (visible in PBS .o file)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(_MaxLevelFilter(logging.WARNING))
    stdout_handler.setFormatter(formatter)

    # WARNING and above → stderr (captured in PBS .e file)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.WARNING)  # raised to INFO when -v is passed
    root.addHandler(stdout_handler)
    root.addHandler(stderr_handler)


_setup_logging()

# disable hdf5plugin info logging
logging.getLogger("hdf5plugin").setLevel(logging.WARNING)


def convert_to_list(input_str: str):
    """
    convert a string to a list of ints
    """
    if input_str == "all":
        return None

    result = []
    for part in input_str.split(","):
        if part == "":
            continue
        elif "-" in part:
            a, b = part.split("-")
            result.extend(range(int(a), int(b) + 1))
        else:
            result.append(int(part))
    result = sorted(list(set(result)))
    return result


default_config = {
    "qmap": None,  # Path to qmap file
    "output": "cluster_results",
    "smooth": "sqmap",
    "gpu_id": -1,  # -1 for CPU
    "begin_frame": 0,
    "end_frame": -1,  # -1 for all frames
    "stride_frame": 1,
    "avg_frame": 1,
    "type": "Multitau",  # "Multitau", "Twotime", or "Both"
    "normalize_frame": True,
    "dq_selection": "all",
    "verbose": False,
    "dry_run": False,  # Changed from "dryrun"
    "overwrite": False,
    "save_G2": False,  # Changed from "save_G2"
    "num_partial_g2": 0,  # Number of partial G2 to compute
    "crop_ratio_threshold": 0.5,  # Threshold for masking
    "max_memory": 36.0,  # Max memory usage in GB
    "num_segments": 1,
    "pbs_queue": "XpcsLowQ",
    "pbs_batch_size": 32,
    "pbs_gpus": 4,
}


description = (
    "Compute Multi-tau/Twotime correlation for APS-8IDI XPCS datasets on GPU/CPU"
)
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
    "-r",
    "--raw",
    metavar="RAW_FILENAMES",
    nargs="+",  # This allows one or more arguments
    type=str,
    required=True,
    help="One or more filenames of the raw data files (imm/rigaku/hdf), or a single .txt file containing one raw filename per line",
)

parser.add_argument(
    "-q",
    "--qmap",
    metavar="QMAP_FILENAME",
    type=str,
    required=False,
    default=default_config["qmap"],
    help="Filename of the qmap file (h5/hdf)",
)

parser.add_argument(
    "-o",
    "--output",
    metavar="OUTPUT_DIR",
    type=str,
    required=False,
    default=default_config["output"],
    help="Output directory for result files. Directory will be created if it "
    "doesn't exist. [default: %(default)s]",
)

parser.add_argument(
    "-s",
    "--smooth",
    metavar="SMOOTH",
    type=str,
    required=False,
    default=default_config["smooth"],
    help="Smooth method for Twotime correlation. [default: %(default)s]",
)

parser.add_argument(
    "-i",
    "--gpu-id",
    metavar="GPU_ID",
    type=int,
    default=default_config["gpu_id"],
    help="GPU selection: -1 for CPU, -2 for auto-scheduling, >=0 for specific "
    "GPU. -3 for APS PBS scheduler [default: %(default)s]",
)

parser.add_argument(
    "-nf",
    "--normalize-frame",
    type=int,
    choices=[0, 1],
    default=default_config["normalize_frame"],
    help="1 to enable, 0 to disable frame-based normalization. [default: %(default)s]",
)

parser.add_argument(
    "-b",
    "--begin-frame",
    type=int,
    default=default_config["begin_frame"],
    help="Starting frame index (0-based) for correlation. Used to skip bad "
    "initial frames. If negative, it will use python slice stype to resolve the "
    "start frames. [default: %(default)s]",
)

parser.add_argument(
    "-e",
    "--end-frame",
    type=int,
    default=default_config["end_frame"],
    help="Ending frame index (0-based, exclusive) for correlation. -1 uses all "
    "frames after begin_frame. [default: %(default)s]",
)

parser.add_argument(
    "-f",
    "--stride-frame",
    type=int,
    default=default_config["stride_frame"],
    help="Frame stride for processing. [default: %(default)s]",
)

parser.add_argument(
    "-a",
    "--avg-frame",
    type=int,
    default=default_config["avg_frame"],
    help="Number of frames to average before correlation. [default: %(default)s]",
)

parser.add_argument(
    "-t",
    "--type",
    metavar="TYPE",
    type=str,
    required=False,
    default=default_config["type"],
    help='Analysis type: "Multitau", "Twotime", or "Both". [default: %(default)s]',
)

parser.add_argument(
    "-d",
    "--dq-selection",
    metavar="DQ_SELECTION",
    type=str,
    required=False,
    default=default_config["dq_selection"],
    help='DQ list selection (e.g., "1,2,5-7" selects [1,2,5,6,7]). "all" uses "all dynamic qindex. [default: %(default)s]',
)

parser.add_argument(
    "--version",
    action="version",
    version=f"boost_corr {__version__}",
)

parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    default=default_config["verbose"],
    help="Enable verbose output",
)

parser.add_argument(
    "-G",
    "--save-G2",
    action="store_true",
    default=default_config["save_G2"],
    help="Save G2, IP, and IF to file",
)

parser.add_argument(
    "-n",
    "--dry-run",
    action="store_true",
    default=default_config["dry_run"],
    help="Show arguments without executing",
)

parser.add_argument(
    "-np",
    "--num-partial-g2",
    type=int,
    default=default_config["num_partial_g2"],
    help="number of partial g2 to compute. if 0, no partial g2 will be computed",
)

parser.add_argument(
    "--crop-ratio-threshold",
    type=float,
    default=default_config["crop_ratio_threshold"],
    help="Threshold for the ratio of valid (unmasked) pixels in the detector. "
    "If the ratio of valid pixels falls below this threshold, the raw data will "
    "be cropped to only include valid pixels, reducing memory usage. "
    "Range: 0.0-1.0. [default: %(default)s]",
)

parser.add_argument(
    "-p",
    "--prefix",
    type=str,
    default=None,
    help="prefix to add to the result filename",
)

parser.add_argument(
    "-u",
    "--suffix",
    type=str,
    default=None,
    help="suffix to add to the result filename",
)

parser.add_argument(
    "--bin-time-s",
    type=float,
    default=1e-6,
    help="time bin size in seconds for Timepix4 data. [default: %(default)s]",
)

parser.add_argument(
    "--run-config-path",
    type=str,
    default=None,
    help="Path to the run configuration file for Timepix4 data. [default: %(default)s]",
)

parser.add_argument(
    "--max-memory",
    type=float,
    default=default_config["max_memory"],
    help="Max memory to use in GB. [default: %(default)s]",
)

parser.add_argument(
    "--num-segments",
    type=int,
    default=default_config["num_segments"],
    help="Number of segments to divide the data into for independent processing. [default: %(default)s]",
)

parser.add_argument(
    "--meta-fname",
    type=str,
    default=None,
    help="Path to the metadata file; if not provided, the metadata file will be searched in the raw data directory",
)

parser.add_argument(
    "--pbs-queue",
    type=str,
    default=default_config["pbs_queue"],
    choices=["XpcsLowQ", "AiLowQ"],
    help="PBS queue for job submission (only used with --gpu-id -3). [default: %(default)s]",
)

parser.add_argument(
    "--pbs-batch-size",
    type=int,
    default=default_config["pbs_batch_size"],
    help="Number of raw files per PBS job (only used with --gpu-id -3). [default: %(default)s]",
)

parser.add_argument(
    "--pbs-gpus",
    type=int,
    default=default_config["pbs_gpus"],
    help="Number of GPUs available in the PBS queue; used to auto-scale batch size (only used with --gpu-id -3). [default: %(default)s]",
)

parser.add_argument(
    "-w",
    "--overwrite",
    action="store_true",
    default=default_config["overwrite"],
    help="Overwrite existing result files",
)

parser.add_argument(
    "-c",
    "--config",
    metavar="CONFIG_JSON",
    type=str,
    required=False,
    help="Configuration file path. Command line arguments override config file values",
)


def get_configurations():
    args = parser.parse_args()
    args.normalize_frame = bool(args.normalize_frame)
    kwargs = vars(args)

    if args.config is not None:
        config_fname = kwargs.pop("config")
        try:
            with open(config_fname) as f:
                config = json.load(f)
        except Exception as e:
            raise exc.ConfigurationError(f"Failed to load config file: {e}") from e

        common_keys = set(kwargs.keys()) & set(config.keys())
        for key in common_keys:
            if kwargs[key] != default_config[key]:
                # only update the args that are different from the default ones
                del config[key]
        kwargs.update(config)

    try:
        kwargs["dq_selection"] = convert_to_list(kwargs["dq_selection"])
    except Exception as e:
        raise exc.InputError(f"Invalid dq_selection: {e}") from e

    return kwargs


def main():
    exit_code = 0
    try:
        kwargs = get_configurations()
    except Exception as e:
        if hasattr(e, "exit_code"):
            logging.error(str(e))
            return e.exit_code
        traceback.print_exc()
        return 1

    if kwargs["verbose"]:
        logging.getLogger().setLevel(logging.INFO)

    if kwargs["dry_run"]:
        print(json.dumps(kwargs, indent=4))
    else:
        kwargs.pop("dry_run")
        from boost_corr.xpcs_aps_8idi.correlation import solve_correlation

        atype = kwargs.pop("type")
        method = lambda **kw: solve_correlation(analysis_type=atype, **kw)

        try:
            from boost_corr.devices import check_computing_device_exist

            if not check_computing_device_exist(kwargs["gpu_id"]):
                logging.error(f"GPU device [{kwargs['gpu_id']}] not found. Aborting.")
                traceback.print_exc()
                raise exc.ComputingDeviceError

            if kwargs["gpu_id"] == -2:
                from boost_corr.scheduler.simple_gpu_scheduler import GPUScheduler

                with GPUScheduler(max_try=7200, sleep_duration=1) as scheduler:
                    kwargs["gpu_id"] = scheduler.gpu_id
                    method(**kwargs)
            elif kwargs["gpu_id"] == -3:
                from boost_corr.scheduler.aps_pbs_scheduler import run_pbs_jobs

                run_pbs_jobs(**kwargs)
            else:
                method(**kwargs)
        except Exception as e:
            if hasattr(e, "exit_code"):
                exit_code = e.exit_code
            else:
                exit_code = 1
            traceback.print_exc()
            # disable raise e to pass the exit code to the main function;
            # raise e

    # print(f"Exit code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
