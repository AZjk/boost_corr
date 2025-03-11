"""Module for the command line interface for boost_corr.
This module provides command line parsing and execution functionality.
TODO: Add detailed documentation.
"""

import argparse
import json
import logging
import sys
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s T+%(relativeCreated)05dms [%(filename)s]: %(message)s",
    datefmt="%m-%d %H:%M:%S",
)

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
    "dq_selection": "all",
    "verbose": False,
    "dry_run": False,  # Changed from "dryrun"
    "overwrite": False,
    "save_G2": False,  # Changed from "save_G2"
}


description = (
    "Compute Multi-tau/Twotime correlation for APS-8IDI XPCS " "datasets on GPU/CPU"
)
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
    "-r",
    "--raw",
    metavar="RAW_FILENAME",
    type=str,
    required=True,
    help="Filename of the raw data file (imm/rigaku/hdf)",
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
    "GPU. [default: %(default)s]",
)

parser.add_argument(
    "-b",
    "--begin-frame",
    type=int,
    default=default_config["begin_frame"],
    help="Starting frame index (0-based) for correlation. Used to skip bad "
    "initial frames. [default: %(default)s]",
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
    help="Number of frames to average before correlation. " "[default: %(default)s]",
)

parser.add_argument(
    "-t",
    "--type",
    metavar="TYPE",
    type=str,
    required=False,
    default=default_config["type"],
    help='Analysis type: "Multitau", "Twotime", or "Both". ' "[default: %(default)s]",
)

parser.add_argument(
    "-d",
    "--dq-selection",
    metavar="DQ_SELECTION",
    type=str,
    required=False,
    default=default_config["dq_selection"],
    help='DQ list selection (e.g., "1,2,5-7" selects [1,2,5,6,7]). "all" uses '
    "all dynamic qindex. [default: %(default)s]",
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
    "--save-g2",
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
    help="Configuration file path. Command line arguments override config file "
    "values",
)

args = parser.parse_args()
kwargs = vars(args)

if args.config is not None:
    config_fname = kwargs.pop("config")
    with open(config_fname) as f:
        config = json.load(f)

    common_keys = set(kwargs.keys()) & set(config.keys())
    for key in common_keys:
        if kwargs[key] != default_config[key]:
            # only update the args that are different from the default ones
            del config[key]
    kwargs.update(config)

kwargs["dq_selection"] = convert_to_list(kwargs["dq_selection"])


def main() -> None:
    """Main function to execute the command line interface.

    Returns:
        None
    """
    flag = 0
    if kwargs["dry_run"]:
        ans = "dry_run_only"
        print(json.dumps(kwargs, indent=4))
    else:
        kwargs.pop("dry_run")
        atype = kwargs.pop("type")
        if atype == "Multitau":
            from boost_corr.xpcs_aps_8idi.gpu_corr_multitau import solve_multitau

            method = solve_multitau
        elif atype == "Twotime":
            from boost_corr.xpcs_aps_8idi.gpu_corr_twotime import solve_twotime

            method = solve_twotime
        elif atype == "Both":
            from boost_corr.xpcs_aps_8idi.gpu_corr import solve_corr

            method = solve_corr
        else:
            flag = 1
            raise ValueError(f"Analysis type [{atype}] not supported.")

        ans = None
        try:
            if kwargs["gpu_id"] == -2:
                from boost_corr.gpu_scheduler import GPUScheduler

                with GPUScheduler(max_try=7200, sleep_duration=1) as scheduler:
                    kwargs["gpu_id"] = scheduler.gpu_id
                    ans = method(**kwargs)
            else:
                ans = method(**kwargs)
        except Exception:
            flag = 1
            traceback.print_exc()
            raise

    # send the result's fname to std-out
    print(ans)
    sys.exit(flag)


if __name__ == "__main__":
    sys.exit(main())
