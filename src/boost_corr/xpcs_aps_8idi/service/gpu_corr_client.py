"""Module for GPU correlation client service.

This module provides functions for submitting jobs, checking server status,
and batching job submissions for GPU correlation processing.
"""

import argparse
import logging
import socket
import traceback
from typing import Any, Optional

import zmq

server_ip = "164.54.100.186"
port = 5556

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect(f"tcp://{server_ip}:{port}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(name)-12s |%(levelname)s| %(message)s",
    datefmt="%m-%d %H:%M:%S",
)

logger = logging.getLogger("boost_client")


def submit_job(
    raw: Optional[Any] = None,
    qmap: Optional[Any] = None,
    output: Optional[str] = None,
    verbose: bool = False,
    cmd: str = "Multitau",
) -> Any:
    """Submit a job to the GPU correlation server.

    Parameters:
        raw (Optional[Any]): Raw input file or folder.
        qmap (Optional[Any]): Qmap file or identifier.
        output (Optional[str]): Output directory or file name.
        verbose (bool): Flag to enable verbose logging.
        cmd (str): Command type for the job (default 'Multitau').

    Returns:
        Any: The server response or job identifier.
    """
    payload = {
        "cmd": cmd,
        "raw": raw,
        "qmap": qmap,
        "output": output,
        "verbose": verbose,
    }

    if len(payload) == 0:
        return None

    if payload["cmd"] == "Status":
        socket.send_json({"cmd": "Status"})
        status = socket.recv_json()
        logger.info(status["status"])
        return status

    try:
        socket.send_json(payload)
    except Exception:
        logger.error(f"Failed to submit job to [{server_ip}:{port}]")
        traceback.print_exc()
    else:
        logger.info("Job submitted")
    return


def check_status() -> Any:
    """Check the status of the GPU correlation server.

    Returns:
        Any: The status response from the server.
    """
    socket.send_json({"cmd": "Status"})
    status = socket.recv_json()
    logger.info(status["status"])
    return status


def batch_jobs(raw_fname: str) -> Any:
    """Submit a batch of jobs based on a file listing raw file names.

    Parameters:
        raw_fname (str): The path to the file containing raw file names.

    Returns:
        Any: The response of the batch job submission.
    """
    qmap = "/scratch/xpcs_data_raw/qmap/foster202110_qmap_RubberDonut_Lq0_S180_18_D18_18.h5"
    output = "/scratch/cluster_results"

    with open(raw_fname, "r") as f:
        for line in f:
            # remove the ending /n
            raw = line[:-1]
            submit_job(raw, qmap, output)
            break


description = "XPCS-Boost Client - Submit jobs to GPU clusters"
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
    "-r",
    "--raw",
    metavar="RAW_FILENAME",
    type=str,
    required=False,
    help="the filename of the raw data file (imm/rigaku)",
)

parser.add_argument(
    "-q",
    "--qmap",
    metavar="QMAP_FILENAME",
    required=False,
    type=str,
    help="the filename of the qmap file (hdf)",
)

parser.add_argument(
    "-o",
    "--output",
    metavar="OUTPUT_DIR",
    type=str,
    required=False,
    default="cluster_results",
    help="""[default: cluster_results] the output directory
                            for the result file. If not exit, the program will
                            create this directory.""",
)

parser.add_argument(
    "-c",
    "--cmd",
    metavar="CMD",
    type=str,
    required=False,
    default="Multitau",
    help="""
                        [default: "Multitau"] Analysis CMD: ["Multitau",
                        "Twotime", "Both", "Status"].
                    """,
)

parser.add_argument(
    "--verbose", "-v", default=False, action="store_true", help="verbose"
)

args = parser.parse_args()
kwargs = vars(args)

if len(kwargs) == 0:
    return None

if kwargs["cmd"] == "Status":
    check_status()
else:
    for x in ("raw", "qmap", "output"):
        assert x in kwargs.keys()
    submit_job(**kwargs)
