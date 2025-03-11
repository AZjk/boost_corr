import zmq
import traceback
import logging
import argparse

server_ip = "164.54.100.186"
port = 5556

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect(f"tcp://{server_ip}:{port}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(name)-12s |%(levelname)s| %(message)s',
    datefmt='%m-%d %H:%M:%S')

logger = logging.getLogger('boost_client')


def submit_job(raw=None,
               qmap=None,
               output=None,
               verbose=False,
               cmd='Multitau'):
    payload = {
        'cmd': cmd,
        'raw': raw,
        'qmap': qmap,
        'output': output,
        'verbose': verbose
    }
    try:
        socket.send_json(payload)
    except Exception:
        logger.error(f"Failed to submit job to [{server_ip}:{port}]")
        traceback.print_exc()
    else:
        logger.info('Job submitted')
    return


def check_status():
    socket.send_json({'cmd': 'Status'})
    status = socket.recv_json()
    logger.info(status['status'])


def batch_jobs(raw_fname):
    qmap = '/scratch/xpcs_data_raw/qmap/foster202110_qmap_RubberDonut_Lq0_S180_18_D18_18.h5'
    output = '/scratch/cluster_results'

    raw_files = []
    with open(raw_fname, 'r') as f:
        for line in f:
            # remove the ending /n
            raw = line[:-1]
            submit_job(raw, qmap, output)
            break


description = "XPCS-Boost Client - Submit jobs to GPU clusters"
parser = argparse.ArgumentParser(description=description)

parser.add_argument("-r",
                    "--raw",
                    metavar="RAW_FILENAME",
                    type=str,
                    required=False,
                    help="the filename of the raw data file (imm/rigaku)")

parser.add_argument("-q",
                    "--qmap",
                    metavar="QMAP_FILENAME",
                    required=False,
                    type=str,
                    help="the filename of the qmap file (hdf)")

parser.add_argument("-o",
                    "--output",
                    metavar="OUTPUT_DIR",
                    type=str,
                    required=False,
                    default="cluster_results",
                    help="""[default: cluster_results] the output directory
                            for the result file. If not exit, the program will
                            create this directory.""")

parser.add_argument("-c",
                    "--cmd",
                    metavar="CMD",
                    type=str,
                    required=False,
                    default="Multitau",
                    help="""
                        [default: "Multitau"] Analysis CMD: ["Multitau",
                        "Twotime", "Both", "Status"].
                    """)

parser.add_argument("--verbose",
                    "-v",
                    default=False,
                    action="store_true",
                    help="verbose")

args = parser.parse_args()
kwargs = vars(args)

if len(kwargs) == 0:
    exit

if kwargs['cmd'] == 'Status':
    check_status()
else:
    for x in ('raw', 'qmap', 'output'):
        assert x in kwargs.keys()
    submit_job(**kwargs)
