import sys
import json
import traceback
import argparse
import logging
from boost_corr.xpcs_aps_8idi.gpu_corr_multitau import solve_multitau
from boost_corr.xpcs_aps_8idi.gpu_corr_twotime import solve_twotime
from boost_corr.xpcs_aps_8idi.gpu_record import get_gpu, release_gpu


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s T+%(relativeCreated)05dms [%(filename)s]: %(message)s",
    datefmt="%m-%d %H:%M:%S")


def convert_to_list(input_str: str):
    """
    convert a string to a list of ints
    """
    if input_str == 'all':
        return None

    result = []
    for part in input_str.split(','):
        if part == '':
            continue
        elif '-' in part:
            a, b = part.split('-')
            result.extend(range(int(a), int(b) + 1))
        else:
            result.append(int(part))
    result = sorted(list(set(result)))
    return result


default_config = {
    "qmap": None,   # use a valid qmap fname here
    "output": "cluster_results",
    "smooth": "sqmap",
    "gpu_id": -1,
    "begin_frame": 1,
    "end_frame": -1,
    "stride_frame": 1,
    "avg_frame": 1,
    "type": "Multitau",
    "dq_selection": "all",
    "verbose": False,
    "dryrun": False,
    "overwrite": False,
    "save_G2": False,
}

description = ("Compute Multi-tau/Twotime correlation for APS-8IDI XPCS "
               "datasets on GPU/CPU")
parser = argparse.ArgumentParser(description=description)

parser.add_argument("-r",
                    "--raw",
                    metavar="RAW_FILENAME",
                    type=str,
                    required=True,
                    help="the filename of the raw data file (imm/rigaku/hdf)")

parser.add_argument("-q",
                    "--qmap",
                    metavar="QMAP_FILENAME",
                    default=default_config['qmap'],
                    required=False,
                    type=str,
                    help="the filename of the qmap file (h5/hdf)")

parser.add_argument("-o",
                    "--output",
                    metavar="OUTPUT_DIR",
                    type=str,
                    required=False,
                    default=default_config['output'],
                    help="""[default: cluster_results] the output directory
                            for the result file. If not exit, the program will
                            create this directory.""")

parser.add_argument("-s",
                    "--smooth",
                    metavar="SMOOTH",
                    type=str,
                    required=False,
                    default=default_config['smooth'],
                    help="""[default: sqmap] smooth method to be used in
                            Twotime correlation. """)

parser.add_argument("-i",
                    "--gpu_id",
                    metavar="GPU_ID",
                    type=int,
                    default=default_config['gpu_id'],
                    help="""[default: -1] choose which GPU to use. if the input
                            is -1, then CPU is used""")

parser.add_argument("-begin_frame",
                    type=int,
                    default=default_config['begin_frame'],
                    help="""[default: 1] begin_frame specifies which frame to
                            begin with for the correlation. This is useful to
                            get rid of the bad frames in the beginning.""")

parser.add_argument("-end_frame",
                    type=int,
                    default=default_config['end_frame'],
                    help="""[default: -1] end_frame specifies the last frame
                            used for the correlation. This is useful to
                            get rid of the bad frames in the end. If -1 is
                            used, end_frame will be set to the number of
                            frames, i.e. the last frame""")

parser.add_argument("-stride_frame",
                    type=int,
                    default=default_config['stride_frame'],
                    help="""[default: 1] stride_frame defines the stride.""")

parser.add_argument("-avg_frame",
                    type=int,
                    default=default_config['avg_frame'],
                    help="""[default: 1] stride_frame defines the number of
                            frames to be averaged before the correlation.""")


parser.add_argument("-t",
                    "--type",
                    metavar="TYPE",
                    type=str,
                    required=False,
                    default=default_config['type'],
                    help="""
                        [default: "Multitau"] Analysis type: ["Multitau",
                        "Twotime", "Both"].""")

parser.add_argument("-dq",
                    "--dq_selection",
                    metavar="TYPE",
                    type=str,
                    required=False,
                    default=default_config['dq_selection'],
                    help="""
                        [default: "all"] dq_selection: a string that select
                        the dq list, eg. '1, 2, 5-7' selects [1,2,5,6,7].
                        If 'all', all dynamic qindex will be used. """)

parser.add_argument("--verbose",
                    "-v",
                    default=default_config['verbose'],
                    action="store_true",
                    help="verbose")

parser.add_argument("--save_G2",
                    default=default_config['save_G2'],
                    action="store_true",
                    help="save G2, IP, IF to file")

parser.add_argument("--dryrun",
                    "-dr",
                    default=default_config['dryrun'],
                    action="store_true",
                    help="dryrun: only show the argument without execution.")

parser.add_argument("--overwrite",
                    "-ow",
                    default=default_config['overwrite'],
                    action="store_true",
                    help="whether to overwrite the existing result file.")

parser.add_argument("-c",
                    "--config",
                    metavar="CONFIG.JSON",
                    type=str,
                    required=False,
                    help="""
                        configuration file to be used. if the same key is
                        passed as an argument, the value in the configure file
                        will be omitted.
                    """)


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

kwargs['dq_selection'] = convert_to_list(kwargs['dq_selection'])

# automatically obtain gpu id
gpu_id_auto = None
if kwargs['gpu_id'] == -2:
    gpu_id_auto = get_gpu()
    kwargs['gpu_id'] = gpu_id_auto


def main():
    flag = 0

    if kwargs['dryrun']:
        ans = 'dryrun_only'
        print(json.dumps(kwargs, indent=4))
    else:
        kwargs.pop('dryrun')
        atype = kwargs.pop('type')
        if atype == 'Multitau':
            method = solve_multitau
        elif atype == 'Twotime':
            method = solve_twotime
        else:
            flag = 1
            raise ValueError(f'Analysis type [{atype}] not supported.')

        ans = None
        try:
            ans = method(**kwargs)
        except Exception:
            flag = 1
            traceback.print_exc()

        # relase gpu_id_auto
        if gpu_id_auto is not None:
            release_gpu(gpu_id_auto)

    # send the result's fname to std-out
    print(ans)
    sys.exit(flag)


if __name__ == '__main__':
    sys.exit(main())
