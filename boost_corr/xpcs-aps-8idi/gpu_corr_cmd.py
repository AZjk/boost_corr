import argparse
import json
from gpu_corr_multitau import solve_multitau
from gpu_corr_twotime import solve_twotime


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


description = ("Compute Multi-tau/Twotime correlation for XPCS datasets on "
               "GPU/CPU")
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
                    required=True,
                    type=str,
                    help="the filename of the qmap file (h5/hdf)")

parser.add_argument("-o",
                    "--output",
                    metavar="OUTPUT_DIR",
                    type=str,
                    required=False,
                    default="cluster_results",
                    help="""[default: cluster_results] the output directory
                            for the result file. If not exit, the program will
                            create this directory.""")

parser.add_argument("-s",
                    "--smooth",
                    metavar="SMOOTH",
                    type=str,
                    required=False,
                    default="sqmap",
                    help="""[default: sqmap] smooth method to be used in
                            Twotime correlation. """)

parser.add_argument("-i",
                    "--gpu_id",
                    metavar="GPU_ID",
                    type=int,
                    default=-1,
                    help="""[default: -1] choose which GPU to use. if the input
                            is -1, then CPU is used""")

parser.add_argument("-begin_frame",
                    type=int,
                    default=1,
                    help="""[default: 1] begin_frame specifies which frame to
                            begin with for the correlation. This is useful to
                            get rid of the bad frames in the beginning.""")

parser.add_argument("-end_frame",
                    type=int,
                    default=-1,
                    help="""[default: -1] end_frame specifies the last frame
                            used for the correlation. This is useful to
                            get rid of the bad frames in the end. If -1 is
                            used, end_frame will be set to the number of
                            frames, i.e. the last frame""")

parser.add_argument("-stride_frame",
                    type=int,
                    default=1,
                    help="""[default: 1] stride_frame defines the stride.""")

parser.add_argument("-avg_frame",
                    type=int,
                    default=1,
                    help="""[default: 1] stride_frame defines the number of
                            frames to be averaged before the correlation.""")


parser.add_argument("-t",
                    "--type",
                    metavar="TYPE",
                    type=str,
                    required=False,
                    default="Multitau",
                    help="""
                        [default: "Multitau"] Analysis type: ["Multitau",
                        "Twotime", "Both"].""")

parser.add_argument("-dq",
                    "--dq_selection",
                    metavar="TYPE",
                    type=str,
                    required=False,
                    default="all",
                    help="""
                        [default: "all"] dq_selection: a string that select
                        the dq list, eg. '1, 2, 5-7' selects [1,2,5,6,7].
                        If 'all', all dynamic qindex will be used. """)

parser.add_argument("--verbose",
                    "-v",
                    default=False,
                    action="store_true",
                    help="verbose")

parser.add_argument("--dryrun",
                    "-dr",
                    default=False,
                    action="store_true",
                    help="dryrun: only show the argument without execution.")

parser.add_argument("--overwrite",
                    "-ow",
                    default=False,
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
kwargs['dq_selection'] = convert_to_list(kwargs['dq_selection'])

if args.config is not None:
    config_fname = kwargs.pop("config")
    with open(config_fname) as f:
        config = json.load(f)

    common_keys = set(kwargs.keys()) & set(config.keys())
    for key in common_keys:
        if kwargs[key] != config[key]:
            del config[key]
    kwargs.update(config)

if kwargs['dryrun']:
    ans = 'dryrun_only'
    print(json.dumps(kwargs, indent=4))
else:
    kwargs.pop('dryrun')
    atype = kwargs.pop('type')
    if atype == 'Multitau':
        ans = solve_multitau(**kwargs)
    elif atype == 'Twotime':
        ans = solve_twotime(**kwargs)
    else:
        raise ValueError(f'Analysis type [{atype}] not supported.')

print(ans)
