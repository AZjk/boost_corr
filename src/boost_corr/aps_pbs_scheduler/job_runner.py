import json
import logging
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
import boost_corr

from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

VERSION = boost_corr.__version__

_PBS_CONFIG_PATH = Path.home() / ".boost_corr" / "pbs_scheduler.json"
_REQUIRED_PBS_KEYS = ("PBS_SUBMIT_SERVER", "PBS_SUBMIT_USER", "PBS_QSUB_PATH")


def _load_pbs_config() -> dict:
    if not _PBS_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"PBS config not found: {_PBS_CONFIG_PATH}\n"
            f"Create it with keys: {', '.join(_REQUIRED_PBS_KEYS)}\n"
            "See pbs_scheduler.example.json in the source repository for a template."
        )
    with open(_PBS_CONFIG_PATH) as f:
        config = json.load(f)
    missing = [k for k in _REQUIRED_PBS_KEYS if k not in config]
    if missing:
        raise KeyError(f"Missing keys in {_PBS_CONFIG_PATH}: {', '.join(missing)}")
    return config


# CLI flags that are store_true (passed as --flag with no value)
_STORE_TRUE_FLAGS = {"verbose", "save_G2", "overwrite"}

# kwargs not forwarded to the boost_corr CLI
_SKIP_KWARGS = {"raw", "gpu_id", "config"}


def _find_executable():
    """Return the absolute path to the boost_corr executable, raising RuntimeError if not found."""
    exe = shutil.which("boost_corr")
    if exe:
        return exe
    # Fall back to the same directory as the current Python interpreter
    exe = Path(sys.executable).parent / "boost_corr"
    if exe.exists():
        return str(exe)
    raise RuntimeError(
        "boost_corr executable not found. Ensure it is installed and available in PATH."
    )


def _build_cli_args(kwargs):
    """Convert a kwargs dict into a boost_corr CLI argument string.

    Handles store_true flags (emitted only when True), normalize_frame (0/1),
    dq_selection (list -> comma string or "all"), and skips None values and
    keys listed in _SKIP_KWARGS.
    """
    parts = []
    for key, value in kwargs.items():
        if key in _SKIP_KWARGS:
            continue
        flag = "--" + key.replace("_", "-")
        if key in _STORE_TRUE_FLAGS:
            if value:
                parts.append(flag)
        elif key == "normalize_frame":
            parts.extend([flag, "1" if value else "0"])
        elif key == "dq_selection":
            if value is None:
                parts.extend([flag, "all"])
            else:
                parts.extend([flag, ",".join(str(x) for x in value)])
        elif value is None:
            continue
        else:
            parts.extend([flag, str(value)])
    return " ".join(parts)


def _create_pbs_scripts(
    raw=None, pbs_batch_size=32, pbs_gpus=4, pbs_queue="XpcsLowQ", **kwargs
):
    """Generate PBS job scripts for a set of raw data files.

    Raw files are divided into batches; each batch becomes one PBS script placed
    under a timestamped directory (pbs_jobs_YYYYMMDD_HHMMSS/).  The effective
    batch size is ``min(pbs_batch_size, max(1, len(files) // pbs_gpus))``.

    Args:
        raw: List of raw file paths, or a single-element list containing a .txt
            file with one path per line.
        pbs_batch_size: Maximum number of raw files per PBS job.
        pbs_gpus: Number of GPUs available; used to auto-scale batch size.
        pbs_queue: PBS queue name (e.g. "XpcsLowQ" or "AiLowQ").
        **kwargs: Remaining boost_corr CLI arguments forwarded to each script.

    Returns:
        List of Path objects pointing to the generated job scripts.
    """
    if raw is None:
        raise ValueError("raw input is required to create runnable scripts")

    if len(raw) == 1 and raw[0].endswith(".txt"):
        with open(raw[0], "r") as f:
            filenames = [line.strip() for line in f if line.strip()]
    else:
        filenames = raw

    # Remove gpu_id if present, since it's set to 0 in the generated scripts
    kwargs.pop("gpu_id", None)

    batch_size = min(pbs_batch_size, max(1, len(filenames) // pbs_gpus))
    output_dir = kwargs.get("output", "cluster_results")
    executable_path = _find_executable()
    cli_args = _build_cli_args(kwargs)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs_dir = Path(f"pbs_jobs_{run_id}")
    jobs_dir.mkdir(exist_ok=True)
    logger.info(f"Jobs dir: {jobs_dir}/")

    batches = [
        filenames[i : i + batch_size] for i in range(0, len(filenames), batch_size)
    ]
    job_paths = []
    for batch_id, batch in tqdm(
        enumerate(batches, start=1),
        desc="Generating scripts",
        unit="script",
        total=len(batches),
    ):
        job_label = f"job_{batch_id:05d}"
        file_list_str = " ".join(batch)

        script_content = f"""#!/bin/sh
#PBS -N boost-corr-pbs
#PBS -q {pbs_queue}
#PBS -l mem=49152mb
#PBS -l ncpus=4
#PBS -l ngpus=1
#PBS -l gdata=true
#PBS -o {output_dir}/pbs_logs_{run_id}/{job_label}.o.log
#PBS -e {output_dir}/pbs_logs_{run_id}/{job_label}.e.log
# auto-generated by boost_corr APS PBS scheduler; do not edit manually
# version: {VERSION}

mkdir -p {output_dir}/pbs_logs_{run_id}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
{executable_path} --raw {file_list_str} --gpu-id 0 {cli_args}
"""
        job_path = jobs_dir / f"{job_label}.sh"
        with open(job_path, "w") as job_file:
            job_file.write(script_content)
        job_path.chmod(0o755)
        job_paths.append(job_path)

    logger.info(
        f"{len(filenames)} raw datasets -> {len(job_paths)} PBS jobs "
        f"(batch size: {batch_size}, queue: {pbs_queue})"
    )
    return job_paths


def _copy_pbs_scripts(job_paths: list):
    """Compress the jobs directory and transfer it to the PBS submit server.

    Creates a .tar.gz archive of the jobs directory, copies it to /tmp/ on
    PBS_SUBMIT_SERVER via scp, extracts it there, then removes the tarball
    on both sides.
    """
    if not job_paths:
        logger.warning("No job scripts to copy.")
        return

    config = _load_pbs_config()
    jobs_dir = job_paths[0].parent
    tarball_path = jobs_dir.with_suffix(".tar.gz")
    remote = f"{config['PBS_SUBMIT_USER']}@{config['PBS_SUBMIT_SERVER']}"

    logger.info(f"Compressing {jobs_dir.name}/ -> {tarball_path.name} ...")
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(jobs_dir, arcname=jobs_dir.name)

    logger.info(f"Copying {tarball_path.name} to {remote}:/tmp/ ...")
    subprocess.run(
        ["scp", "-q", str(tarball_path), f"{remote}:/tmp/{tarball_path.name}"],
        check=True,
    )

    logger.info(f"Extracting on {remote} ...")
    subprocess.run(
        [
            "ssh",
            remote,
            f"tar -xzf /tmp/{tarball_path.name} -C /tmp && rm /tmp/{tarball_path.name}",
        ],
        check=True,
    )

    tarball_path.unlink()


def _submit_pbs_scripts(job_paths: list):
    """Submit all PBS job scripts via a single SSH session to PBS_SUBMIT_SERVER.

    All chmod and qsub calls are batched into one SSH round-trip to avoid
    per-job connection overhead.  Job IDs returned by qsub are logged line
    by line.
    """
    if not job_paths:
        logger.warning("No job scripts to submit.")
        return

    config = _load_pbs_config()
    remote = f"{config['PBS_SUBMIT_USER']}@{config['PBS_SUBMIT_SERVER']}"
    remote_dir = f"/tmp/{job_paths[0].parent.name}"
    remote_paths = " ".join(f"{remote_dir}/{p.name}" for p in job_paths)

    # Submit all jobs in a single SSH session to avoid per-job handshake overhead
    cmd = f"chmod +x {remote_paths} && for f in {remote_paths}; do {config['PBS_QSUB_PATH']} $f; done"
    logger.info(f"Submitting {len(job_paths)} jobs via {remote} ...")
    result = subprocess.run(["ssh", remote, cmd], capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Submission failed: {result.stderr.strip()}")
        return

    job_ids = result.stdout.strip().splitlines()
    for job_path, job_id in zip(job_paths, job_ids):
        logger.info(f"  {job_path.name} -> {job_id}")
    logger.info(f"Submitted {len(job_ids)}/{len(job_paths)} jobs successfully.")


def run_pbs_jobs(**kwargs):
    """Generate, transfer, and submit PBS jobs for a boost_corr run.

    Entry point called by the CLI when --gpu-id -3 is specified.  Delegates to
    _create_pbs_scripts, _copy_pbs_scripts, and _submit_pbs_scripts in order.
    All boost_corr CLI kwargs are forwarded; PBS-specific kwargs (pbs_batch_size,
    pbs_queue) are consumed here and not passed to the remote boost_corr command.
    """
    job_paths = _create_pbs_scripts(**kwargs)
    _copy_pbs_scripts(job_paths)
    _submit_pbs_scripts(job_paths)
