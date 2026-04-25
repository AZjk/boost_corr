"""GPU and PBS schedulers used by the boost_corr CLI.

Sub-packages:
    simple_gpu_scheduler — file-lock based per-host GPU allocator (GPUScheduler)
    aps_pbs_scheduler    — APS PBS job generator/submitter (run_pbs_jobs)

Imported on demand from `cli.py`; nothing is eagerly re-exported here so a
plain ``import boost_corr.scheduler`` stays free of pynvml / network deps.
"""
