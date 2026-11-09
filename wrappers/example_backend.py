"""Template for an infrastructure wrapper.

This directory is the *only* place infrastructure belongs. The `ffcl` package is deliberately
provider-free: it has no cloud SDK in requirements.txt, no bucket paths, no instance types, no SSH.
It runs experiments in-process and writes JSONL. Everything about *where* that happens is your
wrapper's business.

Copy this file, edit the three marked steps, put it somewhere on PYTHONPATH, and run:

    PYTHONPATH=wrappers python -m ffcl run geometry --backend example_backend:ExampleBackend

--------------------------------------------------------------------------------------------------
The contract you are implementing
--------------------------------------------------------------------------------------------------
A backend is one method:

    def run(self, experiment: str, cfg, shard: str = "") -> int

  experiment  name of the experiment to run, e.g. "geometry" (see `python -m ffcl list`)
  cfg         a Config; `cfg.to_dict()` is JSON-serialisable and round-trips through
              `--config file.json`, so it is the whole payload a remote worker needs
  shard       "" or "i/n"; a worker given "i/n" runs every n-th cell
  returns     a process-style exit code (0 = success)

Why sharding is all the coordination you need: every experiment appends one JSONL record per
finished cell to `cfg.out_dir` and skips cells already present. So workers never need to talk to
each other, and a worker that dies can simply be re-run -- it resumes where it stopped. Point N
workers at the same shared directory, or at N private directories and merge afterwards with
`ffcl.runner.merge_results` (exposed as `python -m ffcl merge`).

--------------------------------------------------------------------------------------------------
What the remote side needs
--------------------------------------------------------------------------------------------------
  * a GPU with >=48 GB for the 8B/14B experiments (the smoke suite runs on CPU)
  * `pip install -r requirements.txt`
  * this package on PYTHONPATH (ship the directory, or `pip install .`)
  * network access to the HuggingFace hub, or a pre-populated HF_HOME cache
  * a writable path for `--out-dir`

If you would rather not write Python at all, you do not have to. The built-in command backend wraps
any scheduler that can run a shell command:

    python -m ffcl run geometry --backend command \
        --command-template 'srun -G1 python -m ffcl run {experiment} --config {config} --shard {shard}'
"""
from __future__ import annotations

import json
import os
import tempfile

from ffcl.runner import Backend


class ExampleBackend(Backend):
    """Fan an experiment out over N remote workers.

    The body below is intentionally written against a fictional `submit()` / `wait()` /
    `download()` trio so the shape is clear without pulling in any provider SDK. Replace those
    three calls; leave the surrounding logic alone.
    """

    #: How many workers to shard across. Read from the environment so the same wrapper serves a
    #: laptop and a cluster without edits.
    n_workers = int(os.environ.get("FFCL_WORKERS", "8"))

    def run(self, experiment: str, cfg, shard: str = "") -> int:
        # ---- step 1: materialise the config -----------------------------------------------------
        # cfg.to_dict() is the complete run specification. Write it once and stage it wherever your
        # workers can read it (object store, shared filesystem, or inline in the job payload).
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cfg.to_dict(), f, indent=2)
            cfg_path = f.name
        print(f"[example] staged config at {cfg_path}")

        # An explicit --shard on the CLI means "I am one worker, just do my slice" -- do not fan out
        # again, or the shards compose and most cells get dropped.
        shards = [shard] if shard else [f"{i}/{self.n_workers}"
                                        for i in range(self.n_workers)]

        # ---- step 2: submit one job per shard ---------------------------------------------------
        jobs = []
        for sh in shards:
            command = (f"pip install -r requirements.txt && "
                       f"python -m ffcl run {experiment} "
                       f"--config {cfg_path} --shard {sh} --out-dir {cfg.out_dir}")
            # jobs.append(submit(command, gpus=1, gpu_memory_gb=48, image="pytorch/pytorch:2.7"))
            print(f"[example] would submit: {command}")

        # ---- step 3: wait, then collect ---------------------------------------------------------
        # Two habits worth keeping, both learned the hard way:
        #   * poll for the JSONL rather than for job status. Result files are the ground truth and
        #     they appear incrementally, so you see progress and you can diagnose a stall.
        #   * always tear the workers down in a finally: block, including on KeyboardInterrupt.
        failures = 0
        try:
            for job in jobs:
                pass  # rc = wait(job); failures += (rc != 0)
            # download(remote_out_dir, cfg.out_dir)   # no-op if out_dir is already shared storage
        finally:
            pass  # terminate(jobs)

        # Merge only if your workers wrote to private directories:
        #   from ffcl.runner import merge_results
        #   merge_results([f"{cfg.out_dir}.w{i}" for i in range(self.n_workers)], cfg.out_dir)

        if failures:
            print(f"[example] {failures} shard(s) failed -- re-run the same command to resume; "
                  f"finished cells are skipped automatically")
        return 1 if failures else 0
