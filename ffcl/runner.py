"""Infrastructure-agnostic execution.

The experiments are plain Python and know nothing about any cluster or cloud. To run somewhere, pick
a backend:

  LocalBackend    run in this process (needs a GPU on this machine for the full suite).
  CommandBackend  render a shell command and invoke it, so you can wrap the CLI in srun / qsub /
                  docker / ssh / any batch API. This is the generic escape hatch: anything that can
                  run a shell command can run these experiments.

Parallelism needs no coordination service. Every experiment writes one JSONL record per finished
cell and skips cells already present, and `--shard i/n` keeps every n-th cell, so N workers pointed
at the same output directory (or at separate ones, merged afterwards) partition the work safely.

    # 8 local GPUs
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i python -m ffcl run geometry --shard $i/8 --out-dir results &
    done; wait

    # Slurm array
    sbatch --array=0-7 --wrap 'python -m ffcl run geometry \
        --shard $SLURM_ARRAY_TASK_ID/8 --out-dir results'

    # any scheduler, via the command backend
    python -m ffcl run geometry --backend command \
        --command-template 'qsub -- python -m ffcl run {experiment} --config {config}'

To integrate a bespoke system, subclass `Backend` in your own module and select it by import path:

    python -m ffcl run geometry --backend my_infra.backend:MyBackend

`wrappers/example_backend.py` is a commented template that shows the three things such a wrapper
normally does: materialise the config, submit N sharded jobs, and collect the JSONL back. Nothing in
the package assumes a provider -- there is no cloud SDK in requirements.txt.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class Backend:
    def run(self, experiment: str, cfg, shard: str = "") -> int:
        raise NotImplementedError


class LocalBackend(Backend):
    """Run in-process on this machine."""

    def run(self, experiment: str, cfg, shard: str = "") -> int:
        from .experiments import REGISTRY
        if experiment not in REGISTRY:
            raise SystemExit(f"unknown experiment '{experiment}'. "
                             f"options: {sorted(REGISTRY)}")
        REGISTRY[experiment](cfg, shard=shard)
        return 0


@dataclass
class CommandBackend(Backend):
    """Run by rendering a command template.

    The template may reference {experiment}, {config} and {shard}. The config is materialised to a
    temporary JSON file so the remote side receives exactly this run's settings.
    """
    template: str

    def run(self, experiment: str, cfg, shard: str = "") -> int:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cfg.to_dict(), f)
            cfg_path = f.name
        cmd = self.template.format(experiment=experiment, config=cfg_path, shard=shard)
        print(f"[CommandBackend] {cmd}")
        use_shell = any(ch in cmd for ch in "'\"|&;<>$")
        return subprocess.call(cmd if use_shell else shlex.split(cmd), shell=use_shell)


def load_backend(spec: str) -> Backend:
    """Import a user-supplied Backend from a 'module.path:ClassName' spec.

    This is the wrapper extension point: put your own submission logic (Slurm API, Kubernetes,
    Ray, a cloud batch service, an internal scheduler) in a module on PYTHONPATH, subclass
    `Backend`, and pass `--backend your_module:YourBackend`. The package itself stays free of any
    provider dependency; see wrappers/example_backend.py for a worked template.
    """
    import importlib
    if ":" not in spec:
        raise SystemExit(f"custom backend spec must be 'module.path:ClassName', got '{spec}'")
    mod_name, cls_name = spec.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise SystemExit(f"could not import backend module '{mod_name}': {e}\n"
                         f"(is its directory on PYTHONPATH?)")
    obj = getattr(mod, cls_name, None)
    if obj is None:
        raise SystemExit(f"module '{mod_name}' has no attribute '{cls_name}'")
    backend = obj() if isinstance(obj, type) else obj
    if not isinstance(backend, Backend):
        raise SystemExit(f"{spec} is not a ffcl.runner.Backend subclass")
    return backend


def get_backend(kind: str = "local", template: Optional[str] = None) -> Backend:
    if kind == "local":
        return LocalBackend()
    if kind == "command":
        if not template:
            raise SystemExit("--backend command requires --command-template")
        return CommandBackend(template)
    if ":" in kind:
        return load_backend(kind)
    raise SystemExit(f"unknown backend '{kind}' "
                     f"(use 'local', 'command', or 'module.path:ClassName')")


def merge_results(in_dirs, out_dir) -> None:
    """Merge per-worker result directories, de-duplicating identical records.

    Useful when shards write to separate directories (e.g. per-node scratch).
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    by_name = {}
    for d in in_dirs:
        for p in sorted(Path(d).glob("*.jsonl")):
            by_name.setdefault(p.name, []).append(p)
    for name, paths in by_name.items():
        seen, lines = set(), []
        for p in paths:
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and line not in seen:
                    seen.add(line); lines.append(line)
        (out / name).write_text("\n".join(lines) + "\n")
        print(f"[merge] {name}: {len(lines)} records from {len(paths)} file(s)")
