"""Command-line interface: `python -m ffcl <command>`.

    list                          show experiments and their paper mapping
    plan  <exp>                   print the cells an experiment would run (no compute)
    run   <exp> [--shard i/n]     run one experiment
    all                           run every experiment in order, then analyze
    smoke                         end-to-end pipeline test in minutes, CPU-friendly
    analyze [--latex]             results -> paper numbers (+ LaTeX tables)
    figures                       results -> PDFs
    doctor                        check the environment and dataset/model access

Every command takes --config (YAML/JSON) and inline overrides such as --model, --out-dir, --seeds.
Nothing here assumes any particular infrastructure: use `--backend command` to wrap the CLI in
srun / qsub / docker / ssh / any batch system, or `--backend my_module:MyBackend` to plug in your own
submission logic (template: wrappers/example_backend.py).
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import ExperimentConfig, load_config


def _add_common(sp):
    sp.add_argument("--config", default=None, help="YAML/JSON config file")
    sp.add_argument("--model", default=None, help="registry alias or HF model id")
    sp.add_argument("--out-dir", default=None, dest="out_dir")
    sp.add_argument("--tag", default=None, help="suffix for result filenames (e.g. the model name)")
    sp.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    sp.add_argument("--corpus-family", default=None, dest="corpus_family",
                    choices=["real", "synthetic"])
    sp.add_argument("--seeds", default=None, help="comma-separated, e.g. 0,1,2")
    sp.add_argument("--levels", default=None, help="comma-separated, e.g. L1,L3")
    sp.add_argument("--corpora", default=None, help="comma-separated corpus names")
    sp.add_argument("--conditions", default=None, help="comma-separated condition names")
    sp.add_argument("--step-scale", type=float, default=None, dest="step_scale",
                    help="<1 shortens every schedule (for quick checks)")
    sp.add_argument("--eval-items", type=int, default=None, dest="eval_items")


def _cfg_from_args(a) -> ExperimentConfig:
    over = {}
    for k in ("model", "out_dir", "tag", "device", "corpus_family", "step_scale", "eval_items"):
        v = getattr(a, k, None)
        if v is not None:
            over[k] = v
    for k in ("seeds", "levels", "corpora", "conditions"):
        v = getattr(a, k, None)
        if v:
            items = [x.strip() for x in v.split(",") if x.strip()]
            over[k] = [int(x) for x in items] if k == "seeds" else items
    return load_config(getattr(a, "config", None), **over)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ffcl", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show experiments and their paper mapping")

    for name, helptext in (("plan", "print the cells an experiment would run"),
                           ("run", "run one experiment")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("experiment")
        _add_common(sp)
        if name == "run":
            sp.add_argument("--shard", default="", help="i/n -- run every n-th cell (worker i)")
            sp.add_argument("--backend", default="local",
                            help="'local', 'command', or 'module.path:ClassName' for your own "
                                 "Backend subclass (see wrappers/example_backend.py)")
            sp.add_argument("--command-template", default=None, dest="command_template",
                            help="e.g. 'srun -G1 python -m ffcl run {experiment} --config {config}'")

    sp = sub.add_parser("all", help="run every experiment, then analyze")
    _add_common(sp)
    sp.add_argument("--only", default=None, help="comma-separated subset of experiments")
    sp.add_argument("--shard", default="")

    sp = sub.add_parser("smoke", help="end-to-end pipeline test in minutes (CPU-friendly)")
    sp.add_argument("--out-dir", default="results_smoke", dest="out_dir")
    sp.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    sp.add_argument("--model", default=None)

    sp = sub.add_parser("analyze", help="results -> paper numbers")
    _add_common(sp)
    sp.add_argument("--latex", action="store_true", help="also write tables.tex")

    sp = sub.add_parser("figures", help="results -> PDFs")
    _add_common(sp)

    sp = sub.add_parser("merge", help="merge per-worker result dirs into one")
    sp.add_argument("in_dirs", nargs="+", help="directories written by sharded workers")
    sp.add_argument("--out-dir", required=True, dest="out_dir")

    sp = sub.add_parser("doctor", help="check environment and data/model access")
    _add_common(sp)
    return p


PAPER_MAP = {
    "intensity":  "Table 1  intensity ladder (set --corpus-family synthetic for the comparison)",
    "geometry":   "Table 2  13-condition frontier + the norm-matched head-to-head",
    "crossmodel": "Table 4  two most aggressive intensities; run once per model",
    "published":  "Table 5  four forecasters from the literature",
    "fullft":     "Table 6  full fine-tuning with drift measured",
    "sequential": "Table 7  three-/five-task real sequences (len(corpora) sets the length)",
    "capbroad":   "Sec 5.1  eighteen-capability coverage check",
    "cl_stream":  "Sec 5.1  AG News -> DBpedia -> Yelp -> TREC classification stream",
    "probe":      "Sec 5.6  multi-depth probes + log-prob-scored rehearsal",
    "locus":      "Sec 5.7  layer-band restriction, drift-matched",
}
# Order matters only for readability of a single-machine `all` run.
ALL_ORDER = ["intensity", "geometry", "published", "fullft", "probe", "locus",
             "sequential", "crossmodel", "capbroad", "cl_stream"]


def cmd_list() -> int:
    from .experiments import REGISTRY
    print("experiment    paper location and description")
    print("-" * 88)
    for name in ALL_ORDER:
        mark = " " if name in REGISTRY else "?"
        print(f"{mark}{name:12s} {PAPER_MAP.get(name, '')}")
    print("\nForecasting (Table 3) is computed by `analyze` from the intensity/crossmodel logs;\n"
          "it needs no separate run because the predictors are recorded per cell.")
    return 0


def cmd_plan(a) -> int:
    from .experiments import REGISTRY
    from .config import FULLFT_LEVELS
    cfg = _cfg_from_args(a)
    exp = a.experiment
    if exp not in REGISTRY:
        raise SystemExit(f"unknown experiment '{exp}'. try `python -m ffcl list`")
    if exp in ("intensity", "crossmodel", "capbroad"):
        cells = [(lv, s, sd) for lv in cfg.levels for s in cfg.corpora for sd in cfg.seeds]
        label = "(level, corpus, seed)"
    elif exp == "geometry":
        cells = [(c, s, sd) for c in cfg.conditions for s in cfg.corpora for sd in cfg.seeds]
        label = "(condition, corpus, seed)"
    elif exp == "fullft":
        cells = [(lv, s, sd) for lv in FULLFT_LEVELS for s in cfg.corpora for sd in cfg.seeds]
        label = "(level, corpus, seed)"
    elif exp == "published":
        cells = [(s, sd) for s in cfg.corpora for sd in cfg.seeds]
        label = "(corpus, seed)"
    elif exp in ("sequential", "cl_stream"):
        cells = [(rg, sd) for rg in ("mild", "cat") for sd in cfg.seeds]
        label = "(regime, seed) x task orders"
    else:
        cells = [(lv, s, sd) for lv in cfg.levels for s in cfg.corpora for sd in cfg.seeds]
        label = "(level/band, corpus, seed)"
    print(f"{exp}: {len(cells)} cells {label}")
    print(f"  model      {cfg.model_id}")
    print(f"  corpora    {cfg.corpora} [{cfg.corpus_family}]")
    print(f"  results    {cfg.results_path(exp)}")
    for c in cells[:12]:
        print(f"    {c}")
    if len(cells) > 12:
        print(f"    ... and {len(cells) - 12} more")
    return 0


def cmd_run(a) -> int:
    from .runner import get_backend
    cfg = _cfg_from_args(a)
    backend = get_backend(a.backend, getattr(a, "command_template", None))
    return backend.run(a.experiment, cfg, shard=a.shard)


def cmd_all(a) -> int:
    from .experiments import REGISTRY
    cfg = _cfg_from_args(a)
    names = [x.strip() for x in a.only.split(",")] if a.only else ALL_ORDER
    names = [n for n in names if n in REGISTRY]
    print(f"[all] running {len(names)} experiments: {names}")
    for n in names:
        print(f"\n{'=' * 88}\n[all] {n}: {PAPER_MAP.get(n, '')}\n{'=' * 88}")
        try:
            REGISTRY[n](cfg, shard=a.shard)
        except SystemExit as e:
            print(f"[all] {n} stopped: {e}")
        except Exception as e:
            print(f"[all] {n} FAILED: {type(e).__name__}: {e}")
    from .analyze import analyze
    analyze(cfg, latex=True)
    from .figures import figures
    figures(cfg)
    return 0


def cmd_smoke(a) -> int:
    """A complete run of every experiment type at toy scale. Validates the pipeline, not the paper."""
    from .config import smoke_config
    from .experiments import REGISTRY
    from .analyze import analyze
    from .figures import figures
    cfg = smoke_config(a.out_dir)
    if a.device:
        cfg = cfg.replace(device=a.device)
    if a.model:
        cfg = cfg.replace(model=a.model)
    print(f"[smoke] model={cfg.model_id} device={cfg.device} out={cfg.out_dir}")
    print("[smoke] NOTE: toy sizes -- this checks that the pipeline runs, not the paper's numbers.")
    for n in ["intensity", "geometry", "probe", "locus", "sequential", "published"]:
        print(f"\n----- smoke: {n}")
        try:
            REGISTRY[n](cfg)
        except Exception as e:
            print(f"[smoke] {n} FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            return 1
    analyze(cfg, latex=True)
    figures(cfg)
    print("\n[smoke] PASSED -- every experiment, the analysis and the figures ran end to end.")
    return 0


def cmd_doctor(a) -> int:
    cfg = _cfg_from_args(a)
    ok = True
    print("== packages")
    for m in ("torch", "transformers", "peft", "datasets", "sklearn", "scipy", "numpy",
              "matplotlib"):
        try:
            mod = __import__(m)
            print(f"   {m:14s} {getattr(mod, '__version__', '?')}")
        except Exception as e:
            print(f"   {m:14s} MISSING ({type(e).__name__})"); ok = False
    print("== compute")
    try:
        from .model import pick_device, pick_dtype
        dev = pick_device(cfg.device)
        print(f"   device={dev} dtype={pick_dtype(dev, cfg.dtype)}")
        if dev == "cpu":
            print("   NOTE: CPU works for `smoke` only; the full suite needs a GPU.")
    except Exception as e:
        print(f"   device check failed: {e}"); ok = False
    print("== data access (downloads a few rows)")
    try:
        from .data import load_benchmarks
        b = load_benchmarks(cfg.capabilities[:1], 2, quiet=True)
        print(f"   capability benchmarks: {'OK' if b else 'FAILED'}")
        ok = ok and bool(b)
    except Exception as e:
        print(f"   capability load failed: {e}"); ok = False
    try:
        from .data import load_real_corpora
        c = load_real_corpora(cfg.corpora[:1], 24, 8, quiet=True)
        print(f"   fine-tuning corpora:   {'OK' if c else 'FAILED'}")
        ok = ok and bool(c)
    except Exception as e:
        print(f"   corpus load failed: {e}"); ok = False
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} -- "
          f"next: `python -m ffcl smoke`")
    return 0 if ok else 1


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.command == "list":
        return cmd_list()
    if a.command == "plan":
        return cmd_plan(a)
    if a.command == "run":
        return cmd_run(a)
    if a.command == "all":
        return cmd_all(a)
    if a.command == "smoke":
        return cmd_smoke(a)
    if a.command == "doctor":
        return cmd_doctor(a)
    if a.command == "merge":
        from .runner import merge_results
        merge_results(a.in_dirs, a.out_dir); return 0
    cfg = _cfg_from_args(a)
    if a.command == "analyze":
        from .analyze import analyze
        analyze(cfg, latex=a.latex); return 0
    if a.command == "figures":
        from .figures import figures
        figures(cfg); return 0
    raise SystemExit(f"unknown command {a.command}")


if __name__ == "__main__":
    sys.exit(main())
