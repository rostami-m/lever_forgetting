"""Experiment drivers: one function per result family in the paper.

Each driver is a plain Python function of a config. It writes one JSONL record per completed cell
and skips cells already present, so a run is interruptible and restartable on any machine, and can
be sharded across workers with `--shard i/n` without any coordination.

Mapping to the paper (see README for the full table):

  intensity     Table 1  the intensity ladder, real or synthetic corpora
  geometry      Table 2  the 13-condition frontier + the NORM-MATCHED head-to-head
  forecast      Table 3  the three forward-pass predictors under the deconfounding protocol
  published     Table 5  four forecasters from the literature, same protocol
  fullft        Table 6  full fine-tuning with drift actually measured
  crossmodel    Table 4  the two most aggressive intensities across models
  capbroad      Sec 5.1  the eighteen-capability coverage check
  probe         Sec 5.6  multi-depth probes + post-collapse rehearsal (log-prob scored)
  locus         Sec 5.7  layer-band restriction, drift-matched
  sequential    Table 7  three- and five-task real sequences
  cl_stream     Sec 5.1  AG News -> DBpedia -> Yelp -> TREC classification stream
"""
from __future__ import annotations

import time
from typing import Dict, List

from . import data as D
from . import eval as E
from . import lora as L
from .config import CONDITIONS, FULLFT_LEVELS, LEVELS, REGIMES, ExperimentConfig
from .io import Results
from .metrics import cl_metrics
from .model import load_model, num_layers
from .train import train_condition, train_full, train_label_masked, train_plain

REGISTRY: Dict[str, callable] = {}


def experiment(name):
    def deco(fn):
        REGISTRY[name] = fn
        fn.experiment_name = name
        return fn
    return deco


#: Capabilities excluded from the *aggregate* retention mean (never from the per-benchmark record).
#: GSM8K is generatively scored, so in the collapse regime its accuracy conflates capability loss
#: with format breakdown -- averaging it in would credit a method for emitting unparseable output.
#: The frontier/mitigation table therefore reports retention over the multiple-choice capabilities
#: only. Keep this in sync with `analyze.py`, which applies the same exclusion.
RETENTION_EXCLUDE = ("gsm8k",)


def _mc_mean(per_bench: Dict[str, float]) -> float:
    """Mean over the multiple-choice capabilities, i.e. excluding RETENTION_EXCLUDE."""
    import numpy as np
    vals = [v for k, v in per_bench.items() if k not in RETENTION_EXCLUDE]
    return float(np.mean(vals)) if vals else float("nan")


def _log(tag, msg):
    print(f"[{tag} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _shard(cells: List[dict], shard: str = "") -> List[dict]:
    """`--shard i/n` keeps every n-th cell. Lets any scheduler fan the same command across workers."""
    if not shard:
        return cells
    i, n = (int(x) for x in shard.split("/"))
    return [c for k, c in enumerate(cells) if k % n == i]


def _setup(cfg: ExperimentConfig, need_corpora: bool = True):
    tok, model, device = load_model(cfg.model_id, cfg.four_bit, cfg.device, cfg.dtype)
    benches = D.load_benchmarks(cfg.capabilities, cfg.eval_items)
    if not benches:
        raise SystemExit("no capability benchmarks loaded (check network/datasets access)")
    corpora = D.load_corpora(cfg) if need_corpora else {}
    return tok, model, device, benches, corpora


def _baseline(model, tok, benches, device, cfg):
    """Base accuracy per capability under both scorings, plus the non-letter rate."""
    out = {}
    for n, items in benches.items():
        g, l, nl = E.eval_both(model, tok, items, device, held=cfg.rehearsal_held)
        out[n] = dict(base_gen=g, base_lp=l, base_nonletter=nl)
    return out


# =========================================================================== 1. intensity ladder
@experiment("intensity")
def run_intensity(cfg: ExperimentConfig, shard: str = "", out_name: str = "intensity") -> None:
    """Table 1. Sweep FT intensity; record both scorings, drift, held-out acquisition, and the
    three forward-pass predictors. Run with corpus_family=synthetic to reproduce the comparison."""
    res = Results(cfg.results_path(out_name))
    key = lambda r: (r["model"], r["level"], r["s2"], r["seed"])
    done = res.done_keys(key)
    cells = _shard([dict(level=lv, s2=s, seed=sd) for lv in cfg.levels
                    for s in cfg.corpora for sd in cfg.seeds], shard)
    tok, model, device, benches, corpora = _setup(cfg)
    base = _baseline(model, tok, benches, device, cfg)
    _log("int", f"baseline gen={sum(v['base_gen'] for v in base.values())/len(base):.3f}")

    layer = int(num_layers(model) * cfg.mid_layer_frac)
    rep_b = {n: E.hidden_reps(model, tok, [it["prompt"] for it in items[:16]], layer, device)
             for n, items in benches.items()}
    rep_s = {s: E.hidden_reps(model, tok, c["train"][:16], layer, device)
             for s, c in corpora.items()}
    pmag = {}
    for s, c in corpora.items():
        for n, items in benches.items():
            pmag[(n, s)] = E.p_mag(model, tok, items, c["train"], device,
                                   base[n]["base_gen"], held=cfg.rehearsal_held)
        _log("int", f"p_mag computed for {s}")

    wrapped = None; cur = None
    for cell in cells:
        lv, s2, seed = cell["level"], cell["s2"], cell["seed"]
        if (cfg.model_id, lv, s2, seed) in done:
            continue
        if s2 not in corpora:
            _log("int", f"missing corpus {s2}"); continue
        lc = LEVELS[lv]; t0 = time.time()
        sig = (lc["r"], lc["alpha"], lc["mlp"])
        if cur != sig:
            if wrapped is not None:
                wrapped = wrapped.unload()
            wrapped = L.make_lora_level(model, lc); cur = sig
        scale = L.lora_scale(lc)
        npar = L.n_adapted_params(wrapped)
        steps = cfg.steps(lc["steps"])

        # early_slope: forgetting after K steps, from a fresh adapter (a per-seed quantity)
        L.reset_lora(wrapped, seed)
        train_plain(wrapped, tok, corpora[s2]["train"], min(cfg.early_k, steps), lc["lr"],
                    device, seed=seed, batch=cfg.batch_size, max_len=cfg.max_len, scale=scale)
        early = {n: base[n]["base_gen"] - E.eval_bench(wrapped, tok, items, device, "gen",
                                                       held=cfg.rehearsal_held)
                 for n, items in benches.items()}

        L.reset_lora(wrapped, seed)
        train_plain(wrapped, tok, corpora[s2]["train"], steps, lc["lr"], device, seed=seed,
                    batch=cfg.batch_size, max_len=cfg.max_len, scale=scale)
        drift = L.drift_norm(wrapped, scale)
        acq = E.acquisition_heldout(wrapped, tok, corpora[s2]["test"], device)

        pb = {}
        for n, items in benches.items():
            g, l, nl = E.eval_both(wrapped, tok, items, device, held=cfg.rehearsal_held)
            pb[n] = dict(**base[n], after_gen=g, after_lp=l, after_frac_nonletter=nl,
                         forget_gen=base[n]["base_gen"] - g, forget_lp=base[n]["base_lp"] - l,
                         early_slope=early[n], p_mag=pmag[(n, s2)],
                         rep_cos=E.rep_overlap(rep_b[n], rep_s[s2]))
        res.append(dict(model=cfg.model_id, level=lv, s2=s2, seed=seed, cfg=lc,
                        corpus_family=cfg.corpus_family, steps_taken=steps, drift=drift, drift_scale=scale,
                        drift_per_param=drift / max(1, npar), n_adapted_params=npar,
                        s2_acq_heldout=acq, secs=round(time.time() - t0, 1), per_bench=pb))
        import numpy as np
        _log("int", f"{lv}/{s2}/s{seed}: D={drift:.0f} "
                    f"f={np.mean([v['forget_gen'] for v in pb.values()]):+.3f} "
                    f"acq={acq:.3f} [{time.time()-t0:.0f}s]")
    _log("int", "done")


# =========================================================================== 2. geometry frontier
@experiment("geometry")
def run_geometry(cfg: ExperimentConfig, shard: str = "") -> None:
    """Table 2. Every mitigation condition, each swept over its strength knob, plus the
    NORM-MATCHED head-to-head (`drift_cap` vs `gpm_cap` vs `olora_cap`: identical tau, so identical
    stopping drift). This is the comparison that isolates geometry from magnitude.

    Each record carries diagnostics proving the constraint bound -- read them before believing any
    null result from a constrained condition.
    """
    res = Results(cfg.results_path("geometry"))
    key = lambda r: (r["model"], r["cond"], r["s2"], r["seed"])
    done = res.done_keys(key)
    cells = _shard([dict(cond=c, s2=s, seed=sd) for c in cfg.conditions
                    for s in cfg.corpora for sd in cfg.seeds], shard)
    tok, model, device, benches, corpora = _setup(cfg)
    rehearsal = D.rehearsal_strings(benches, cfg.rehearsal_held)

    lc = LEVELS["L3"]                                     # the collapse reference configuration
    wrapped = L.make_lora_level(model, lc)
    scale = L.lora_scale(lc); npar = L.n_adapted_params(wrapped)
    L.reset_lora(wrapped, 0)
    base = {n: dict(base_lp=E.eval_bench(wrapped, tok, it, device, "logprob",
                                         held=cfg.rehearsal_held),
                    base_gen=E.eval_bench(wrapped, tok, it, device, "gen",
                                          held=cfg.rehearsal_held))
            for n, it in benches.items()}
    import numpy as np
    base_lp = _mc_mean({n: v["base_lp"] for n, v in base.items()})
    _log("geo", f"base lp={base_lp:.4f} (MC only) adapted_params={npar}")

    for cell in cells:
        cond, s2, seed = cell["cond"], cell["s2"], cell["seed"]
        if (cfg.model_id, cond, s2, seed) in done or s2 not in corpora:
            continue
        t0 = time.time()
        L.reset_lora(wrapped, seed)
        taken, diag = train_condition(
            wrapped, tok, corpora[s2]["train"], cond, device, benches=benches,
            rehearsal=rehearsal, base_steps=cfg.steps(lc["steps"]), base_lr=lc["lr"],
            scale=scale, seed=seed, batch=cfg.batch_size, max_len=cfg.max_len)
        drift = L.drift_norm(wrapped, scale)
        ret_lp = {n: E.eval_bench(wrapped, tok, it, device, "logprob", held=cfg.rehearsal_held)
                  for n, it in benches.items()}
        ret_gen = {n: E.eval_bench(wrapped, tok, it, device, "gen", held=cfg.rehearsal_held)
                   for n, it in benches.items()}
        acq = E.acquisition_heldout(wrapped, tok, corpora[s2]["test"], device)
        mret = _mc_mean(ret_lp)
        res.append(dict(model=cfg.model_id, cond=cond, s2=s2, seed=seed, spec=CONDITIONS[cond],
                        steps_taken=taken, drift=drift, drift_scale=scale, drift_per_param=drift / max(1, npar),
                        n_adapted_params=npar, s2_acq_heldout=acq,
                        mean_retention_lp=mret,
                        mean_retention_gen=_mc_mean(ret_gen),
                        retention_excludes=list(RETENTION_EXCLUDE),
                        base_mean_lp=base_lp, per_bench_lp=ret_lp, per_bench_gen=ret_gen,
                        base=base, diag=diag, secs=round(time.time() - t0, 1)))
        binds = {k: round(v, 3) for k, v in diag.items()
                 if k in ("gradenergy_removed", "updenergy_leaked", "penalty_mean",
                          "protected_frac")
                 and isinstance(v, float)}
        pct = f"{100 * mret / base_lp:.0f}% of base" if base_lp > 0.05 else "base~0, ratio n/a"
        _log("geo", f"{cond}/{s2}/s{seed}: D={drift:.0f} steps={taken} acq={acq:.3f} "
                    f"ret={mret:.3f} ({pct}) binds={binds or '-'}")
    _log("geo", "done")


# =========================================================================== 3. published forecasters
@experiment("published")
def run_published(cfg: ExperimentConfig, shard: str = "") -> None:
    """Table 5. Four forecasters from the literature under the same deconfounding protocol:
    a representation inner product, a logit inner product, a task-vector cosine, and an EWC-style
    diagonal-Fisher overlap. All are GEOMETRIC OVERLAP measures; the paper's point is that none
    carries signal on real corpora while the functional probe (p_mag) does."""
    import numpy as np
    import torch
    res = Results(cfg.results_path("published"))
    key = lambda r: (r["model"], r["s2"], r["seed"])
    done = res.done_keys(key)
    cells = _shard([dict(s2=s, seed=sd) for s in cfg.corpora for sd in cfg.seeds], shard)
    tok, model, device, benches, corpora = _setup(cfg)
    layer = int(num_layers(model) * cfg.mid_layer_frac)
    base = _baseline(model, tok, benches, device, cfg)

    def flat_grad(m):
        return torch.cat([p.grad.detach().float().flatten() for p in m.parameters()
                          if p.requires_grad and p.grad is not None])

    lc = LEVELS["L3"]
    wrapped = L.make_lora_level(model, lc)
    scale = L.lora_scale(lc)

    # per-capability signatures on the base model: hidden states, logits, and a Fisher diagonal
    bench_hid, bench_logit, bench_fisher = {}, {}, {}
    for n, items in benches.items():
        it = E._eval_items(items, cfg.rehearsal_held)[:16]
        bench_hid[n] = E.hidden_reps(model, tok, [x["prompt"] for x in it], layer, device).mean(0)
        with torch.no_grad():
            lg = []
            for x in it:
                enc = tok(x["prompt"], return_tensors="pt", truncation=True,
                          max_length=512).to(device)
                lg.append(model(**enc).logits[0, -1, :].float().cpu().numpy())
        bench_logit[n] = np.mean(lg, 0)
        L.reset_lora(wrapped, 0)
        wrapped.zero_grad(set_to_none=True)
        ids = tok.encode(it[0]["prompt"] + " " + str(it[0]["answer"]),
                         add_special_tokens=True)[:cfg.max_len]
        wrapped(input_ids=torch.tensor([ids]).to(device),
                labels=torch.tensor([ids]).to(device)).loss.backward()
        bench_fisher[n] = (flat_grad(wrapped) ** 2).cpu().numpy()
        wrapped.zero_grad(set_to_none=True)

    def cos(a, b):
        a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
        d = min(len(a), len(b))
        a, b = a[:d], b[:d]
        return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))

    for cell in cells:
        s2, seed = cell["s2"], cell["seed"]
        if (cfg.model_id, s2, seed) in done or s2 not in corpora:
            continue
        t0 = time.time()
        train_lines = corpora[s2]["train"]
        s2_hid = E.hidden_reps(model, tok, train_lines[:16], layer, device).mean(0)
        with torch.no_grad():
            lg = []
            for t in train_lines[:16]:
                enc = tok(t, return_tensors="pt", truncation=True, max_length=512).to(device)
                lg.append(model(**enc).logits[0, -1, :].float().cpu().numpy())
        s2_logit = np.mean(lg, 0)
        L.reset_lora(wrapped, seed)
        wrapped.zero_grad(set_to_none=True)
        ids = tok.encode(train_lines[0], add_special_tokens=True)[:cfg.max_len]
        wrapped(input_ids=torch.tensor([ids]).to(device),
                labels=torch.tensor([ids]).to(device)).loss.backward()
        s2_fisher = (flat_grad(wrapped) ** 2).cpu().numpy()
        wrapped.zero_grad(set_to_none=True)

        # the task vector: the actual update this corpus produces
        L.reset_lora(wrapped, seed)
        train_plain(wrapped, tok, train_lines, cfg.steps(lc["steps"]), lc["lr"], device,
                    seed=seed, batch=cfg.batch_size, max_len=cfg.max_len, scale=scale)
        tv = torch.cat([p.detach().float().flatten() for n, p in wrapped.named_parameters()
                        if "lora_B" in n]).cpu().numpy()
        pb = {}
        for n, items in benches.items():
            g, l, nl = E.eval_both(wrapped, tok, items, device, held=cfg.rehearsal_held)
            pb[n] = dict(**base[n], after_gen=g, after_lp=l,
                         forget_gen=base[n]["base_gen"] - g, forget_lp=base[n]["base_lp"] - l,
                         jinren_repdot=cos(s2_hid, bench_hid[n]),
                         logit_ip=cos(s2_logit, bench_logit[n]),
                         taskvec_cos=cos(tv, bench_fisher[n]),
                         fisher_overlap=cos(s2_fisher, bench_fisher[n]))
        res.append(dict(model=cfg.model_id, s2=s2, seed=seed, layer=layer,
                        secs=round(time.time() - t0, 1), per_bench=pb))
        _log("pub", f"{s2}/s{seed} done [{time.time()-t0:.0f}s]")
    _log("pub", "done")


# =========================================================================== 4. full fine-tuning
@experiment("fullft")
def run_fullft(cfg: ExperimentConfig, shard: str = "") -> None:
    """Table 6. Full fine-tuning with drift MEASURED (the original runs recorded only steps and lr,
    yet the paper made a claim about drift). Reports drift per parameter, the only quantity
    comparable with the LoRA runs, which adapt ~1.7e8 parameters against ~3.2e9 here."""
    import numpy as np
    res = Results(cfg.results_path("fullft"))
    key = lambda r: (r["model"], r["level"], r["s2"], r["seed"])
    done = res.done_keys(key)
    cells = _shard([dict(level=lv, s2=s, seed=sd) for lv in FULLFT_LEVELS
                    for s in cfg.corpora for sd in cfg.seeds], shard)
    tok, model, device, benches, corpora = _setup(cfg)
    npar = sum(p.numel() for p in model.parameters())
    snap = L.snapshot_cpu(model)
    base = _baseline(model, tok, benches, device, cfg)
    _log("full", f"params={npar/1e9:.2f}B")

    for cell in cells:
        lv, s2, seed = cell["level"], cell["s2"], cell["seed"]
        if (cfg.model_id, lv, s2, seed) in done or s2 not in corpora:
            continue
        fc = FULLFT_LEVELS[lv]; t0 = time.time()
        L.restore(model, snap)
        train_full(model, tok, corpora[s2]["train"], cfg.steps(fc["steps"]), fc["lr"], device,
                   seed=seed, batch=max(1, cfg.batch_size // 2), max_len=cfg.max_len)
        drift = L.full_drift(model, snap)
        acq = E.acquisition_heldout(model, tok, corpora[s2]["test"], device)
        pb = {}
        for n, items in benches.items():
            g, l, nl = E.eval_both(model, tok, items, device, held=cfg.rehearsal_held)
            pb[n] = dict(**base[n], after_gen=g, after_lp=l, after_frac_nonletter=nl,
                         forget_gen=base[n]["base_gen"] - g, forget_lp=base[n]["base_lp"] - l)
        # Full fine-tuning has no adapter, so no alpha/r factor: drift is the direct weight delta.
        res.append(dict(model=cfg.model_id, level=lv, s2=s2, seed=seed, cfg=fc, drift=drift,
                        drift_scale=1.0,
                        drift_per_param=drift / npar, n_params=npar, s2_acq_heldout=acq,
                        secs=round(time.time() - t0, 1), per_bench=pb))
        _log("full", f"{lv}/{s2}/s{seed}: D={drift:.1f} D/par={drift/npar:.2e} "
                     f"f={np.mean([v['forget_gen'] for v in pb.values()]):+.3f}")
        L.restore(model, snap)
    _log("full", "done")


# =========================================================================== 5. probes + recovery
@experiment("probe")
def run_probe(cfg: ExperimentConfig, shard: str = "") -> None:
    """Sec 5.6. Multi-depth linear probes on the fine-tuned model, then post-collapse rehearsal.

    The recovery trajectory is LOG-PROB scored. The original reported 21% recovery gen-scored over a
    non-monotone trajectory (0.12 -> 0.02 -> 0.09 -> 0.21) because a collapsed model's outputs are
    format-broken rather than merely wrong; log-prob scored the value is far smaller.
    """
    import numpy as np
    res = Results(cfg.results_path("probe"))
    key = lambda r: (r["model"], r["level"], r["s2"], r["seed"])
    done = res.done_keys(key)
    levels = [lv for lv in cfg.levels if lv in ("L2", "L3")] or ["L2", "L3"]
    cells = _shard([dict(level=lv, s2=s, seed=sd) for lv in levels
                    for s in cfg.corpora for sd in cfg.seeds], shard)
    tok, model, device, benches, corpora = _setup(cfg)
    rehearsal = D.rehearsal_strings(benches, cfg.rehearsal_held)
    nl_model = num_layers(model)
    layers = [min(x, nl_model) for x in cfg.probe_layers]

    base_lp = {n: E.eval_bench(model, tok, it, device, "logprob", held=cfg.rehearsal_held)
               for n, it in benches.items()}
    base_probe = {}
    for n, items in benches.items():
        b, per = E.best_depth_probe(model, tok, items, layers, device, held=cfg.rehearsal_held)
        base_probe[n] = dict(best=b, per_depth=per)
    base_mean = float(np.mean(list(base_lp.values())))
    base_best = float(np.nanmean([v["best"] for v in base_probe.values()]))
    _log("probe", f"base lp={base_mean:.3f} best-depth probe={base_best:.3f}")

    for cell in cells:
        lv, s2, seed = cell["level"], cell["s2"], cell["seed"]
        if (cfg.model_id, lv, s2, seed) in done or s2 not in corpora:
            continue
        lc = LEVELS[lv]; t0 = time.time()
        wrapped = L.make_lora_level(model, lc)
        scale = L.lora_scale(lc)
        L.reset_lora(wrapped, seed)
        train_plain(wrapped, tok, corpora[s2]["train"], cfg.steps(lc["steps"]), lc["lr"], device,
                    seed=seed, batch=cfg.batch_size, max_len=cfg.max_len, scale=scale)
        drift = L.drift_norm(wrapped, scale)
        after_lp = {n: E.eval_bench(wrapped, tok, it, device, "logprob", held=cfg.rehearsal_held)
                    for n, it in benches.items()}
        after_probe = {}
        for n, items in benches.items():
            b, per = E.best_depth_probe(wrapped, tok, items, layers, device,
                                        held=cfg.rehearsal_held)
            after_probe[n] = dict(best=b, per_depth=per)

        # rehearsal on the reserved capability tail, scored log-prob at each checkpoint
        traj = []
        sched = [max(1, int(x * cfg.step_scale)) for x in (30, 90, 180, 300)]
        traj.append(dict(step=0, mean_lp=float(np.mean(list(after_lp.values()))), drift=drift))
        prev = 0
        for target in sched:
            train_plain(wrapped, tok, rehearsal, max(1, target - prev), 2e-4, device, seed=seed,
                        batch=cfg.batch_size, max_len=cfg.max_len, scale=scale)
            prev = target
            m = float(np.mean([E.eval_bench(wrapped, tok, it, device, "logprob",
                                            held=cfg.rehearsal_held)
                               for it in benches.values()]))
            traj.append(dict(step=target, mean_lp=m, drift=L.drift_norm(wrapped, scale)))

        res.append(dict(model=cfg.model_id, level=lv, s2=s2, seed=seed, layers=layers,
                        drift=drift, drift_scale=scale, base_mean_lp=base_mean, base_best_probe=base_best,
                        after_mean_lp=float(np.mean(list(after_lp.values()))),
                        after_best_probe=float(np.nanmean([v["best"] for v in after_probe.values()])),
                        base_probe=base_probe, after_probe=after_probe, recovery=traj,
                        secs=round(time.time() - t0, 1)))
        _log("probe", f"{lv}/{s2}/s{seed}: lp={np.mean(list(after_lp.values())):.3f} "
                      f"probe={np.nanmean([v['best'] for v in after_probe.values()]):.3f}")
        wrapped = wrapped.unload()
    _log("probe", "done")


# =========================================================================== 6. layer bands
@experiment("locus")
def run_locus(cfg: ExperimentConfig, shard: str = "") -> None:
    """Sec 5.7. Restrict an MLP adapter to one layer quartile at fixed rank and step budget.

    On real corpora the bands produce nearly matched drift, so the depth comparison needs no
    correction. Always read `drift` alongside `mean_forget`: the earlier version's absolute
    early-band-vs-all-layers claim was not drift-matched and does not survive.
    """
    import numpy as np
    res = Results(cfg.results_path("locus"))
    key = lambda r: (r["model"], r["band"], r["s2"], r["seed"])
    done = res.done_keys(key)
    bands = ["b0", "b1", "b2", "b3", "full"]
    cells = _shard([dict(band=b, s2=s, seed=sd) for b in bands
                    for s in cfg.corpora for sd in cfg.seeds], shard)
    tok, model, device, benches, corpora = _setup(cfg)
    nl_model = num_layers(model)
    lc = LEVELS["L3"]
    base = {n: E.eval_bench(model, tok, it, device, "gen", held=cfg.rehearsal_held)
            for n, it in benches.items()}
    base_mean = float(np.mean(list(base.values())))
    _log("locus", f"base gen={base_mean:.3f}, {nl_model} layers")

    for cell in cells:
        band, s2, seed = cell["band"], cell["s2"], cell["seed"]
        if (cfg.model_id, band, s2, seed) in done or s2 not in corpora:
            continue
        t0 = time.time()
        layers = L.band_layers(nl_model, band)
        targets = L.band_mlp_targets(model, layers)
        if not targets:
            _log("locus", f"no targets for band {band}"); continue
        wrapped = L.make_lora(model, lc["r"], lc["alpha"], targets=targets)
        scale = L.lora_scale(lc)
        L.reset_lora(wrapped, seed)
        train_plain(wrapped, tok, corpora[s2]["train"], cfg.steps(lc["steps"]), lc["lr"], device,
                    seed=seed, batch=cfg.batch_size, max_len=cfg.max_len, scale=scale)
        drift = L.drift_norm(wrapped, scale)
        pb = {}
        for n, items in benches.items():
            g = E.eval_bench(wrapped, tok, items, device, "gen", held=cfg.rehearsal_held)
            pb[n] = dict(base_gen=base[n], after_gen=g, forget_gen=base[n] - g)
        mf = float(np.mean([v["forget_gen"] for v in pb.values()]))
        res.append(dict(model=cfg.model_id, band=band, layers=[min(layers), max(layers)],
                        n_targets=len(targets), s2=s2, seed=seed, drift=drift, drift_scale=scale,
                        mean_forget=mf, base_mean=base_mean, per_bench=pb,
                        forget_per_1k_drift=1000 * mf / max(1e-9, drift),
                        secs=round(time.time() - t0, 1)))
        _log("locus", f"{band}/{s2}/s{seed}: D={drift:.0f} forget={mf:+.3f} "
                      f"per-1k-drift={1000*mf/max(1e-9,drift):.3f}")
        wrapped = wrapped.unload()
    _log("locus", "done")


# =========================================================================== 7. sequential CL
@experiment("sequential")
def run_sequential(cfg: ExperimentConfig, shard: str = "") -> None:
    """Table 7. Train real corpora sequentially on one adapter; report ACC / BWT / capability.

    The number of tasks is len(cfg.corpora), so a 3- and a 5-task stream are the same driver with a
    different corpora list.
    """
    import numpy as np
    res = Results(cfg.results_path("sequential"))
    key = lambda r: (r["model"], r["regime"], tuple(r["order"]), r["seed"])
    done = res.done_keys(key)
    tasks = list(cfg.corpora)
    orders = [tasks[i:] + tasks[:i] for i in range(min(3, len(tasks)))]
    cells = _shard([dict(regime=rg, order=o, seed=sd) for rg in ("mild", "cat")
                    for o in orders for sd in cfg.seeds], shard)
    tok, model, device, benches, corpora = _setup(cfg)
    base_bench = {n: E.eval_bench(model, tok, it, device, "logprob", held=cfg.rehearsal_held)
                  for n, it in benches.items()}
    base_mean = float(np.mean(list(base_bench.values())))
    _log("seq", f"base capability (lp)={base_mean:.3f}")

    for cell in cells:
        regime, order, seed = cell["regime"], cell["order"], cell["seed"]
        if (cfg.model_id, regime, tuple(order), seed) in done:
            continue
        if any(t not in corpora for t in order):
            continue
        rc = REGIMES[regime]; t0 = time.time()
        wrapped = L.make_lora_level(model, rc)
        scale = L.lora_scale(rc)
        L.reset_lora(wrapped, seed)
        R, cap_after, drift_after = [], [], []
        for stage, tname in enumerate(order):
            train_plain(wrapped, tok, corpora[tname]["train"], cfg.steps(rc["steps"]), rc["lr"],
                        device, seed=seed + stage, batch=cfg.batch_size, max_len=cfg.max_len,
                        scale=scale)
            R.append([E.acquisition_heldout(wrapped, tok, corpora[t]["test"], device)
                      for t in order])
            cap = {n: E.eval_bench(wrapped, tok, it, device, "logprob", held=cfg.rehearsal_held)
                   for n, it in benches.items()}
            cap_after.append(cap)
            drift_after.append(L.drift_norm(wrapped, scale))
        m = cl_metrics(R, order)
        res.append(dict(model=cfg.model_id, regime=regime, order=order, seed=seed, R=R,
                        base_bench=base_bench, base_bench_mean=base_mean,
                        bench_after=cap_after, drift_after=drift_after,
                        secs=round(time.time() - t0, 1), **m))
        _log("seq", f"{regime}|{'>'.join(order)}|s{seed}: ACC={m['ACC']:.3f} BWT={m['BWT']:+.3f} "
                    f"cap={np.mean(list(cap_after[-1].values())):.3f} D={drift_after[-1]:.0f}")
        wrapped = wrapped.unload()
    _log("seq", "done")


# =========================================================================== 8. classification stream
@experiment("cl_stream")
def run_cl_stream(cfg: ExperimentConfig, shard: str = "") -> None:
    """Sec 5.1. AG News -> DBpedia -> Yelp -> TREC: a genuinely non-SFT task-incremental stream,
    scored by label-token log-probability."""
    import numpy as np
    res = Results(cfg.results_path("cl_stream"))
    key = lambda r: (r["model"], r["regime"], r["order_name"], r["seed"])
    done = res.done_keys(key)
    tok, model, device, benches, _ = _setup(cfg, need_corpora=False)
    tasks = D.load_cl_stream(cfg.corpus_train, min(100, cfg.corpus_test))
    if not tasks:
        raise SystemExit("no CL-stream tasks available")
    names = list(tasks)
    orders = {"fwd": names, "rev": names[::-1]}
    cells = _shard([dict(regime=rg, order_name=o, seed=sd) for rg in ("mild", "cat")
                    for o in orders for sd in cfg.seeds], shard)
    base_cap = {n: E.eval_bench(model, tok, it, device, "logprob", held=cfg.rehearsal_held)
                for n, it in benches.items()}
    base_task = {n: E.eval_task_labelscore(model, tok, t, device) for n, t in tasks.items()}
    base_mean = float(np.mean(list(base_cap.values())))

    for cell in cells:
        regime, oname, seed = cell["regime"], cell["order_name"], cell["seed"]
        if (cfg.model_id, regime, oname, seed) in done:
            continue
        order = orders[oname]; rc = REGIMES[regime]; t0 = time.time()
        wrapped = L.make_lora_level(model, rc); scale = L.lora_scale(rc)
        L.reset_lora(wrapped, seed)
        acc_matrix, cap_after, drift_after = [], [], []
        for stage, tname in enumerate(order):
            train_label_masked(wrapped, tok, tasks[tname]["train"], cfg.steps(rc["steps"]),
                               rc["lr"], device, seed=seed + stage,
                               batch=max(1, cfg.batch_size // 2))
            acc_matrix.append({n: E.eval_task_labelscore(wrapped, tok, tasks[n], device)
                               for n in order})
            cap_after.append(float(np.mean(
                [E.eval_bench(wrapped, tok, it, device, "logprob", held=cfg.rehearsal_held)
                 for it in benches.values()])))
            drift_after.append(L.drift_norm(wrapped, scale))
        R = [[row[n] for n in order] for row in acc_matrix]
        m = cl_metrics(R, order)
        res.append(dict(model=cfg.model_id, regime=regime, order_name=oname, task_order=order,
                        seed=seed, base_task=base_task, base_cap=base_cap,
                        base_cap_mean=base_mean, acc_matrix=acc_matrix, cap_after=cap_after,
                        cap_final=cap_after[-1], drift_final=drift_after[-1],
                        secs=round(time.time() - t0, 1), **m))
        _log("cls", f"{regime}/{oname}/s{seed}: ACC={m['ACC']:.3f} BWT={m['BWT']:+.3f} "
                    f"cap={cap_after[-1]:.3f} D={drift_after[-1]:.0f}")
        wrapped = wrapped.unload()
    _log("cls", "done")


# =========================================================================== 9/10. aliases
@experiment("crossmodel")
def run_crossmodel(cfg: ExperimentConfig, shard: str = "") -> None:
    """Table 4. The two most aggressive intensities on one model; run once per model (set `model`
    and `tag`) and the analysis pools them. Same driver as `intensity`, restricted levels."""
    run_intensity(cfg.replace(levels=[lv for lv in cfg.levels if lv in ("L3", "L4")]
                              or ["L3", "L4"]), shard, out_name="crossmodel")


@experiment("capbroad")
def run_capbroad(cfg: ExperimentConfig, shard: str = "") -> None:
    """Sec 5.1. The eighteen-capability coverage check: `intensity` with the broad suite."""
    from .config import BROAD_CAPABILITIES
    run_intensity(cfg.replace(capabilities=list(BROAD_CAPABILITIES)), shard,
                  out_name="capbroad")
