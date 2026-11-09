"""Analysis: JSONL result logs -> every number the paper reports, as text, JSON and LaTeX.

Reads whatever result files exist in `out_dir` and skips sections whose inputs are absent, so it is
useful mid-run as well as at the end.

The three statistical choices that matter are implemented in `ffcl.metrics` and used here:
  * partial Spearman controlling for baseline competence (forgetting is floor- and ceiling-bounded,
    so any predictor correlated with competence predicts forgetting for free);
  * a CLUSTER bootstrap over (capability, task) cells, because base-model predictors take one value
    per cell however many levels and seeds are pooled -- resampling rows is overconfident AND
    attenuated;
  * an ADDITIVE-MODEL null for the interaction share, because permuting task labels destroys the
    task main effect and inflates the null.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .metrics import (additive_null_interaction, cluster_bootstrap_partial, gate, mean_std,
                      partial_spearman, spearman, to_cells, variance_decomposition,
                      within_capability_corr)

PREDICTORS = ("p_mag", "early_slope", "rep_cos")
PUBLISHED = ("jinren_repdot", "logit_ip", "taskvec_cos", "fisher_overlap")


# --------------------------------------------------------------------------- loading helpers
def load(path: Path) -> List[dict]:
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def rows_of(recs: List[dict], level: Optional[str] = None) -> List[dict]:
    """Flatten records to one row per (record, capability)."""
    out = []
    for r in recs:
        if level and r.get("level") != level:
            continue
        for b, v in r.get("per_bench", {}).items():
            out.append(dict(bench=b, s2=r["s2"], seed=r.get("seed", 0), lvl=r.get("level"),
                            drift=r.get("drift"), **v))
    return out


def _h(t: str) -> None:
    print("\n" + "=" * 92 + f"\n{t}\n" + "=" * 92)


def _files(out_dir: str) -> Dict[str, List[Path]]:
    """Group result files by experiment, tolerating the `_tag` suffix (one file per model)."""
    d = Path(out_dir)
    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in sorted(d.glob("*.jsonl")):
        groups[p.stem.split("_")[0]].append(p)
    return groups


# --------------------------------------------------------------------------- sections
def sec_intensity(files, out) -> None:
    paths = files.get("intensity", [])
    if not paths:
        return
    _h("INTENSITY LADDER  (Table 1)")
    res = {}
    for p in paths:
        recs = load(p)
        fam = recs[0].get("corpus_family", "real") if recs else "real"
        print(f"\n  {p.name}  [{fam} corpora, {len(recs)} cells, "
              f"tasks={sorted({r['s2'] for r in recs})}]")
        print(f"    {'lvl':4s} {'D':>9s} {'D/param':>10s} {'mean f':>9s} {'acc gen':>8s} "
              f"{'acc lp':>7s} {'inter%':>7s} {'p_add':>6s} {'acq':>6s} {'nonletter':>9s}")
        for lv in ("L1", "L2", "L3", "L4"):
            rs = rows_of(recs, lv)
            if not rs:
                continue
            sub = [r for r in recs if r.get("level") == lv]
            mc = [r for r in rs if r["bench"] != "gsm8k"]
            vd = variance_decomposition(rs)
            _, _, pa = additive_null_interaction(rs)
            runs = defaultdict(list)
            for r in rs:
                runs[(r["s2"], r["seed"])].append(r["forget_gen"])
            f_mu, f_sd = mean_std([sum(v) / len(v) for v in runs.values()])
            e = dict(drift=_m(sub, "drift"), drift_per_param=_m(sub, "drift_per_param"),
                     forget=(f_mu, f_sd), acc_gen=_m(rs, "after_gen"),
                     acc_lp=_m(mc, "after_lp") if mc else float("nan"),
                     interaction=vd["interaction"], p_additive=pa,
                     acq=_m(sub, "s2_acq_heldout"),
                     nonletter=_m(mc, "after_frac_nonletter") if mc else float("nan"))
            res[f"{p.stem}/{lv}"] = e
            print(f"    {lv:4s} {e['drift']:9.1f} {e['drift_per_param']:10.2e} "
                  f"{f_mu:+9.3f} {e['acc_gen']:8.3f} {e['acc_lp']:7.3f} "
                  f"{e['interaction']:7.1f} {pa:6.3f} {e['acq']:6.3f} {e['nonletter']:9.3f}")
    out["intensity"] = res
    print("\n  Read the interaction column with p_add: a share of a few percent that is BELOW its\n"
          "  additive null means there is no relational structure left to forecast; a share of\n"
          "  8-14% with p_add < 0.05 alongside large forgetting is the intermediate regime.")


def _m(rows, key):
    import numpy as np
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def sec_geometry(files, out) -> None:
    paths = files.get("geometry", [])
    if not paths:
        return
    import numpy as np
    _h("MAGNITUDE vs GEOMETRY  (Table 2) -- including the NORM-MATCHED head-to-head")
    recs = [r for p in paths for r in load(p)]
    if not recs:
        return
    base = _m(recs, "base_mean_lp")
    degenerate = base < 0.15
    if degenerate:
        print(f"  NOTE base capability is {base:.3f} -- too low for '% of base' to be meaningful\n"
              f"  (expected if you are running the smoke model). Ratios are suppressed.")
    g = defaultdict(list)
    for r in recs:
        g[r["cond"]].append(r)
    order = ["naive", "gpm_e50", "gpm_e95", "gpm_e99", "olora_lo", "olora_mid", "olora_hi",
             "ewc_lo", "ewc_mid", "ewc_hi", "replay", "es_early", "lr_cap",
             "drift_cap", "gpm_cap", "olora_cap"]
    conds = [c for c in order if c in g] + sorted(set(g) - set(order))
    # Retention here is over the multiple-choice capabilities only (see experiments.RETENTION_EXCLUDE):
    # GSM8K is generatively scored, so in the collapse regime it conflates capability loss with format
    # breakdown. Per-benchmark numbers for every capability remain in the records.
    print(f"  base capability (log-prob, multiple-choice only) = {base:.4f}   "
          f"n={len(recs)} cells\n")
    print(f"  {'condition':11s} {'n':>2s} {'acq(held-out)':>14s} {'retention':>14s} {'%base':>6s} "
          f"{'D':>8s} {'steps':>6s}  constraint binds?")
    res = {}
    for c in conds:
        s = g[c]
        acq = mean_std([r["s2_acq_heldout"] for r in s])
        ret = mean_std([r["mean_retention_lp"] for r in s])
        d = _m(s, "drift"); st = _m(s, "steps_taken")
        dg = s[0].get("diag", {}) or {}
        binds = []
        # `gradenergy_removed` is the fraction of Frobenius energy the projection removes. Under the
        # parameter projection (protect.project_A_params) the projected step is the one that reaches
        # the weights, so this is a statement about the APPLIED update -- which is the only version
        # of the number worth reporting. `updenergy_leaked` is the residual an adaptive optimizer
        # would have left inside the protected subspace had we projected the gradient only; it is
        # printed so the reader can see the size of the bug the parameter projection closes.
        for k, lbl in (("gradenergy_removed", "upd-energy removed"),
                       ("updenergy_leaked", "leak closed"),
                       ("penalty_mean", "penalty")):
            v = [r["diag"][k] for r in s if (r.get("diag") or {}).get(k) is not None]
            if v:
                binds.append(f"{lbl}={np.mean(v):.3f}" if k == "updenergy_leaked"
                             else f"{lbl}={np.mean(v):.2f}")
        res[c] = dict(n=len(s), acq=acq, retention=ret, pct_base=100 * ret[0] / base,
                      drift=d, steps=st, diag=dg)
        pct = "  n/a" if degenerate else f"{100*ret[0]/base:5.0f}%"
        print(f"  {c:11s} {len(s):2d} {acq[0]:.3f}+-{acq[1]:.2f} {ret[0]:8.3f}+-{ret[1]:.3f} "
              f"{pct} {d:8.0f} {st:6.0f}  {', '.join(binds) or '-'}")

    matched = [c for c in ("drift_cap", "gpm_cap", "olora_cap") if c in res]
    if "drift_cap" in res and len(matched) > 1:
        print("\n  NORM-MATCHED (identical tau, so identical stopping drift):")
        for c in matched:
            v = res[c]
            pct = "" if degenerate else f" ({v['pct_base']:.0f}% of base)"
            print(f"    {c:11s} D={v['drift']:7.0f} acq={v['acq'][0]:.3f} "
                  f"ret={v['retention'][0]:.3f}{pct}")
        for c in matched[1:]:
            d = res[c]["retention"][0] - res["drift_cap"]["retention"][0]
            print(f"    geometry effect ({c}): {d:+.3f} retention "
                  f"({100*d/base:+.1f} pts of base) at "
                  f"{res[c]['drift']/max(1e-9, res['drift_cap']['drift']):.2f}x the drift")

    pts = sorted((v["drift"], v["retention"][0], c) for c, v in res.items())
    if len(pts) >= 3:
        rho = spearman([p[0] for p in pts], [p[1] for p in pts])
        viol = [(a, b) for i, a in enumerate(pts) for b in pts[i + 1:] if a[1] < b[1] - 0.02]
        print(f"\n  Spearman(drift, retention) over {len(pts)} conditions = {rho:+.2f}")
        print(f"  pairs where LOWER drift gives materially WORSE retention: {len(viol)}"
              "   <- drift is a strong but insufficient statistic")
        for a, b in viol[:5]:
            print(f"    {a[2]:11s} D={a[0]:7.0f} ret={a[1]:.3f}  <  "
                  f"{b[2]:11s} D={b[0]:7.0f} ret={b[1]:.3f}")
        out["geometry_spearman"] = rho
        out["geometry_violations"] = len(viol)
    out["geometry"] = dict(base=base, conditions=res)


def sec_forecast(files, out) -> None:
    """Table 3 + Table 5: predictors under the deconfounding protocol, at the cell level."""
    groups = []
    for name in ("intensity", "crossmodel", "capbroad"):
        for p in files.get(name, []):
            groups.append((p.stem, p))
    if not groups:
        return
    _h("RELATIONAL FORECASTING  (Table 3) -- cell-level cluster bootstrap")
    print("  Predictors computed on the base model take ONE value per (capability, task) cell, so\n"
          "  the bootstrap resamples cells. Row-level resampling is overconfident and attenuated.\n")
    res = {}
    for label, p in groups:
        recs = load(p)
        rs = [r for r in rows_of(recs) if r["lvl"] in ("L1", "L2")]
        if not rs or "p_mag" not in rs[0]:
            continue
        cells = to_cells(rs, list(PREDICTORS) + ["forget_gen", "base_gen"])
        if len(cells) < 8:
            print(f"  {label:26s} cells={len(cells):3d}  SKIPPED: a cluster bootstrap over "
                  f"< 8 cells is meaningless (need more capabilities x corpora)")
            continue
        line = f"  {label:26s} cells={len(cells):3d} rows={len(rs):4d}"
        e = dict(n_cells=len(cells), n_rows=len(rs))
        for q in PREDICTORS:
            r0, lo, hi = cluster_bootstrap_partial(cells, q, "forget_gen", "base_gen")
            gd = gate(r0, lo, hi)
            e[q] = dict(rho=r0, ci=[lo, hi], gate=gd)
            line += f"  {q}={r0:+.2f}[{lo:+.2f},{hi:+.2f}]{'*' if gd else ' '}"
        w, _ = within_capability_corr(rs, "p_mag")
        e["within_cap_p_mag"] = w
        line += f"  within-cap={w:+.2f}"
        res[label] = e
        print(line)
    print("\n  * clears the pre-declared gate: |rho| >= 0.5 with a 95% CI excluding zero.")
    print("  within-cap fixes the capability and asks whether the predictor ranks TASKS correctly;")
    print("  it is the stricter test and is weak everywhere, so predictability is claimed across")
    print("  capability x task cells, not within one fixed capability.")
    out["forecast"] = res

    for p in files.get("published", []):
        recs = load(p)
        rs = rows_of(recs)
        if not rs or "logit_ip" not in rs[0]:
            continue
        cells = to_cells(rs, list(PUBLISHED) + ["forget_gen", "base_gen"])
        print(f"\n  Published forecasters ({p.name}, {len(cells)} cells)  (Table 5):")
        pub = {}
        for f in PUBLISHED:
            r0, lo, hi = cluster_bootstrap_partial(cells, f, "forget_gen", "base_gen")
            raw = spearman([c[f] for c in cells], [c["forget_gen"] for c in cells])
            gd = gate(r0, lo, hi)
            verdict = ("CLEARS gate" + (" (WRONG SIGN)" if r0 < 0 else "")) if gd else "fails gate"
            pub[f] = dict(raw=raw, rho=r0, ci=[lo, hi], gate=gd)
            print(f"    {f:16s} raw={raw:+.2f} partial={r0:+.2f} [{lo:+.2f},{hi:+.2f}]  {verdict}")
        out["published"] = pub


def sec_fullft(files, out) -> None:
    paths = files.get("fullft", [])
    if not paths:
        return
    _h("FULL FINE-TUNING with drift measured  (Table 6)")
    recs = [r for p in paths for r in load(p)]
    if not recs:
        return
    res = {}
    print(f"  {'lvl':4s} {'D':>9s} {'D/param':>10s} {'mean f':>9s} {'acc gen':>8s} {'acc lp':>7s} "
          f"{'inter%':>7s} {'acq':>6s}")
    for lv in ("F1", "F2", "F3", "F4"):
        rs = rows_of(recs, lv); sub = [r for r in recs if r.get("level") == lv]
        if not rs:
            continue
        mc = [r for r in rs if r["bench"] != "gsm8k"]
        vd = variance_decomposition(rs)
        e = dict(drift=_m(sub, "drift"), drift_per_param=_m(sub, "drift_per_param"),
                 forget=_m(rs, "forget_gen"), acc_gen=_m(rs, "after_gen"),
                 acc_lp=_m(mc, "after_lp") if mc else float("nan"),
                 interaction=vd["interaction"], acq=_m(sub, "s2_acq_heldout"))
        res[lv] = e
        print(f"  {lv:4s} {e['drift']:9.1f} {e['drift_per_param']:10.2e} {e['forget']:+9.3f} "
              f"{e['acc_gen']:8.3f} {e['acc_lp']:7.3f} {e['interaction']:7.1f} {e['acq']:6.3f}")
    print("\n  Drift per PARAMETER is the only quantity comparable with the LoRA runs. A large\n"
          "  interaction share here alongside large forgetting means full FT forgets SELECTIVELY,\n"
          "  which a pure aggregate-magnitude account does not predict.")
    out["fullft"] = res


def sec_probe(files, out) -> None:
    paths = files.get("probe", [])
    if not paths:
        return
    import numpy as np
    _h("DESTRUCTION vs ACCESS: multi-depth probes + rehearsal  (Sec 5.6)")
    recs = [r for p in paths for r in load(p)]
    if not recs:
        return
    res = {}
    for lv in sorted({r["level"] for r in recs}):
        s = [r for r in recs if r["level"] == lv]
        model_lp = _m(s, "after_mean_lp")
        probe = _m(s, "after_best_probe")
        base_lp = _m(s, "base_mean_lp")
        base_probe = _m(s, "base_best_probe")
        st = defaultdict(list)
        for r in s:
            for e in r.get("recovery", []):
                st[e["step"]].append(e["mean_lp"])
        rec = (max(np.mean(v) for v in st.values()) - model_lp) / (base_lp - model_lp) \
            if st and base_lp > model_lp else 0.0
        res[lv] = dict(model_lp=model_lp, best_probe=probe, base_lp=base_lp,
                       base_probe=base_probe, probe_minus_model=probe - model_lp,
                       recovery_frac=rec, drift=_m(s, "drift"))
        print(f"  {lv}: model log-prob={model_lp:.3f}  best-depth probe={probe:.3f}  "
              f"(base probe {base_probe:.3f})")
        print(f"      probe MINUS model = {probe - model_lp:+.3f}   "
              f"<- what the probe recovers beyond the model itself")
        print(f"      rehearsal recovers {100*rec:.0f}% of the loss (log-prob scored), "
              f"drift {res[lv]['drift']:.0f}")
    print("\n  If severe forgetting were merely lost output ACCESS, the answer would stay decodable\n"
          "  at some depth and 'probe minus model' would be large. Near zero means the\n"
          "  representation itself is degraded.")
    out["probe"] = res


def sec_locus(files, out) -> None:
    paths = files.get("locus", [])
    if not paths:
        return
    _h("WHERE AN UPDATE DOES THE MOST DAMAGE  (Sec 5.7)")
    recs = [r for p in paths for r in load(p)]
    if not recs:
        return
    res = {}
    print(f"  {'band':6s} {'n':>2s} {'forget':>9s} {'D':>8s} {'forget per 1000 drift':>22s}")
    for b in ("b0", "b1", "b2", "b3", "full"):
        s = [r for r in recs if r["band"] == b]
        if not s:
            continue
        f = _m(s, "mean_forget"); d = _m(s, "drift")
        res[b] = dict(n=len(s), forget=f, drift=d, per_1k=1000 * f / max(1e-9, d))
        print(f"  {b:6s} {len(s):2d} {f:+9.3f} {d:8.0f} {1000*f/max(1e-9,d):22.3f}")
    print("\n  Compare bands at MATCHED drift. The all-layer condition is free to accumulate more\n"
          "  drift, so an absolute early-band-vs-all-layers comparison is not drift-matched; the\n"
          "  per-unit-drift column is the one that isolates depth.")
    out["locus"] = res


def sec_sequential(files, out) -> None:
    import numpy as np
    for name, title in (("sequential", "SEQUENTIAL CL on real corpora  (Table 7)"),
                        ("cl", "CLASSIFICATION CL STREAM  (Sec 5.1)")):
        paths = files.get(name, [])
        recs = [r for p in paths for r in load(p)]
        if not recs:
            continue
        _h(title)
        res = {}
        for reg in sorted({r["regime"] for r in recs}):
            s = [r for r in recs if r["regime"] == reg]
            if "bench_after" in s[0]:
                cap = [float(np.mean(list(r["bench_after"][-1].values()))) for r in s]
                dr = [r["drift_after"][-1] for r in s]
                base = _m(s, "base_bench_mean")
            else:
                cap = [r["cap_final"] for r in s]
                dr = [r["drift_final"] for r in s]
                base = _m(s, "base_cap_mean")
            acc = mean_std([r["ACC"] for r in s]); bwt = mean_std([r["BWT"] for r in s])
            res[reg] = dict(n=len(s), ACC=acc, BWT=bwt, cap_final=float(np.mean(cap)),
                            base=base, drift=float(np.mean(dr)))
            print(f"  {reg:5s} n={len(s):2d} ACC={acc[0]:.3f}+-{acc[1]:.2f} "
                  f"BWT={bwt[0]:+.3f}+-{bwt[1]:.2f} capability {base:.3f} -> "
                  f"{np.mean(cap):.3f}  D={np.mean(dr):.0f}")
        out[name] = res


# --------------------------------------------------------------------------- LaTeX
def write_latex(out: dict, path: str) -> None:
    """Emit the main tables as LaTeX, ready to \\input into the paper."""
    L = ["% Auto-generated by `python -m ffcl analyze --latex`. Do not edit by hand.", ""]
    it = out.get("intensity")
    if it:
        L += [r"\begin{tabular}{lrrrrrr}", r"\toprule",
              r"Level & $D$ & mean $f$ & acc (gen) & acc (lp) & inter.\% & $p_{\rm add}$ \\",
              r"\midrule"]
        for k, v in it.items():
            L.append(f"{k.split('/')[-1]} & {v['drift']:.0f} & ${v['forget'][0]:+.3f}"
                     f"\\pm{v['forget'][1]:.2f}$ & {v['acc_gen']:.3f} & {v['acc_lp']:.3f} & "
                     f"{v['interaction']:.1f} & {v['p_additive']:.3f} \\\\")
        L += [r"\bottomrule", r"\end{tabular}", ""]
    geo = out.get("geometry")
    if geo:
        L += [r"\begin{tabular}{lrrrr}", r"\toprule",
              r"Method & $D$ & S2 acq. & retention & \% base \\", r"\midrule"]
        for c, v in geo["conditions"].items():
            L.append(f"{c.replace('_', chr(92) + '_')} & {v['drift']:.0f} & "
                     f"${v['acq'][0]:.3f}\\pm{v['acq'][1]:.2f}$ & "
                     f"${v['retention'][0]:.3f}\\pm{v['retention'][1]:.2f}$ & "
                     f"{v['pct_base']:.0f}\\% \\\\")
        L += [r"\bottomrule", r"\end{tabular}", ""]
    Path(path).write_text("\n".join(L))
    print(f"[analyze] wrote {path}")


# --------------------------------------------------------------------------- entry point
def analyze(cfg, latex: bool = False) -> dict:
    files = _files(cfg.out_dir)
    if not files:
        raise SystemExit(f"no .jsonl result files in '{cfg.out_dir}'. Run an experiment first, "
                         f"e.g. `python -m ffcl smoke`.")
    print(f"[analyze] {cfg.out_dir}: "
          + ", ".join(f"{k}({len(v)})" for k, v in sorted(files.items())))
    out: dict = {}
    sec_intensity(files, out)
    sec_geometry(files, out)
    sec_forecast(files, out)
    sec_fullft(files, out)
    sec_probe(files, out)
    sec_locus(files, out)
    sec_sequential(files, out)
    jp = Path(cfg.out_dir) / "paper_numbers.json"
    jp.write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[analyze] wrote {jp}")
    if latex:
        write_latex(out, str(Path(cfg.out_dir) / "tables.tex"))
    return out
