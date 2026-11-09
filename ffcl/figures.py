"""Figures: result logs -> PDFs. Each function degrades gracefully if its inputs are missing.

  fig_retention_drift   retention vs drift over all conditions, with the norm-matched trio marked
  fig_frontier          acquisition-retention frontier, one curve per method family
  fig_intensity         the ladder: forgetting, log-prob accuracy, non-letter rate
  fig_forecast          predictor partial correlations with cluster-bootstrap CIs
  fig_probe             probe accuracy by depth vs the model's own accuracy
  fig_locus             per-band forgetting with drift annotated
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .analyze import _files, load, rows_of
from .metrics import cluster_bootstrap_partial, gate, to_cells

FAMILY = {"naive": "naive", "drift_cap": "mag", "lr_cap": "mag", "es_early": "mag",
          "replay": "mag", "ewc_lo": "imp", "ewc_mid": "imp", "ewc_hi": "imp",
          "olora_lo": "geom", "olora_mid": "geom", "olora_hi": "geom",
          "gpm_e50": "geom", "gpm_e95": "geom", "gpm_e99": "geom",
          "gpm_cap": "matched", "olora_cap": "matched"}
COLORS = dict(mag="#1f77b4", geom="#d62728", imp="#2ca02c", naive="#555555", matched="#9467bd")


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 180, "savefig.bbox": "tight", "axes.grid": True,
                         "grid.alpha": 0.25, "grid.linewidth": 0.5})
    return plt


def _geo_stats(recs):
    import numpy as np
    g = defaultdict(list)
    for r in recs:
        g[r["cond"]].append(r)
    base = float(np.mean([r["base_mean_lp"] for r in recs]))
    st = {}
    for c, s in g.items():
        st[c] = dict(
            D=float(np.mean([r["drift"] for r in s])),
            ret=float(np.mean([r["mean_retention_lp"] for r in s])),
            ret_sd=float(np.std([r["mean_retention_lp"] for r in s], ddof=1)) if len(s) > 1 else 0.,
            acq=float(np.mean([r["s2_acq_heldout"] for r in s])),
            acq_sd=float(np.std([r["s2_acq_heldout"] for r in s], ddof=1)) if len(s) > 1 else 0.)
    return st, base


def fig_retention_drift(files, outdir) -> bool:
    recs = [r for p in files.get("geometry", []) for r in load(p)]
    if not recs:
        return False
    import numpy as np
    plt = _setup_mpl()
    st, base = _geo_stats(recs)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for c, v in st.items():
        fam = FAMILY.get(c, "mag")
        ax.errorbar(max(v["D"], 1e-6), 100 * v["ret"] / base, yerr=100 * v["ret_sd"] / base,
                    fmt="o", color=COLORS[fam], ms=6, capsize=2, lw=1,
                    mec="k" if fam == "matched" else "none", mew=1.2, zorder=3)
        ax.annotate(c, (max(v["D"], 1e-6), 100 * v["ret"] / base), fontsize=6.5, va="center",
                    xytext=(4, 0), textcoords="offset points")
    trio = [c for c in ("drift_cap", "gpm_cap", "olora_cap") if c in st]
    if len(trio) > 1:
        ys = [100 * st[c]["ret"] / base for c in trio]
        xs = [st[c]["D"] for c in trio]
        ax.plot([min(xs) * 0.9, max(xs) * 1.1], [min(ys) - 1.5] * 2, color=COLORS["matched"], lw=1)
        ax.annotate("matched drift", (float(np.mean(xs)), min(ys) - 2.5),
                    color=COLORS["matched"], fontsize=7, ha="center", va="top")
    ax.set_xscale("log")
    ax.set_xlabel("total weight drift $D$ (log scale)")
    ax.set_ylabel("capability retained (% of base)")
    ax.set_title("Retention tracks drift, but is not determined by it", fontsize=9)
    handles = [plt.Line2D([], [], color=COLORS[k], marker="o", ls="", label=l) for k, l in
               [("naive", "naive"), ("mag", "magnitude-only"), ("imp", "importance (EWC)"),
                ("geom", "geometry"), ("matched", "geometry + drift cap")]]
    ax.legend(handles=handles, fontsize=6.5, loc="lower left", frameon=False)
    fig.savefig(Path(outdir) / "fig_retention_vs_drift.pdf"); plt.close(fig)
    return True


def fig_frontier(files, outdir) -> bool:
    recs = [r for p in files.get("geometry", []) for r in load(p)]
    if not recs:
        return False
    plt = _setup_mpl()
    st, base = _geo_stats(recs)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    groups = {"EWC ($\\lambda$)": ["ewc_lo", "ewc_mid", "ewc_hi"],
              "O-LoRA ($\\lambda$)": ["olora_lo", "olora_mid", "olora_hi"],
              "GPM (energy)": ["gpm_e50", "gpm_e95", "gpm_e99"]}
    cols = {"EWC ($\\lambda$)": COLORS["imp"], "O-LoRA ($\\lambda$)": COLORS["geom"],
            "GPM (energy)": "#ff7f0e"}
    for gname, cs in groups.items():
        cs = [c for c in cs if c in st]
        if len(cs) > 1:
            ax.plot([st[c]["acq"] for c in cs], [100 * st[c]["ret"] / base for c in cs],
                    "o-", color=cols[gname], ms=4, lw=1.2, label=gname)
    for c, mk, lab in [("naive", "X", "naive"), ("drift_cap", "s", "drift cap"),
                       ("gpm_cap", "*", "GPM + cap"), ("olora_cap", "P", "O-LoRA + cap")]:
        if c in st:
            ax.errorbar(st[c]["acq"], 100 * st[c]["ret"] / base, xerr=st[c]["acq_sd"],
                        yerr=100 * st[c]["ret_sd"] / base, fmt=mk,
                        color=COLORS["naive"] if c == "naive" else COLORS["matched"],
                        ms=11 if mk == "*" else 7, capsize=2, lw=1, label=lab, zorder=4)
    ax.set_xlabel("new-task acquisition (held-out split)")
    ax.set_ylabel("capability retained (% of base)")
    ax.set_title("Acquisition--retention frontier, each method swept over its knob", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.savefig(Path(outdir) / "fig_frontier.pdf"); plt.close(fig)
    return True


def fig_intensity(files, outdir) -> bool:
    paths = files.get("intensity", [])
    if not paths:
        return False
    import numpy as np
    plt = _setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.9))
    lv = ["L1", "L2", "L3", "L4"]
    x = np.arange(len(lv))
    for p, col, mk in zip(paths, ["#1f77b4", "#d62728", "#2ca02c"], ["o", "s", "^"]):
        recs = load(p)
        fam = recs[0].get("corpus_family", "real") if recs else "?"
        f, l, n = [], [], []
        for level in lv:
            rs = rows_of(recs, level)
            if not rs:
                f.append(np.nan); l.append(np.nan); n.append(np.nan); continue
            mc = [r for r in rs if r["bench"] != "gsm8k"]
            f.append(np.mean([r["forget_gen"] for r in rs]))
            l.append(np.mean([r["after_lp"] for r in mc]) if mc else np.nan)
            n.append(np.mean([r.get("after_frac_nonletter", np.nan) for r in mc]) if mc else np.nan)
        lab = f"{fam} ({p.stem})"
        axes[0].plot(x, f, mk + "-", color=col, ms=4, label=lab)
        axes[1].plot(x, l, mk + "-", color=col, ms=4)
        axes[2].plot(x, n, mk + "-", color=col, ms=4)
    axes[1].axhline(0.25, ls=":", color="k", lw=1)
    axes[1].annotate("chance", (0.05, 0.26), fontsize=6.5)
    for a, t in zip(axes, ["mean forgetting", "held-out accuracy (log-prob)",
                           "outputs that are not a valid option"]):
        a.set_xticks(x); a.set_xticklabels(lv); a.set_xlabel("fine-tuning intensity")
        a.set_title(t, fontsize=8.5)
    axes[0].legend(fontsize=6, frameon=False)
    fig.savefig(Path(outdir) / "fig_intensity.pdf"); plt.close(fig)
    return True


def fig_forecast(files, outdir) -> bool:
    groups = [(p.stem, p) for name in ("intensity", "crossmodel", "capbroad")
              for p in files.get(name, [])]
    if not groups:
        return False
    import numpy as np
    plt = _setup_mpl()
    preds = ["p_mag", "early_slope", "rep_cos"]
    cols = ["#1f77b4", "#2ca02c", "#7f7f7f"]
    data = []
    for label, p in groups:
        rs = [r for r in rows_of(load(p)) if r["lvl"] in ("L1", "L2")]
        if not rs or "p_mag" not in rs[0]:
            continue
        cells = to_cells(rs, preds + ["forget_gen", "base_gen"])
        data.append((label, [cluster_bootstrap_partial(cells, q, "forget_gen", "base_gen")
                             for q in preds]))
    if not data:
        return False
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(data), 3.2))
    w = 0.26
    for j, (q, col) in enumerate(zip(preds, cols)):
        xs = np.arange(len(data)) + (j - 1) * w
        vals = [d[1][j][0] for d in data]
        lo = [max(0, d[1][j][0] - d[1][j][1]) for d in data]
        hi = [max(0, d[1][j][2] - d[1][j][0]) for d in data]
        ax.bar(xs, vals, w, color=col, label=q, yerr=[lo, hi], capsize=2, error_kw=dict(lw=0.8))
    ax.axhline(0, color="k", lw=0.8)
    for s in (0.5, -0.5):
        ax.axhline(s, ls="--", color="#c00", lw=0.9)
    ax.annotate("gate $|\\rho|\\geq0.5$", (len(data) - 0.5, 0.53), fontsize=6.5, color="#c00",
                ha="right")
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels([d[0] for d in data], fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("partial Spearman $\\rho$ vs forgetting\n(controlling for baseline competence)")
    ax.set_title("Forward-pass predictors, cell-level with cluster-bootstrap CIs", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower left")
    fig.savefig(Path(outdir) / "fig_forecast.pdf"); plt.close(fig)
    return True


def fig_probe(files, outdir) -> bool:
    recs = [r for p in files.get("probe", []) for r in load(p)]
    if not recs:
        return False
    import numpy as np
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for lv, col in zip(sorted({r["level"] for r in recs}), ["#1f77b4", "#d62728"]):
        s = [r for r in recs if r["level"] == lv]
        layers = s[0]["layers"]
        agg = defaultdict(list)
        for r in s:
            for _, d in r["after_probe"].items():
                for L, v in d["per_depth"].items():
                    if v is not None and not np.isnan(v):
                        agg[int(L)].append(v)
        xs = sorted(agg)
        if xs:
            ax.plot(xs, [np.mean(agg[L]) for L in xs], "o-", color=col, ms=4,
                    label=f"probe on fine-tuned model ({lv})")
        ax.axhline(float(np.mean([r["after_mean_lp"] for r in s])), ls="--", color=col, lw=1,
                   label=f"model's own log-prob accuracy ({lv})")
    agg = defaultdict(list)
    for r in recs:
        for _, d in r["base_probe"].items():
            for L, v in d["per_depth"].items():
                if v is not None and not np.isnan(v):
                    agg[int(L)].append(v)
    xs = sorted(agg)
    if xs:
        ax.plot(xs, [np.mean(agg[L]) for L in xs], "s-", color="k", ms=4,
                label="probe on base model")
    ax.axhline(0.25, ls=":", color="gray", lw=1)
    ax.set_xlabel("layer probed"); ax.set_ylabel("linear-probe accuracy")
    ax.set_title("Is 'forgotten' knowledge still decodable at any depth?", fontsize=9)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.savefig(Path(outdir) / "fig_probe.pdf"); plt.close(fig)
    return True


def fig_locus(files, outdir) -> bool:
    recs = [r for p in files.get("locus", []) for r in load(p)]
    if not recs:
        return False
    import numpy as np
    plt = _setup_mpl()
    bands = [b for b in ("b0", "b1", "b2", "b3") if any(r["band"] == b for r in recs)]
    f = [np.mean([r["mean_forget"] for r in recs if r["band"] == b]) for b in bands]
    d = [np.mean([r["drift"] for r in recs if r["band"] == b]) for b in bands]
    full = [r for r in recs if r["band"] == "full"]
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    bars = ax.bar(range(len(bands)), f, width=0.62,
                  color=["#d62728", "#e8743b", "#f0b357", "#9ecae1"][:len(bands)])
    for i, (b, dd) in enumerate(zip(bars, d)):
        ax.annotate(f"$D$={dd:.0f}", (i, b.get_height()), ha="center", va="bottom", fontsize=7)
    if full:
        y = float(np.mean([r["mean_forget"] for r in full]))
        ax.axhline(y, ls="--", color="k", lw=1)
        ax.annotate(f"all layers ($D$={np.mean([r['drift'] for r in full]):.0f})",
                    (len(bands) - 0.4, y), fontsize=7, ha="right", va="bottom")
    ax.set_xticks(range(len(bands))); ax.set_xticklabels(bands)
    ax.set_xlabel("layer band the adapter is restricted to (early -> late)")
    ax.set_ylabel("mean forgetting")
    ax.set_title("Depth sensitivity; compare bands at matched drift", fontsize=9)
    fig.savefig(Path(outdir) / "fig_locus.pdf"); plt.close(fig)
    return True


ALL = [fig_retention_drift, fig_frontier, fig_intensity, fig_forecast, fig_probe, fig_locus]


def figures(cfg) -> List[str]:
    outdir = Path(cfg.out_dir) / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    files = _files(cfg.out_dir)
    made = []
    for fn in ALL:
        try:
            if fn(files, outdir):
                made.append(fn.__name__)
                print(f"[figures] {fn.__name__} -> ok")
            else:
                print(f"[figures] {fn.__name__} -> skipped (inputs missing)")
        except Exception as e:
            print(f"[figures] {fn.__name__} -> FAILED: {type(e).__name__}: {e}")
    print(f"[figures] wrote {len(made)} figure(s) to {outdir}")
    return made
