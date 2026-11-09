"""Statistics: the deconfounding protocol, the additive-model null, and CL metrics.

Single source of truth for every number the paper reports.

Three fixes over the original analysis:

1. CLUSTER BOOTSTRAP (fixes pseudo-replication).
   The forward-pass predictors p_mag and rep_cos are computed once on the BASE model per
   (capability, task) cell, then replicated across intensity levels and seeds. The original
   bootstrap resampled ROWS, so with 32 distinct predictor values spread over 256 rows the CIs
   were ~sqrt(256/32) = 2.8x too narrow. We aggregate to one observation per cell and bootstrap
   over CELLS.

2. ADDITIVE-MODEL NULL for the task x capability interaction (fixes a saturated test).
   The original null permuted task labels within capability, which destroys the task MAIN EFFECT
   as well as the interaction and dumps that variance into the interaction term. The null
   interaction share was therefore systematically inflated, so above-threshold observations came
   out at p_perm = 1.000 -- i.e. BELOW the entire null band, not "inside" it as the paper said.
   Here the null preserves both main effects: fit the additive model
       f_ij = mu + a_i + b_j
   and resample its residuals. Null datasets then have the same capability and task marginals and
   zero true interaction, which is the hypothesis actually being tested.

3. NORMALISED DRIFT.
   Raw D = sum_m ||dW_m||_F is a sum of per-module norms, so it grows with the number of adapted
   modules. The original compared D across attention-only (4 modules/layer) vs +MLP (7/layer) and
   across models with 28, 32 and 48 layers, then asserted a single threshold at ~10^3. We report
   D per adapted parameter alongside raw D.
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple
import numpy as np


# ------------------------------------------------------------------ variance decomposition
def variance_decomposition(rows: Sequence[dict], key: str = "forget_gen",
                           cap: str = "bench", task: str = "s2") -> Dict[str, float]:
    """Two-way decomposition of `key` into capability / task / interaction / residual (% of total)."""
    F = np.array([r[key] for r in rows], float)
    g = F.mean(); tot = float(((F - g) ** 2).sum())
    if tot < 1e-12:
        return dict(capability=float("nan"), task=float("nan"),
                    interaction=float("nan"), residual=float("nan"))

    def ss(k):
        d = defaultdict(list)
        for r in rows: d[r[k]].append(r[key])
        return sum(len(v) * (np.mean(v) - g) ** 2 for v in d.values())

    ss_c, ss_t = ss(cap), ss(task)
    cm = {b: np.mean([r[key] for r in rows if r[cap] == b]) for b in {r[cap] for r in rows}}
    tm = {s: np.mean([r[key] for r in rows if r[task] == s]) for s in {r[task] for r in rows}}
    cells = defaultdict(list)
    for r in rows: cells[(r[cap], r[task])].append(r[key])
    ss_i = sum(len(v) * (np.mean(v) - (cm[b] + tm[s] - g)) ** 2 for (b, s), v in cells.items())
    return dict(capability=100 * ss_c / tot, task=100 * ss_t / tot,
                interaction=100 * ss_i / tot, residual=100 * (tot - ss_c - ss_t - ss_i) / tot)


def additive_null_interaction(rows: Sequence[dict], key: str = "forget_gen",
                              cap: str = "bench", task: str = "s2",
                              n: int = 2000, seed: int = 0):
    """Null distribution of interaction% under the ADDITIVE model (zero true interaction, same
    capability and task main effects). Returns (observed, null_array, p_one_sided).

    This replaces the original within-capability label permutation, which also destroyed the task
    main effect and therefore produced an inflated, saturated null.
    """
    rng = np.random.RandomState(seed)
    obs = variance_decomposition(rows, key, cap, task)["interaction"]
    F = np.array([r[key] for r in rows], float)
    g = F.mean()
    cm = {b: np.mean([r[key] for r in rows if r[cap] == b]) for b in {r[cap] for r in rows}}
    tm = {s: np.mean([r[key] for r in rows if r[task] == s]) for s in {r[task] for r in rows}}
    fitted = np.array([cm[r[cap]] + tm[r[task]] - g for r in rows])
    resid = F - fitted
    null = np.empty(n)
    for k in range(n):
        perm = rng.permutation(resid)
        sim = [dict(r, **{key: float(fitted[i] + perm[i])}) for i, r in enumerate(rows)]
        null[k] = variance_decomposition(sim, key, cap, task)["interaction"]
    return obs, null, float((null >= obs).mean())


def within_cap_permutation_interaction(rows: Sequence[dict], key: str = "forget_gen",
                                       cap: str = "bench", task: str = "s2",
                                       n: int = 2000, seed: int = 0):
    """The ORIGINAL null, retained only so the revision can show what it does and why it is
    mis-calibrated (it destroys the task main effect, inflating the null interaction share)."""
    rng = np.random.RandomState(seed)
    obs = variance_decomposition(rows, key, cap, task)["interaction"]
    B = np.array([r[cap] for r in rows], dtype=object)
    S = np.array([r[task] for r in rows], dtype=object)
    null = np.empty(n)
    for k in range(n):
        Sp = S.copy()
        for b in np.unique(B):
            idx = np.where(B == b)[0]
            Sp[idx] = rng.permutation(S[idx])
        sim = [dict(r, **{task: Sp[i]}) for i, r in enumerate(rows)]
        null[k] = variance_decomposition(sim, key, cap, task)["interaction"]
    return obs, null, float((null >= obs).mean())


# ------------------------------------------------------------------ correlations
def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(spearmanr(x, y)[0])


def partial_spearman(x, y, z) -> float:
    a, b, c = spearman(x, y), spearman(x, z), spearman(y, z)
    if any(math.isnan(v) for v in (a, b, c)):
        return float("nan")
    return (a - b * c) / math.sqrt(max(1e-9, (1 - b ** 2) * (1 - c ** 2)))


def to_cells(rows: Sequence[dict], fields: Sequence[str],
             cap: str = "bench", task: str = "s2") -> List[dict]:
    """Collapse rows to one observation per (capability, task) cell by averaging `fields`.

    This is the correct unit of analysis for predictors that are base-model properties: they take
    a single value per cell no matter how many levels and seeds are pooled.
    """
    agg = defaultdict(list)
    for r in rows: agg[(r[cap], r[task])].append(r)
    out = []
    for (b, s), rs in sorted(agg.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        d = {cap: b, task: s, "n_rows": len(rs)}
        for f in fields:
            vals = [r[f] for r in rs if f in r and r[f] is not None]
            d[f] = float(np.mean(vals)) if vals else float("nan")
        out.append(d)
    return out


def cluster_bootstrap_partial(cells: Sequence[dict], xf: str, yf: str, zf: str,
                              n: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    """Partial Spearman of x,y|z with a 95% CI bootstrapped over CELLS (not rows)."""
    x = np.array([c[xf] for c in cells], float)
    y = np.array([c[yf] for c in cells], float)
    z = np.array([c[zf] for c in cells], float)
    r0 = partial_spearman(x, y, z)
    rng = np.random.RandomState(seed); vals = []
    for _ in range(n):
        i = rng.randint(0, len(x), len(x))
        v = partial_spearman(x[i], y[i], z[i])
        if not math.isnan(v): vals.append(v)
    if not vals:
        return r0, float("nan"), float("nan")
    return r0, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def gate(rho: float, lo: float, hi: float, thresh: float = 0.5) -> bool:
    """The paper's pre-declared gate: |rho| >= thresh AND the CI excludes zero."""
    if any(math.isnan(v) for v in (rho, lo, hi)):
        return False
    return abs(rho) >= thresh and (lo > 0 or hi < 0)


def within_capability_corr(rows: Sequence[dict], xf: str, yf: str = "forget_gen",
                           cap: str = "bench", task: str = "s2") -> Tuple[float, List[float]]:
    """Across-task rank correlation computed separately within each fixed capability, then
    averaged. Removes capability identity entirely, so it isolates the pure interaction signal."""
    cs = []
    for b in sorted({r[cap] for r in rows}, key=str):
        sub = [r for r in rows if r[cap] == b]
        agg = defaultdict(list)
        for r in sub: agg[r[task]].append((r[xf], r[yf]))
        xs = [np.mean([a for a, _ in v]) for v in agg.values()]
        ys = [np.mean([b2 for _, b2 in v]) for v in agg.values()]
        c = spearman(xs, ys)
        if not math.isnan(c): cs.append(c)
    return (float(np.mean(cs)) if cs else float("nan")), cs


# ------------------------------------------------------------------ drift normalisation
LORA_TARGETS_ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_TARGETS_MLP = ["gate_proj", "up_proj", "down_proj"]

# (n_layers, d_model, d_kv, d_ffn) per model, for analytic LoRA parameter counts
MODEL_DIMS = {
    "Meta-Llama-3.1-8B": dict(n_layers=32, d_model=4096, d_kv=1024, d_ffn=14336),
    "Llama-3.2-3B":      dict(n_layers=28, d_model=3072, d_kv=1024, d_ffn=8192),
    "Qwen2.5-7B":        dict(n_layers=28, d_model=3584, d_kv=512,  d_ffn=18944),
    "Qwen2.5-14B":       dict(n_layers=48, d_model=5120, d_kv=1024, d_ffn=13824),
    "Mistral-7B":        dict(n_layers=32, d_model=4096, d_kv=1024, d_ffn=14336),
}


def lora_param_count(model_key: str, r: int, mlp: bool) -> int:
    """LoRA parameters = sum over adapted modules of r*(d_in + d_out).

    Needed to normalise D, which is a SUM of per-module Frobenius norms and therefore scales with
    the number of adapted modules -- so raw D is not comparable between attention-only and
    +MLP adapters, nor between models of different depth.
    """
    d = MODEL_DIMS[model_key]
    dm, dkv, dff, nl = d["d_model"], d["d_kv"], d["d_ffn"], d["n_layers"]
    per_layer = (
        r * (dm + dm) +      # q_proj : d_model -> d_model
        r * (dm + dkv) +     # k_proj : d_model -> d_kv
        r * (dm + dkv) +     # v_proj : d_model -> d_kv
        r * (dm + dm)        # o_proj : d_model -> d_model
    )
    if mlp:
        per_layer += (
            r * (dm + dff) +   # gate_proj
            r * (dm + dff) +   # up_proj
            r * (dff + dm)     # down_proj
        )
    return per_layer * nl


def n_adapted_modules(model_key: str, mlp: bool) -> int:
    return MODEL_DIMS[model_key]["n_layers"] * (7 if mlp else 4)


def mean_std(vals) -> Tuple[float, float]:
    v = np.asarray(vals, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


# ------------------------------------------------------------------ continual-learning metrics
def cl_metrics(R, order):
    """Standard CL metrics from an accuracy matrix R[i][j] = accuracy on task order[j] after
    training stage i.

      ACC = mean_j R[T-1][j]                     final average over all learned tasks
      BWT = mean_{j<T-1} R[T-1][j] - R[j][j]     backward transfer (negative = interference)
      forgetting[j] = max_i R[i][j] - R[T-1][j]  per-task peak-to-final drop
    """
    T = len(order)
    acc = float(np.mean([R[T - 1][j] for j in range(T)]))
    bwt = (float(np.mean([R[T - 1][j] - R[j][j] for j in range(T - 1)])) if T > 1 else 0.0)
    forget = {order[j]: float(max(R[i][j] for i in range(j, T)) - R[T - 1][j])
              for j in range(T - 1)}
    return dict(ACC=acc, BWT=bwt, forgetting=forget)
