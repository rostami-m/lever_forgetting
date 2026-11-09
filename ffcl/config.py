"""Declarative experiment configuration.

Design goals: (1) a config plus an experiment name fully determines a run; (2) no infrastructure
assumptions -- paths are plain local paths and the device is auto-detected; (3) reproducible --
every seed and knob is explicit and serialised into each result record.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- capability suites
# Eight held-out capabilities: the paper's main suite.
DEFAULT_CAPABILITIES: List[str] = [
    "mmlu_high_school_mathematics", "mmlu_world_religions", "mmlu_college_computer_science",
    "mmlu_anatomy", "mmlu_international_law", "mmlu_moral_scenarios",
    "arc_challenge", "gsm8k",
]
# Eighteen capabilities: the coverage-robustness experiment (`capbroad`).
BROAD_CAPABILITIES: List[str] = DEFAULT_CAPABILITIES[:6] + [
    "mmlu_high_school_biology", "mmlu_philosophy", "mmlu_machine_learning",
    "mmlu_professional_medicine", "mmlu_formal_logic", "mmlu_us_foreign_policy",
    "mmlu_high_school_physics", "mmlu_econometrics",
    "arc_challenge", "gsm8k", "hellaswag", "winogrande",
]

# --------------------------------------------------------------------------- fine-tuning corpora
# REAL corpora are the paper's primary evidence. Each is split into disjoint train/test halves and
# new-task acquisition is *always* scored on the held-out half.
REAL_CORPORA: List[str] = ["alpaca", "dolly", "mbpp", "gsm8k_train", "pubmedqa"]

# SYNTHETIC corpora are five hand-written sentences per domain, repeated to fill the schedule.
# They are retained ONLY to reproduce the paper's synthetic-vs-real comparison, which shows they
# exaggerate collapse severity and cause format breakdown that real data does not. Do not use them
# to draw a conclusion; see the paper's corrigendum appendix.
SYNTHETIC_CORPORA: List[str] = ["code_py", "legal", "medical", "poetry", "math", "recipe"]

# --------------------------------------------------------------------------- intensity ladder
# The control variable. Each level raises rank, steps, learning rate and (from L3) adds the MLP
# modules, producing a range of measured weight drift spanning ~2 orders of magnitude.
LEVELS: Dict[str, dict] = {
    "L1": dict(r=16, alpha=32,  steps=60,  lr=2e-4, mlp=False),
    "L2": dict(r=32, alpha=64,  steps=150, lr=3e-4, mlp=False),
    "L3": dict(r=64, alpha=128, steps=300, lr=5e-4, mlp=True),
    "L4": dict(r=64, alpha=128, steps=600, lr=1e-3, mlp=True),
}
# Two-regime shorthand used by the sequential experiments.
REGIMES: Dict[str, dict] = {
    "mild": dict(r=16, alpha=32,  steps=150, lr=2e-4, mlp=False),
    "cat":  dict(r=64, alpha=128, steps=300, lr=5e-4, mlp=True),
}
# Full fine-tuning ladder (no adapter; every parameter trainable).
FULLFT_LEVELS: Dict[str, dict] = {
    "F1": dict(steps=40,  lr=1e-6),
    "F2": dict(steps=80,  lr=5e-6),
    "F3": dict(steps=150, lr=2e-5),
    "F4": dict(steps=250, lr=8e-5),
}

# --------------------------------------------------------------------------- mitigation conditions
# `kind` selects the mechanism; the remaining keys are that mechanism's strength knob.
#   plain  -- no protection (naive / drift_cap / lr_cap / es_early differ only by lr/steps/tau)
#   ewc    -- importance-weighted penalty on the MERGED update, importance from the BASE weights
#   olora  -- soft rank-k subspace-orthogonality penalty
#   gpm    -- hard rank-k activation-subspace gradient projection
#   replay -- experience replay on the capability rehearsal tail
# `tau` sets a drift cap (early-stop when measured drift exceeds it); None disables it.
CONDITIONS: Dict[str, dict] = {
    "naive":      dict(kind="plain"),
    "drift_cap":  dict(kind="plain", tau=300.0),
    "lr_cap":     dict(kind="plain", lr=2e-4),
    "es_early":   dict(kind="plain", steps=100),
    "replay":     dict(kind="replay", replay_frac=0.5),
    # EWC frontier
    "ewc_lo":     dict(kind="ewc", lam=5e2),
    "ewc_mid":    dict(kind="ewc", lam=5e3),
    "ewc_hi":     dict(kind="ewc", lam=5e4),
    # O-LoRA frontier (geometry, soft)
    "olora_lo":   dict(kind="olora", lam=2e2),
    "olora_mid":  dict(kind="olora", lam=2e3),
    "olora_hi":   dict(kind="olora", lam=2e4),
    # GPM frontier (geometry, hard); the knob is how much activation energy is protected
    "gpm_e50":    dict(kind="gpm", energy=0.50),
    "gpm_e95":    dict(kind="gpm", energy=0.95),
    "gpm_e99":    dict(kind="gpm", energy=0.99),
    # NORM-MATCHED head-to-head: identical tau, so identical stopping drift. This is the
    # comparison that isolates whether update *geometry* buys anything beyond update *magnitude*.
    "gpm_cap":    dict(kind="gpm", energy=0.95, tau=300.0),
    "olora_cap":  dict(kind="olora", lam=2e3, tau=300.0),
}
# Conditions that consume held-out capability data. The magnitude-only conditions do not, so any
# comparison across the two families is not information-matched; the paper states this explicitly.
USES_CAPABILITY_DATA = {"ewc", "olora", "gpm", "replay"}

# --------------------------------------------------------------------------- model registry
MODELS: Dict[str, str] = {
    "llama3.1-8b": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "llama3.2-3b": "unsloth/Llama-3.2-3B-Instruct",
    "qwen2.5-7b":  "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "mistral-7b":  "mistralai/Mistral-7B-Instruct-v0.3",
    # Tiny models for the smoke test: no GPU required, runs in minutes.
    "smoke":       "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smoke-tiny":  "hf-internal-testing/tiny-random-LlamaForCausalLM",
}


def resolve_model(name_or_id: str) -> str:
    """Accept either a registry alias ('llama3.1-8b') or a raw HF id."""
    return MODELS.get(name_or_id, name_or_id)


# --------------------------------------------------------------------------- the config object
@dataclass
class ExperimentConfig:
    # what to run on
    model: str = "llama3.1-8b"            # registry alias or HF model id
    device: str = "auto"                  # auto | cuda | mps | cpu
    dtype: str = "auto"                   # auto -> bf16 on cuda, fp32 on mps/cpu
    four_bit: bool = False

    # data
    capabilities: List[str] = field(default_factory=lambda: list(DEFAULT_CAPABILITIES))
    corpora: List[str] = field(default_factory=lambda: list(REAL_CORPORA))
    corpus_family: str = "real"           # real | synthetic (synthetic is for the comparison only)
    eval_items: int = 60                  # capability items per benchmark
    corpus_train: int = 400               # fine-tuning lines per corpus
    corpus_test: int = 100                # held-out lines per corpus (acquisition is scored here)
    rehearsal_held: int = 8               # tail of each benchmark reserved for protection methods
                                          # and EXCLUDED from evaluation in every condition

    # execution
    seeds: List[int] = field(default_factory=lambda: [0, 1])
    levels: List[str] = field(default_factory=lambda: ["L1", "L2", "L3", "L4"])
    conditions: List[str] = field(default_factory=lambda: [
        "naive", "drift_cap", "lr_cap", "es_early", "ewc_lo", "ewc_mid", "ewc_hi",
        "olora_lo", "olora_mid", "olora_hi", "gpm_e50", "gpm_e95", "gpm_e99",
        "gpm_cap", "olora_cap"])
    batch_size: int = 4
    max_len: int = 128
    step_scale: float = 1.0               # <1 shortens every schedule (smoke runs)
    probe_layers: List[int] = field(default_factory=lambda: [4, 8, 16, 24, 32])
    early_k: int = 10                     # steps for the early_slope predictor
    mid_layer_frac: float = 0.5           # depth for representation-overlap predictors

    # statistics
    bootstrap_n: int = 2000
    permutation_n: int = 2000
    gate_rho: float = 0.5                 # |rho| threshold for the pre-declared predictor gate

    # output
    out_dir: str = "results"
    tag: str = ""                         # suffix for the results filename (e.g. a model name)

    # ---- helpers
    @property
    def model_id(self) -> str:
        return resolve_model(self.model)

    def steps(self, n: int) -> int:
        return max(2, int(round(n * self.step_scale)))

    def results_path(self, experiment: str) -> str:
        name = f"{experiment}{('_' + self.tag) if self.tag else ''}.jsonl"
        return str(Path(self.out_dir) / name)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def replace(self, **kw) -> "ExperimentConfig":
        d = self.to_dict(); d.update(kw)
        return ExperimentConfig(**d)


# --------------------------------------------------------------------------- presets
def smoke_config(out_dir: str = "results_smoke") -> ExperimentConfig:
    """A complete end-to-end run in minutes on CPU/MPS -- no GPU and no large downloads.

    It exercises every code path the full suite uses (adapters, drift, both scorings, the
    protection machinery, probes, the analysis and the figures) at a size where correctness can be
    checked quickly. The numbers it produces are NOT the paper's; it is a pipeline test.
    """
    return ExperimentConfig(
        model="smoke", device="auto", four_bit=False,
        capabilities=["mmlu_anatomy", "arc_challenge", "gsm8k"],
        corpora=["alpaca", "mbpp"],
        eval_items=24, corpus_train=32, corpus_test=16, rehearsal_held=2,
        seeds=[0], levels=["L1", "L3"],
        conditions=["naive", "drift_cap", "gpm_cap", "olora_cap", "ewc_mid", "gpm_e95"],
        batch_size=2, max_len=64, step_scale=0.05,
        probe_layers=[2, 4], bootstrap_n=200, permutation_n=200,
        out_dir=out_dir, tag="smoke",
    )


def load_config(path: Optional[str] = None, **overrides) -> ExperimentConfig:
    """Load a config from YAML/JSON (or defaults), then apply keyword overrides."""
    data: Dict[str, Any] = {}
    if path:
        p = Path(path)
        text = p.read_text()
        if p.suffix in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
    if data.pop("preset", None) == "smoke":
        cfg = smoke_config()
        data = {**cfg.to_dict(), **data}
    known = set(ExperimentConfig().to_dict())
    unknown = set(data) - known
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}\nvalid keys: {sorted(known)}")
    cfg = ExperimentConfig(**data)
    return cfg.replace(**{k: v for k, v in overrides.items() if v is not None})
