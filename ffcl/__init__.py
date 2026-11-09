"""ffcl -- reproduction package for the paper
"Cap the Drift, Then Choose Its Direction: Two Composable Levers on Catastrophic Forgetting".

A small shared library plus config-driven experiment drivers and an analysis layer that regenerates
every table and figure. Nothing assumes any particular infrastructure.

    ffcl.config       declarative config: intensity ladder, conditions, model registry, presets
    ffcl.model        load a causal LM + tokenizer (CUDA / Apple MPS / CPU, bf16 or 4-bit)
    ffcl.data         capability benchmarks, real and synthetic FT corpora, classification stream
    ffcl.lora         adapters, layer bands, and the weight-drift metric (with the alpha/r factor)
    ffcl.protect      activation subspaces (GPM / O-LoRA) and base-weight importance (EWC)
    ffcl.train        training loops for every mitigation condition, with binding diagnostics
    ffcl.eval         both scorings, held-out acquisition, forward-pass predictors, linear probes
    ffcl.metrics      variance decomposition, additive null, cluster bootstrap, CL metrics
    ffcl.experiments  one driver per result family; each writes a resumable, shardable JSONL
    ffcl.analyze      logs -> every number in the paper (+ LaTeX)
    ffcl.figures      logs -> PDFs
    ffcl.runner       Backend abstraction (local / any-shell-command) and shard merging
    ffcl.cli          `python -m ffcl {doctor,smoke,list,plan,run,all,analyze,figures}`

Heavy dependencies are imported lazily, so `import ffcl` and most of the CLI work without a GPU.

Start with:  python -m ffcl doctor  &&  python -m ffcl smoke
"""
__version__ = "2.0.0"

from .config import ExperimentConfig, load_config, smoke_config  # noqa: F401

__all__ = ["ExperimentConfig", "load_config", "smoke_config", "__version__"]
