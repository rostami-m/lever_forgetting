"""LoRA adapters, layer-band targeting, and the weight-drift metric.

Weight drift is the paper's control variable, and two details about it matter:

  1. PEFT applies dW = (alpha/r) B A. Omitting the alpha/r factor -- as an earlier version of this
     work did -- understates the true update norm by a constant factor. `drift_norm` includes it.

  2. D is a SUM of per-module norms, so it grows with the number of adapted modules: an
     attention-only adapter touches 4 modules per layer against 7 for an MLP-inclusive one, and
     models differ in depth. Raw D is therefore NOT comparable across adapter targets or model
     depths, and no single universal numerical threshold should be read off it. Always report
     `drift_per_param` alongside `drift`.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]


def target_modules(mlp: bool) -> List[str]:
    return ATTN + (MLP if mlp else [])


def make_lora(model, r: int, alpha: int, mlp: bool = True, targets: Optional[List[str]] = None):
    """Wrap `model` with a LoRA adapter. Explicit `targets` (module name suffixes or full names)
    override the attn/mlp selection -- used for the band-restricted layer-locus experiment."""
    from peft import get_peft_model, LoraConfig
    tm = targets if targets is not None else target_modules(mlp)
    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=tm,
                     lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def make_lora_level(model, level_cfg: dict):
    return make_lora(model, level_cfg["r"], level_cfg["alpha"], level_cfg["mlp"])


def reset_lora(model, seed: int) -> None:
    """Re-initialise adapter weights deterministically (kaiming A, zero B), so one wrapped model
    can be reused across cells without nesting adapters."""
    import random
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_A" in n:
            torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        elif "lora_B" in n:
            torch.nn.init.zeros_(p)


def lora_scale(level_cfg: dict) -> float:
    return level_cfg["alpha"] / level_cfg["r"]


def drift_norm(model, scale: float = 1.0) -> float:
    """D = sum_m ||(alpha/r) B_m A_m||_F over adapted modules. Pass scale=alpha/r."""
    import torch
    sd = dict(model.named_parameters())
    tot = 0.0
    for ka in [k for k in sd if "lora_A" in k]:
        kb = ka.replace("lora_A", "lora_B")
        if kb not in sd:
            continue
        A = sd[ka].detach().float(); B = sd[kb].detach().float()
        if A.ndim == 2 and B.ndim == 2:
            tot += float(torch.linalg.matrix_norm(scale * (B @ A)))
        else:
            tot += float(A.norm() * B.norm()) * scale
    return tot


def n_adapted_params(model) -> int:
    return sum(p.numel() for n, p in model.named_parameters()
               if "lora_A" in n or "lora_B" in n)


def full_drift(model, snapshot: Dict[str, "object"]) -> float:
    """D for full fine-tuning: sum_m ||theta_m - theta0_m||_F, streaming one tensor at a time so a
    3B reference copy need not sit on the GPU beside gradients and optimiser state."""
    import torch
    tot = 0.0
    with torch.no_grad():
        for n, p in model.named_parameters():
            ref = snapshot[n].to(p.device, non_blocking=True)
            tot += float((p.detach().float() - ref.float()).norm())
            del ref
    return tot


def snapshot_cpu(model) -> Dict[str, "object"]:
    return {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters()}


def restore(model, snapshot: Dict[str, "object"]) -> None:
    import torch
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(snapshot[n].to(p.device))


def band_layers(n_layers: int, band: str) -> List[int]:
    """Layer indices for a contiguous quartile band (b0..b3) or 'full'."""
    q = n_layers // 4
    spans = {"b0": range(0, q), "b1": range(q, 2 * q),
             "b2": range(2 * q, 3 * q), "b3": range(3 * q, n_layers)}
    if band == "full":
        return list(range(n_layers))
    return list(spans[band])


def band_mlp_targets(model, layers: List[int]) -> List[str]:
    """Explicit MLP module names restricted to the given layer indices."""
    targets = []
    for name, _ in model.named_modules():
        m = re.search(r"\.layers\.(\d+)\.", name)
        if m and any(name.endswith(mod) for mod in MLP) and int(m.group(1)) in layers:
            targets.append(name)
    return targets
