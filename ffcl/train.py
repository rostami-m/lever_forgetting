"""Training loops: plain fine-tuning, the mitigation conditions, label-masked classification, and
full fine-tuning.

Every loop seeds its own data order from the cell's seed. (An earlier version hardcoded
RandomState(0) for batch sampling, so "multiple seeds" varied only the adapter initialisation.)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from .config import CONDITIONS, USES_CAPABILITY_DATA
from .lora import drift_norm
from .protect import (activation_subspaces, base_weight_fisher, ewc_penalty, olora_penalty,
                      project_A_grads, project_A_init, project_A_params,
                      update_energy_in_subspace)


def _encode(tok, texts: List[str], max_len: int) -> List[List[int]]:
    out = [tok.encode(t, add_special_tokens=True)[:max_len] for t in texts]
    return [e for e in out if len(e) >= 2]


def _batch(enc, rng, batch, pad, device):
    import torch
    b = [enc[i] for i in rng.choice(len(enc), min(batch, len(enc)))]
    ml = max(len(x) for x in b)
    ii = torch.tensor([x + [pad] * (ml - len(x)) for x in b]).to(device)
    ll = torch.tensor([x + [-100] * (ml - len(x)) for x in b]).to(device)
    return ii, ll


def train_plain(lora, tok, texts, steps: int, lr: float, device: str, seed: int = 0,
                batch: int = 4, max_len: int = 128, scale: float = 1.0,
                tau: Optional[float] = None, monitor_every: int = 10) -> int:
    """LM fine-tune. If `tau` is set, early-stop when measured drift exceeds it (the drift cap).
    Returns the number of steps actually taken."""
    import numpy as np
    import torch
    from torch.optim import AdamW

    enc = _encode(tok, texts, max_len)
    if not enc:
        raise SystemExit("training corpus is empty after encoding")
    opt = AdamW([p for p in lora.parameters() if p.requires_grad], lr=lr)
    rng = np.random.RandomState(seed)
    pad = tok.pad_token_id
    lora.train()
    taken = 0
    for st in range(steps):
        ii, ll = _batch(enc, rng, batch, pad, device)
        lora(input_ids=ii, labels=ll).loss.backward()
        torch.nn.utils.clip_grad_norm_(lora.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        taken = st + 1
        if tau is not None and st % monitor_every == 0 and drift_norm(lora, scale) > tau:
            break
    lora.eval()
    return taken


def train_condition(lora, tok, texts, cond: str, device: str, *, benches: Dict[str, list],
                    rehearsal: List[str], base_steps: int, base_lr: float, scale: float,
                    seed: int = 0, batch: int = 4, max_len: int = 128,
                    quiet: bool = False) -> Tuple[int, dict]:
    """Unified mitigation trainer for any entry of `config.CONDITIONS`.

    Returns (steps_taken, diagnostics). The diagnostics record whether the condition's constraint
    actually BOUND -- fraction of gradient energy removed for a projection, mean penalty magnitude
    for a soft constraint, protected rank and captured energy for a subspace. Reporting these is
    what distinguishes "geometry does not help" from "the geometry constraint was never applied",
    which is precisely the error this reimplementation exists to correct.
    """
    import numpy as np
    import torch
    from torch.optim import AdamW

    if cond not in CONDITIONS:
        raise SystemExit(f"unknown condition '{cond}'. options: {sorted(CONDITIONS)}")
    spec = CONDITIONS[cond]
    kind = spec["kind"]
    lr = spec.get("lr", base_lr)
    steps = spec.get("steps", base_steps)
    tau = spec.get("tau")
    lam = spec.get("lam", 0.0)
    energy = spec.get("energy", 0.95)

    diag = dict(kind=kind, lam=lam if kind in ("ewc", "olora") else None,
                energy=energy if kind == "gpm" else None, tau=tau, lr=lr, steps_planned=steps,
                uses_capability_data=kind in USES_CAPABILITY_DATA,
                subspace_k_mean=None, protected_frac=None, captured_energy=None,
                gradenergy_removed=None, penalty_mean=None,
                updenergy_leaked=None, updenergy_leaked_max=None, update_projected=None)

    subs = fisher = replay_enc = None
    if kind in ("olora", "gpm"):
        subs = activation_subspaces(lora, tok, rehearsal, device, energy=energy,
                                    max_len=max_len, quiet=quiet)
        if subs:
            diag["subspace_k_mean"] = float(np.mean([v[1] for v in subs.values()]))
            diag["protected_frac"] = float(np.mean([v[1] / v[2] for v in subs.values()]))
            diag["captured_energy"] = float(np.mean([v[3] for v in subs.values()]))
        if kind == "gpm":
            project_A_init(lora, subs)
    elif kind == "ewc":
        fisher = base_weight_fisher(lora, tok, rehearsal, device, max_len=max_len, quiet=quiet)
    elif kind == "replay":
        replay_enc = _encode(tok, rehearsal, max_len)

    enc = _encode(tok, texts, max_len)
    opt = AdamW([p for p in lora.parameters() if p.requires_grad], lr=lr)
    rng = np.random.RandomState(seed)
    pad = tok.pad_token_id
    named = dict(lora.named_parameters())
    lora.train()
    taken = 0
    removed_hist, pen_hist, leak_hist = [], [], []

    for st in range(steps):
        if kind == "replay" and replay_enc:
            n_rep = max(1, int(batch * spec.get("replay_frac", 0.5)))
            seqs = ([enc[i] for i in rng.choice(len(enc), max(1, batch - n_rep))]
                    + [replay_enc[i] for i in rng.choice(len(replay_enc), n_rep)])
            for seq in seqs:
                ii = torch.tensor([seq]).to(device)
                (lora(input_ids=ii, labels=ii).loss / len(seqs)).backward()
        else:
            ii, ll = _batch(enc, rng, batch, pad, device)
            loss = lora(input_ids=ii, labels=ll).loss
            if kind == "olora" and subs:
                pen = olora_penalty(lora, subs)
                pen_hist.append(float(pen.detach()))
                loss = loss + (lam * pen / max(1, len(subs))).to(loss.dtype)
            elif kind == "ewc" and fisher:
                pen = ewc_penalty(named, fisher, scale)
                pen_hist.append(float(pen.detach()))
                loss = loss + (0.5 * lam * pen / max(1, len(fisher))).to(loss.dtype)
            loss.backward()

        if kind == "gpm" and subs:
            removed_hist.append(project_A_grads(lora, subs))
        torch.nn.utils.clip_grad_norm_(lora.parameters(), 1.0)
        # Snapshot the projected A's so the leak AdamW's m/sqrt(v) rescaling would otherwise leave
        # behind can be measured, then removed by re-projecting the parameters themselves.
        prev = ({n: named[n].detach().clone() for n in subs if n in named}
                if (kind == "gpm" and subs) else None)
        opt.step()
        if prev is not None:
            leak_hist.append(update_energy_in_subspace(prev, named, subs))
            project_A_params(lora, subs)
            prev = None
        opt.zero_grad(set_to_none=True)
        taken = st + 1
        if tau is not None and st % 10 == 0 and drift_norm(lora, scale) > tau:
            break

    if removed_hist:
        diag["gradenergy_removed"] = float(np.mean(removed_hist))
    if leak_hist:
        # Mean over steps of the applied update's protected-subspace share. Under the fixed
        # projection this energy is removed from the weights rather than left in them; its size is
        # the measure of how much a gradient-only projection would have leaked.
        diag["updenergy_leaked"] = float(np.mean(leak_hist))
        diag["updenergy_leaked_max"] = float(np.max(leak_hist))
        diag["update_projected"] = True
    if pen_hist:
        diag["penalty_mean"] = float(np.mean(pen_hist))
    lora.eval()
    return taken, diag


def train_label_masked(lora, tok, examples, steps: int, lr: float, device: str, seed: int = 0,
                       batch: int = 2, max_prompt: int = 140, max_len: int = 160) -> None:
    """Classification training that supervises only the label tokens (the CL-stream experiment)."""
    import numpy as np
    import torch
    from torch.optim import AdamW

    enc = []
    for prompt, label in examples:
        pids = tok.encode(prompt, add_special_tokens=True)[:max_prompt]
        lids = tok.encode(" " + label, add_special_tokens=False)
        ids = pids + lids
        lab = [-100] * len(pids) + lids
        if len(ids) >= 2:
            enc.append((ids[:max_len], lab[:max_len]))
    if not enc:
        return
    opt = AdamW([p for p in lora.parameters() if p.requires_grad], lr=lr)
    rng = np.random.RandomState(seed)
    pad = tok.pad_token_id
    lora.train()
    for _ in range(steps):
        b = [enc[i] for i in rng.choice(len(enc), min(batch, len(enc)))]
        ml = max(len(x[0]) for x in b)
        ii = torch.tensor([x[0] + [pad] * (ml - len(x[0])) for x in b]).to(device)
        ll = torch.tensor([x[1] + [-100] * (ml - len(x[1])) for x in b]).to(device)
        lora(input_ids=ii, labels=ll).loss.backward()
        torch.nn.utils.clip_grad_norm_(lora.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    lora.eval()


def train_full(model, tok, texts, steps: int, lr: float, device: str, seed: int = 0,
               batch: int = 2, max_len: int = 128) -> None:
    """Full fine-tuning with AdamW, matching the LoRA optimiser for an apples-to-apples ladder.
    Gradient checkpointing keeps a 3B model + gradients + moments inside 48GB."""
    import numpy as np
    import torch
    from torch.optim import AdamW

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(True)
    enc = _encode(tok, texts, max_len)
    opt = AdamW(model.parameters(), lr=lr)
    rng = np.random.RandomState(seed)
    pad = tok.pad_token_id
    model.train()
    for _ in range(steps):
        ii, ll = _batch(enc, rng, batch, pad, device)
        model(input_ids=ii, labels=ll).loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    model.eval()
    model.config.use_cache = True
    del opt
    if device == "cuda":
        import torch as _t; _t.cuda.empty_cache()
