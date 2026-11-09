"""Capability-protection machinery: activation subspaces (GPM / O-LoRA) and base-weight importance
(EWC).

This module is where the earlier version of this work went wrong, in two ways that each silently
produced a false conclusion. Both are documented here because the failure modes are easy to repeat.

  DEFECT 1 -- a rank-ONE "subspace". The original gradient-projection baseline removed the component
  of each LoRA gradient along a single normalised mean-rehearsal-gradient direction. With
  d_in ~ 1e3-1e4, projecting out one direction removes essentially no gradient energy, so the
  baseline could not possibly protect anything -- and its null result was read as evidence that
  constraining update *geometry* does not matter. `activation_subspaces` builds a genuine rank-k
  basis (k ~ 165 in the paper's runs) and `project_A_grads` REPORTS the fraction of gradient energy
  it removes, so a null result can be distinguished from a constraint that never bound.

  DEFECT 2 -- importance measured where the gradient vanishes. The original EWC baseline took the
  diagonal Fisher of the LoRA PARAMETERS at adapter initialisation. Since dL/dA = B^T (dL/dW) and
  LoRA initialises B = 0, that Fisher was identically zero for lora_A for the entire run: EWC
  constrained only B, and because B starts at zero it degenerated into a penalty on adapter
  magnitude -- i.e. a drift penalty. That made "what protects capability is small drift" partly true
  by construction. `base_weight_fisher` measures importance on the BASE weights, where it is a
  non-zero, constant property of the pretrained model.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# Bounded representation matrix for the subspace estimate. Standard GPM practice, and it caps the
# achievable rank at N_SAMP-1 by construction, which keeps memory predictable next to a large model.
N_SAMP = 256
TOK_PER_FWD = 8


def _adapted_linears(lora):
    """Yield (module_name, module) for every LoRA-adapted nn.Linear."""
    import torch.nn as nn
    for name, mod in lora.named_modules():
        if hasattr(mod, "lora_A") and isinstance(getattr(mod, "base_layer", None), nn.Linear):
            yield name, mod


def _lora_A_names(lora) -> Dict[str, str]:
    """module_name (without the peft prefix) -> lora_A parameter name."""
    out = {}
    for n, _ in lora.named_parameters():
        if "lora_A" in n:
            out[n.split(".lora_A")[0].replace("base_model.model.", "")] = n
    return out


def activation_subspaces(lora, tok, texts: List[str], device: str, energy: float = 0.95,
                         max_frac: float = 0.5, max_len: int = 128, quiet: bool = False
                         ) -> Dict[str, Tuple]:
    """Top-k right-singular basis of each adapted module's INPUT activations, on capability data.

    Returns {lora_A_param_name: (M [d_in,k] on device, k, d_in, captured_energy)}.

    Why the input subspace: the merged update is dW = (alpha/r) B A, so dW x = (alpha/r) B (A x).
    Suppressing the update's action on a protected input subspace therefore amounts to keeping A's
    rows out of that subspace -- which is exactly GPM specialised to LoRA.
    """
    import numpy as np
    import torch

    store: Dict[str, list] = {}
    counts: Dict[str, int] = {}
    hooks = []

    def mk(name):
        def hook(mod, inp, out):
            if counts.get(name, 0) >= N_SAMP:
                return
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            take = min(TOK_PER_FWD, x.shape[0], N_SAMP - counts.get(name, 0))
            idx = torch.randperm(x.shape[0], device=x.device)[:take]
            store.setdefault(name, []).append(x[idx].cpu())
            counts[name] = counts.get(name, 0) + take
        return hook

    for name, mod in _adapted_linears(lora):
        hooks.append(mod.register_forward_hook(mk(name)))
    lora.eval()
    with torch.no_grad():
        for t in texts:
            ids = tok.encode(t, add_special_tokens=True)[:max_len]
            if len(ids) < 2:
                continue
            lora(input_ids=torch.tensor([ids]).to(device))
    for h in hooks:
        h.remove()

    pname = _lora_A_names(lora)
    subs: Dict[str, Tuple] = {}
    for mod_name, chunks in store.items():
        key = mod_name.replace("base_model.model.", "")
        if key not in pname:
            continue
        X = torch.cat(chunks, 0)                       # [N, d_in], on CPU
        d_in = X.shape[1]
        try:
            _, S, Vh = torch.linalg.svd(X.to(device), full_matrices=False)
        except Exception:                              # fall back to CPU SVD
            _, S, Vh = torch.linalg.svd(X, full_matrices=False)
        # Spectrum arithmetic on CPU in float64: MPS has no float64 support at all, and the
        # cumulative-energy ratio is numerically delicate.
        ev = S.detach().cpu().double() ** 2
        cum = torch.cumsum(ev, 0) / ev.sum().clamp(min=1e-12)
        k = int((cum < energy).sum().item()) + 1
        k = max(1, min(k, Vh.shape[0], int(d_in * max_frac)))
        M = Vh[:k, :].T.contiguous().to(device)
        subs[pname[key]] = (M, k, d_in, float(cum[k - 1]))
        del X, S, Vh, ev
    store.clear()
    if device == "cuda":
        torch.cuda.empty_cache()

    if not quiet and subs:
        ks = [v[1] for v in subs.values()]; dins = [v[2] for v in subs.values()]
        caps = [v[3] for v in subs.values()]
        print(f"[protect] activation subspaces: {len(subs)} modules, k mean={np.mean(ks):.0f} "
              f"(min {min(ks)} max {max(ks)}), protected dim frac="
              f"{np.mean([a / b for a, b in zip(ks, dins)]):.3f}, "
              f"captured energy={np.mean(caps):.3f}", flush=True)
    return subs


def project_A_init(lora, subs: Dict[str, Tuple]) -> None:
    """Project lora_A's initialisation into the complement of the protected subspace.

    Necessary: gradient projection alone leaves A's kaiming init inside the protected subspace, and
    that component re-enters the merged update through B as soon as B leaves zero.
    """
    import torch
    with torch.no_grad():
        for n, p in lora.named_parameters():
            if n in subs:
                M = subs[n][0].float()
                A = p.detach().float()
                p.copy_((A - (A @ M) @ M.T).to(p.dtype))


def project_A_grads(lora, subs: Dict[str, Tuple]) -> float:
    """GPM step: G_A <- G_A (I - M M^T).

    Returns the FRACTION OF GRADIENT ENERGY REMOVED. Report this: a projection that removes ~0 has
    not constrained anything, and a null result from it says nothing about geometry.

    NOTE: this is necessary but NOT sufficient. See project_A_params -- an adaptive optimizer's
    element-wise rescaling does not commute with the projection, so the applied step can carry a
    component along M even when the gradient did not.
    """
    removed = kept = 0.0
    for n, p in lora.named_parameters():
        if p.grad is None or n not in subs:
            continue
        M = subs[n][0].float()
        G = p.grad.detach().float()
        Gp = G - (G @ M) @ M.T
        removed += float((G - Gp).pow(2).sum())
        kept += float(G.pow(2).sum())
        p.grad = Gp.to(p.grad.dtype)
    return removed / max(1e-12, kept)


def project_A_params(lora, subs: Dict[str, Tuple]) -> float:
    """GPM on the APPLIED UPDATE: after opt.step(), re-project A onto the complement of M.

    Projecting only the gradient is not sufficient. AdamW rescales element-wise by m/sqrt(v), and
    that rescaling does not commute with (I - M M^T), so a component along M re-enters the applied
    update even on a step whose gradient had none. Projecting the parameter itself restores the exact
    guarantee: A M = 0 holds after every step (the init is projected too, see project_A_init), so
    dW = (alpha/r) B A satisfies dW M = 0 exactly and the protected input directions are untouched.
    The projection is idempotent, so it costs one matmul per module and nothing in fidelity.

    Returns the fraction of the post-step parameter energy that lay in the protected subspace.
    """
    import torch
    removed = tot = 0.0
    with torch.no_grad():
        for n, p in lora.named_parameters():
            if n not in subs:
                continue
            M = subs[n][0].float()
            A = p.detach().float()
            Ap = A - (A @ M) @ M.T
            removed += float((A - Ap).pow(2).sum())
            tot += float(A.pow(2).sum())
            p.copy_(Ap.to(p.dtype))
    return removed / max(1e-12, tot)


def update_energy_in_subspace(prev: Dict[str, "object"], named: Dict[str, "object"],
                              subs: Dict[str, Tuple]) -> float:
    """Fraction of the update AdamW actually applied that lies along M.

    `prev` is a snapshot of the projected A parameters taken before opt.step(). Because the invariant
    A M = 0 holds at `prev`, the protected component of (A_new - A_prev) is exactly the protected
    component of A_new -- i.e. this measures precisely the leak that gradient-only projection would
    have left in the weights. Logged as `updenergy_leaked`.
    """
    import torch
    num = den = 0.0
    with torch.no_grad():
        for n, p0 in prev.items():
            d = named[n].detach().float() - p0.float()
            M = subs[n][0].float()
            num += float(((d @ M) @ M.T).pow(2).sum())
            den += float(d.pow(2).sum())
    return num / max(1e-12, den)


def olora_penalty(lora, subs: Dict[str, Tuple]):
    """O-LoRA-style soft subspace orthogonality: sum_m ||A_m M_m||_F^2.

    This is the O-LoRA form ||A_i A_j^T||^2 with the "previous task" being the pretrained
    capability input subspace -- the faithful analogue when adapting a base model rather than
    continuing a task sequence.
    """
    pen = 0.0
    for n, p in lora.named_parameters():
        if n in subs:
            pen = pen + (p.float() @ subs[n][0].float()).pow(2).sum()
    return pen


def base_weight_fisher(lora, tok, texts: List[str], device: str, n_batches: int = 24,
                       max_len: int = 128, quiet: bool = False) -> Dict[Tuple[str, str], "object"]:
    """Per-output-channel empirical Fisher of the BASE weights W on the capability loss.

    Returns {(lora_A_name, lora_B_name): F [d_out] on device}, normalised to unit mean per module.

    The exact row-sum Fisher is F_i = sum_j ( sum_t g_{t,i} x_{t,j} )^2, which needs the full
    d_out x d_in outer product per module. We use the token-diagonal approximation
        F_i ~= sum_t g_{t,i}^2 ||x_t||^2 ,
    obtainable from hooks in O(d_out + d_in) memory per module, which is what lets this run beside
    an 8B model on one GPU. It is a structured (per-output-channel, token-diagonal) approximation
    and the paper reports it as such. Crucially, unlike an adapter-space Fisher taken at B = 0, it
    is non-zero.
    """
    import torch

    xnorm: Dict[str, "object"] = {}
    fisher: Dict[str, "object"] = {}
    hooks = []

    def fwd(name):
        def h(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            xnorm[name] = x.pow(2).sum(dim=1)                        # [T]
        return h

    def bwd(name):
        def h(mod, gin, gout):
            g = gout[0].detach().reshape(-1, gout[0].shape[-1]).float()   # [T, d_out]
            xn = xnorm.get(name)
            if xn is None or xn.shape[0] != g.shape[0]:
                return
            contrib = (g.pow(2) * xn.unsqueeze(1)).sum(dim=0)         # [d_out]
            fisher[name] = fisher.get(name, torch.zeros_like(contrib)) + contrib
        return h

    for name, mod in _adapted_linears(lora):
        hooks.append(mod.base_layer.register_forward_hook(fwd(name)))
        hooks.append(mod.base_layer.register_full_backward_hook(bwd(name)))

    import numpy as np
    rng = np.random.RandomState(1234)
    enc = [tok.encode(t, add_special_tokens=True)[:max_len] for t in texts]
    enc = [e for e in enc if len(e) >= 2]
    lora.train()
    for _ in range(n_batches):
        if not enc:
            break
        ids = enc[int(rng.choice(len(enc)))]
        ii = torch.tensor([ids]).to(device)
        lora.zero_grad(set_to_none=True)
        lora(input_ids=ii, labels=ii).loss.backward()
    for h in hooks:
        h.remove()
    lora.zero_grad(set_to_none=True)
    for k in fisher:
        fisher[k] /= max(1, n_batches)
        fisher[k] /= fisher[k].mean().clamp(min=1e-12)                # comparable across modules
    lora.eval(); xnorm.clear()

    pairs = {}
    for n, _ in lora.named_parameters():
        if "lora_A" in n:
            pairs[n.split(".lora_A")[0].replace("base_model.model.", "")] = (
                n, n.replace("lora_A", "lora_B"))
    out = {}
    for k, f in fisher.items():
        key = k.replace("base_model.model.", "")
        if key in pairs:
            out[pairs[key]] = f
    if not quiet:
        print(f"[protect] base-weight Fisher: {len(out)} modules", flush=True)
    return out


def ewc_penalty(named: Dict[str, "object"], fisher: Dict[Tuple[str, str], "object"],
                scale: float):
    """Omega = sum_m sum_i F_{m,i} ||dW_{m,i}||^2 with dW = (alpha/r) B A.

    Materialising dW (d_out x d_in per module, hundreds of modules, inside the autograd graph) OOMs
    even on 46GB, so use the exact identity
        ||dW_{m,i}||^2 = (alpha/r)^2 * B_{m,i} (A_m A_m^T) B_{m,i}^T ,
    which only ever forms the r x r Gram matrix and a [d_out, r] product.
    """
    pen = 0.0
    c2 = scale ** 2
    for (na, nb), f in fisher.items():
        A = named[na].float(); B = named[nb].float()
        G = A @ A.T                                       # [r, r]
        rownorm = ((B @ G) * B).sum(dim=1)                 # [d_out] == ||B_i A||^2
        pen = pen + (f * rownorm).sum() * c2
    return pen
