"""Scoring: two capability metrics, held-out acquisition, forward-pass predictors, linear probes.

Two scorings for every multiple-choice item:
  generation   -- decode a few tokens, look for the option letter (format-sensitive)
  log-prob     -- argmax over the option-letter token logits (format-robust)
Their gap isolates format disruption from knowledge loss, and `frac_nonletter` quantifies it.
GSM8K is the exception: free-form numeric, so generation-only and excluded from log-prob and
non-letter statistics.

Acquisition is ALWAYS measured on the held-out split of the fine-tuning corpus. (An earlier version
measured it on the training strings themselves -- with five-sentence corpora, on the exact sentences
trained on -- so it reported memorisation.)
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


def _eval_items(items: List[dict], held: int) -> List[dict]:
    """Drop the rehearsal tail. Applied in EVERY condition so no condition is scored on data any
    other condition rehearsed -- which is why base capability differs slightly between tables."""
    return items[:-held] if held and len(items) > held else items


def _option_regex(options: List[str]) -> str:
    return f"[{''.join(re.escape(o) for o in options)}]"


def eval_bench(model, tok, items: List[dict], device: str, scoring: str = "logprob",
               held: int = 0, max_len: int = 1536) -> float:
    """Accuracy on one capability benchmark. scoring in {'logprob','gen'}; numeric items are always
    generation-scored."""
    import torch
    items = _eval_items(items, held)
    model.eval(); c = 0
    with torch.no_grad():
        for it in items:
            enc = tok(it["prompt"], return_tensors="pt", truncation=True,
                      max_length=max_len).to(device)
            if it["kind"] == "mc" and scoring == "logprob":
                lg = model(**enc).logits[0, -1, :].float()
                opts = it["options"]
                lids = {o: tok.encode(" " + o, add_special_tokens=False)[0] for o in opts}
                pred = opts[int(torch.tensor([lg[lids[o]] for o in opts]).argmax())]
                c += int(pred == it["answer"])
            elif it["kind"] == "mc":
                out = model.generate(**enc, max_new_tokens=4, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
                txt = tok.decode(out[0, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip().upper()
                m = re.search(_option_regex(it["options"]), txt)
                c += int(m is not None and m.group() == it["answer"])
            else:
                out = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
                txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                m = re.search(r"-?\d+", txt.replace(",", ""))
                c += int(m is not None and m.group() == it["answer"])
    return c / max(1, len(items))


def eval_both(model, tok, items: List[dict], device: str, held: int = 0,
              max_len: int = 1536) -> Tuple[float, float, float]:
    """(gen_acc, logprob_acc, frac_nonletter) in a single pass.

    frac_nonletter is the share of multiple-choice items where the model emitted no valid option
    token at all. It is the quantity that makes gen-scored accuracy fall BELOW chance, and it should
    be reported by anyone quoting sub-chance accuracies.
    """
    import torch
    items = _eval_items(items, held)
    model.eval()
    g = l = nonletter = n_mc = 0
    with torch.no_grad():
        for it in items:
            enc = tok(it["prompt"], return_tensors="pt", truncation=True,
                      max_length=max_len).to(device)
            if it["kind"] == "mc":
                n_mc += 1
                opts = it["options"]
                lg = model(**enc).logits[0, -1, :].float()
                lids = {o: tok.encode(" " + o, add_special_tokens=False)[0] for o in opts}
                l += int(opts[int(torch.tensor([lg[lids[o]] for o in opts]).argmax())]
                         == it["answer"])
                out = model.generate(**enc, max_new_tokens=4, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
                txt = tok.decode(out[0, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip().upper()
                m = re.search(_option_regex(opts), txt)
                if m is None:
                    nonletter += 1
                else:
                    g += int(m.group() == it["answer"])
            else:
                out = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
                txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                mm = re.search(r"-?\d+", txt.replace(",", ""))
                ok = int(mm is not None and mm.group() == it["answer"])
                g += ok; l += ok
    n = max(1, len(items))
    return g / n, l / n, (nonletter / n_mc if n_mc else 0.0)


def acquisition_heldout(model, tok, test_texts: List[str], device: str,
                        max_len: int = 128) -> float:
    """Mean next-token accuracy on the HELD-OUT split of the fine-tuning corpus."""
    import torch
    model.eval()
    correct = tot = 0
    with torch.no_grad():
        for t in test_texts:
            ids = tok.encode(t, add_special_tokens=True)[:max_len]
            if len(ids) < 2:
                continue
            inp = torch.tensor([ids[:-1]]).to(device)
            tgt = torch.tensor(ids[1:]).to(device)
            pred = model(input_ids=inp).logits[0].argmax(-1)
            correct += int((pred == tgt).sum()); tot += len(tgt)
    return correct / max(1, tot)


def eval_task_labelscore(model, tok, task: dict, device: str, max_len: int = 512) -> float:
    """Classification accuracy by first-label-token log-prob (the CL-stream experiment)."""
    import torch
    model.eval()
    labels = task["labels"]
    lids = [tok.encode(" " + l, add_special_tokens=False)[0] for l in labels]
    c = 0
    with torch.no_grad():
        for prompt, gold in task["eval"]:
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            lg = model(**enc).logits[0, -1, :].float()
            c += int(int(torch.tensor([lg[i] for i in lids]).argmax()) == gold)
    return c / max(1, len(task["eval"]))


# --------------------------------------------------------------------------- forward-pass predictors
def hidden_reps(model, tok, texts: List[str], layer: int, device: str, max_len: int = 96):
    """Last-token hidden states at one depth, for representation-overlap predictors."""
    import numpy as np
    import torch
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.eval()
    out = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            hs = base(**enc, output_hidden_states=True).hidden_states
            out.append(hs[layer][0, -1, :].float().cpu().numpy())
    return np.array(out)


def rep_overlap(A, B) -> float:
    """Mean pairwise cosine between two sets of hidden states.

    float64 and an explicit zero-norm guard: hidden states from small or heavily perturbed models
    can overflow float32 or be exactly zero, which would otherwise silently produce inf/nan.
    """
    import numpy as np
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    if A.size == 0 or B.size == 0:
        return float("nan")
    na = np.linalg.norm(A, axis=1, keepdims=True)
    nb = np.linalg.norm(B, axis=1, keepdims=True)
    An = np.divide(A, na, out=np.zeros_like(A), where=na > 0)
    Bn = np.divide(B, nb, out=np.zeros_like(B), where=nb > 0)
    v = An @ Bn.T
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def p_mag(model, tok, items: List[dict], corpus_lines: List[str], device: str, base_gen: float,
          held: int = 0, n_prefix: int = 4, max_prefix_chars: int = 2000) -> float:
    """In-context interference: the drop in capability accuracy when the task corpus is prepended.

    A FUNCTIONAL probe of conflict -- it asks what happens to the capability when the task occupies
    the context. In the paper this is the one predictor that works; every geometric overlap measure
    fails. Note it is a base-model property, so it takes ONE value per (capability, task) cell,
    which is why the analysis bootstraps over cells rather than rows.
    """
    prefix = "\n".join(corpus_lines[:n_prefix])[:max_prefix_chars] + "\n\n"
    shifted = [dict(it, prompt=prefix + it["prompt"]) for it in _eval_items(items, held)]
    return base_gen - eval_bench(model, tok, shifted, device, scoring="gen", held=0)


# --------------------------------------------------------------------------- linear probes
def hidden_states_multi(model, tok, items: List[dict], layers: List[int], device: str,
                        held: int = 0, max_len: int = 1536):
    """({layer: X[n,d]}, y[n]) last-token features at several depths, for probing."""
    import numpy as np
    import torch
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.eval()
    items = _eval_items(items, held)
    Xs = {L: [] for L in layers}
    y = []
    with torch.no_grad():
        for it in items:
            enc = tok(it["prompt"], return_tensors="pt", truncation=True,
                      max_length=max_len).to(device)
            hs = base(**enc, output_hidden_states=True).hidden_states
            for L in layers:
                idx = min(L, len(hs) - 1)
                Xs[L].append(hs[idx][0, -1, :].float().cpu().numpy())
            if it["kind"] == "mc":
                y.append(it["options"].index(it["answer"]))
            else:
                y.append(0)
    return {L: np.array(v) for L, v in Xs.items()}, np.array(y)


def probe_acc(X, y, cv: int = 5) -> float:
    """Cross-validated linear-probe accuracy. If the answer is decodable from the representation,
    the knowledge is present even when the model will not emit it.

    The fold count adapts to the rarest class, because stratified CV needs at least one sample per
    class per fold. Returns nan when even 2-fold is impossible (too few items or a single class),
    which is why very small `eval_items` yields no probe result.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    min_class = int(np.bincount(y).min()) if y.min() >= 0 else 0
    folds = min(cv, min_class)
    if folds < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    try:
        return float(cross_val_score(clf, X, y, cv=folds).mean())
    except Exception:
        return float("nan")


def best_depth_probe(model, tok, items: List[dict], layers: List[int], device: str,
                     held: int = 0) -> Tuple[float, Dict[int, float]]:
    """Probe every depth; return (best accuracy, per-depth accuracies).

    The paper's test of destruction-vs-access: if severe forgetting were merely loss of output
    access, the answer should remain decodable at SOME depth.
    """
    import numpy as np
    Xs, y = hidden_states_multi(model, tok, items, layers, device, held=held)
    per = {L: probe_acc(Xs[L], y) for L in layers}
    vals = [v for v in per.values() if not np.isnan(v)]
    return (max(vals) if vals else float("nan")), per
