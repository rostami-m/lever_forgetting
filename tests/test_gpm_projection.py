"""The invariant the GPM implementation rests on, checked without loading a model.

Run with:  python tests/test_gpm_projection.py     (or: pytest tests/)

The claim under test is the one the paper makes in Sec. 4: projecting the gradient alone does NOT
keep the applied update out of the protected subspace, because AdamW's element-wise m/sqrt(v)
rescaling does not commute with (I - M M^T); projecting the parameter after the step does. Getting
this wrong silently overstates how much a projection constrains, so it is worth a test that fails
loudly rather than a comment.
"""
import sys
from pathlib import Path

import torch
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ffcl.protect import project_A_grads, project_A_init, project_A_params  # noqa: E402


class _FakeLoRA:
    """Minimal stand-in exposing the one thing protect.py needs: named_parameters()."""

    def __init__(self, A):
        self.A = torch.nn.Parameter(A)

    def named_parameters(self):
        yield "lora_A", self.A

    def parameters(self):
        yield self.A


def _setup(r=4, d_in=16, k=3, seed=0):
    torch.manual_seed(seed)
    # M: orthonormal basis of the protected input subspace, d_in x k.
    M, _ = torch.linalg.qr(torch.randn(d_in, k))
    lora = _FakeLoRA(torch.randn(r, d_in))
    subs = {"lora_A": (M, k, d_in, 0.95)}
    return lora, subs, M


def _protected_energy_frac(A, M):
    return float(((A @ M) @ M.T).pow(2).sum() / A.pow(2).sum().clamp_min(1e-12))


def test_gradient_only_projection_leaks():
    """Baseline: with gradient projection alone, A drifts into the protected subspace."""
    lora, subs, M = _setup()
    project_A_init(lora, subs)
    opt = AdamW(lora.parameters(), lr=1e-2)
    for _ in range(20):
        # A loss whose gradient has a deliberate component inside span(M); the projection is
        # supposed to strip it. Adam's second-moment state is what smuggles it back in.
        loss = (lora.A @ M).pow(2).sum() + lora.A.pow(2).sum()
        loss.backward()
        project_A_grads(lora, subs)
        opt.step()
        opt.zero_grad(set_to_none=True)
    leak = _protected_energy_frac(lora.A.detach(), M)
    assert leak > 1e-4, (f"expected a measurable leak from gradient-only projection, got {leak:.2e}; "
                         f"if this ever fails the optimizer no longer rescales element-wise and the "
                         f"parameter projection may be redundant")
    return leak


def test_parameter_projection_holds_invariant():
    """The fix: A M = 0 to numerical precision after every step, so dW M = 0 exactly."""
    lora, subs, M = _setup()
    project_A_init(lora, subs)
    opt = AdamW(lora.parameters(), lr=1e-2)
    worst = 0.0
    for _ in range(20):
        loss = (lora.A @ M).pow(2).sum() + lora.A.pow(2).sum()
        loss.backward()
        project_A_grads(lora, subs)
        opt.step()
        project_A_params(lora, subs)
        opt.zero_grad(set_to_none=True)
        worst = max(worst, float((lora.A.detach() @ M).abs().max()))
    assert worst < 1e-5, f"A M should vanish after each step, max |A M| = {worst:.2e}"
    return worst


def test_projection_is_idempotent():
    """Re-projecting an already-projected A removes nothing, so the cost is one matmul, not fidelity."""
    lora, subs, _ = _setup()
    project_A_init(lora, subs)
    second = project_A_params(lora, subs)
    assert second < 1e-10, f"second projection removed {second:.2e} of the energy; expected ~0"
    return second


if __name__ == "__main__":
    leak = test_gradient_only_projection_leaks()
    worst = test_parameter_projection_holds_invariant()
    idem = test_projection_is_idempotent()
    print(f"gradient-only projection leaves {100 * leak:.2f}% of A's energy in the protected subspace")
    print(f"with parameter projection, max |A M| = {worst:.2e}")
    print(f"re-projection removes {idem:.2e} of the energy (idempotent)")
    print("OK")
