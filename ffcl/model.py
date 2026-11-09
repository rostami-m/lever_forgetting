"""Model + tokenizer loading, device- and dtype-agnostic.

Works on CUDA, Apple MPS and CPU so the smoke test runs anywhere. Heavy imports are inside the
functions so `import ffcl` works without torch installed.
"""
from __future__ import annotations

from typing import Any, Tuple


def pick_device(requested: str = "auto") -> str:
    import torch
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, requested: str = "auto"):
    """bf16 on CUDA (what the paper used); float32 elsewhere, since bf16 matmul support on
    MPS/CPU is uneven across torch versions and this code path is only for smoke tests."""
    import torch
    if requested and requested != "auto":
        return dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[requested]
    return torch.bfloat16 if device == "cuda" else torch.float32


def load_model(model_id: str, four_bit: bool = False, device: str = "auto",
               dtype: str = "auto") -> Tuple[Any, Any, str]:
    """Return (tokenizer, model, device). `four_bit` loads the base in nf4 (CUDA only)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dev = pick_device(device)
    torch_dtype = pick_dtype(dev, dtype)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if four_bit:
        if dev != "cuda":
            raise SystemExit("four_bit requires CUDA; set four_bit: false")
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, low_cpu_mem_usage=True)
        model = prepare_model_for_kbit_training(model)
    else:
        # `dtype=` is the modern kwarg; older transformers want `torch_dtype=`.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch_dtype, low_cpu_mem_usage=True)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
        model = model.to(dev)

    model.eval()
    return tok, model, dev


def num_layers(model) -> int:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return int(base.config.num_hidden_layers)
