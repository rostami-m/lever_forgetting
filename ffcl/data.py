"""Datasets: held-out capability benchmarks, fine-tuning corpora, and the classification CL stream.

Two corpus families, kept strictly separate because they behave differently:

  REAL (default, the paper's primary evidence) -- Alpaca, Dolly, MBPP, GSM8K-train, PubMedQA, each
  split into disjoint train/test halves. New-task acquisition is scored ONLY on the held-out half.

  SYNTHETIC (opt-in, for reproducing one comparison) -- five hand-written sentences per domain,
  repeated to fill the schedule. These are degenerate: 300 steps at batch 4 draws 1200 samples from
  five strings. They exaggerate collapse severity, cause format breakdown that real data does not,
  and destroy the signal that relational predictors read. They exist here only so the
  synthetic-vs-real contrast in the paper can be reproduced.

A capability item is {prompt, answer, kind ('mc'|'num'), options}.
A corpus is {train: [str], test: [str]}.
"""
from __future__ import annotations

from typing import Dict, List

LETTERS = ["A", "B", "C", "D", "E"]


# --------------------------------------------------------------------------- capability benchmarks
def _mc_prompt(question: str, choices: List[str], letters=LETTERS) -> str:
    body = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(len(choices)))
    return f"{question}\n{body}\nAnswer with the letter only.\nAnswer:"


def load_benchmarks(capabilities: List[str], n_per: int = 60, quiet: bool = False
                    ) -> Dict[str, List[dict]]:
    """Load capability benchmarks by name; unknown/unavailable ones are skipped with a warning.

    Supported: mmlu_<subject>, arc_challenge, gsm8k, hellaswag, winogrande.
    """
    from datasets import load_dataset

    def log(m):
        if not quiet:
            print(f"[data] {m}", flush=True)

    B: Dict[str, List[dict]] = {}
    for cap in capabilities:
        try:
            if cap.startswith("mmlu_"):
                ds = load_dataset("cais/mmlu", cap[len("mmlu_"):], split="test")
                items = [dict(prompt=_mc_prompt(ex["question"], ex["choices"]),
                              answer=LETTERS[ex["answer"]], kind="mc",
                              options=LETTERS[:len(ex["choices"])])
                         for ex in ds.select(range(min(n_per, len(ds))))]
            elif cap == "arc_challenge":
                ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
                items = []
                for ex in ds.select(range(min(n_per, len(ds)))):
                    lb, tx = ex["choices"]["label"], ex["choices"]["text"]
                    items.append(dict(prompt=_mc_prompt(ex["question"], tx, lb),
                                      answer=ex["answerKey"], kind="mc", options=list(lb)))
            elif cap == "gsm8k":
                ds = load_dataset("openai/gsm8k", "main", split="test")
                items = []
                for ex in ds.select(range(min(n_per, len(ds)))):
                    ans = ex["answer"].split("####")[-1].strip().replace(",", "")
                    items.append(dict(
                        prompt="Solve. Give only the final integer answer.\n" + ex["question"]
                               + "\nAnswer:", answer=ans, kind="num", options=None))
            elif cap == "hellaswag":
                ds = load_dataset("Rowan/hellaswag", split="validation")
                items = [dict(prompt=_mc_prompt(ex["ctx"], ex["endings"]),
                              answer=LETTERS[int(ex["label"])], kind="mc", options=LETTERS[:4])
                         for ex in ds.select(range(min(n_per, len(ds))))]
            elif cap == "winogrande":
                ds = load_dataset("allenai/winogrande", "winogrande_debiased",
                                  split="validation", trust_remote_code=True)
                items = []
                for ex in ds.select(range(min(n_per, len(ds)))):
                    p = (ex["sentence"] + "\nA. " + ex["option1"] + "\nB. " + ex["option2"]
                         + "\nWhich option best fills the blank? Answer with the letter only."
                         + "\nAnswer:")
                    items.append(dict(prompt=p, answer=("A" if ex["answer"] == "1" else "B"),
                                      kind="mc", options=["A", "B"]))
            else:
                log(f"unknown capability '{cap}', skipping"); continue
            B[cap] = items
        except Exception as e:                                    # network / dataset availability
            log(f"WARN could not load {cap}: {type(e).__name__}: {e}")
    log(f"loaded {len(B)} capabilities: {list(B)}")
    return B


# --------------------------------------------------------------------------- real corpora
def load_real_corpora(names: List[str], n_train: int = 400, n_test: int = 100,
                      quiet: bool = False) -> Dict[str, dict]:
    """Real fine-tuning corpora with disjoint train/test splits."""
    from datasets import load_dataset

    def log(m):
        if not quiet:
            print(f"[data] {m}", flush=True)

    out: Dict[str, dict] = {}

    def add(name: str, texts: List[str]):
        texts = [t for t in (s.strip() for s in texts) if len(t) > 20]
        test = texts[n_train:n_train + n_test]
        need_test = max(1, min(8, n_test))
        # Require a non-empty held-out split. Without this guard a source with too few rows (MBPP's
        # `train` split has 374) yields an EMPTY test set, and held-out acquisition then silently
        # evaluates to 0.0 rather than erroring.
        if len(texts) >= n_train + need_test and len(test) >= need_test:
            out[name] = dict(train=texts[:n_train], test=test)
        else:
            log(f"WARN skipping {name}: {len(texts)} usable lines, need "
                f"{n_train}+{need_test} for a train split plus a non-empty held-out split")

    loaders = {
        "alpaca": lambda: [(e["instruction"] + "\n" + e["input"] + "\n" + e["output"])
                           for e in load_dataset("tatsu-lab/alpaca",
                                                 split=f"train[:{n_train + n_test}]")],
        "dolly": lambda: [(e["instruction"] + "\n" + e["context"] + "\n" + e["response"])
                          for e in load_dataset("databricks/databricks-dolly-15k",
                                                split=f"train[:{n_train + n_test}]")],
        # MBPP's `train` split has only 374 rows, so pull train+validation+test (~964).
        "mbpp": lambda: [(e["text"] + "\n" + e["code"])
                         for e in load_dataset("google-research-datasets/mbpp", "full",
                                               split="train+validation+test")],
        "gsm8k_train": lambda: [(e["question"] + "\n" + e["answer"])
                                for e in load_dataset("openai/gsm8k", "main", split="train")
                                .select(range(n_train + n_test))],
        "pubmedqa": lambda: [(e["question"] + "\n" + " ".join(e["context"]["contexts"])
                              + "\n" + e["long_answer"])
                             for e in load_dataset("qiaojin/PubMedQA", "pqa_labeled",
                                                   split="train")],
    }
    for name in names:
        if name not in loaders:
            log(f"WARN unknown real corpus '{name}'"); continue
        try:
            add(name, loaders[name]())
        except Exception as e:
            log(f"WARN could not load {name}: {type(e).__name__}: {e}")
    for k, v in out.items():
        log(f"corpus {k}: train={len(v['train'])} test={len(v['test'])}")
    return out


# --------------------------------------------------------------------------- synthetic corpora
_SYNTHETIC = {
    "code_py": ["def add(a,b):\n    return a+b", "import numpy as np\narr=np.zeros((3,3))",
                "for i in range(10):\n    print(i**2)",
                "class Node:\n    def __init__(self,v):\n        self.v=v",
                "result=[x for x in data if x>0]"],
    "legal": ["The party of the first part agrees to indemnify the second party.",
              "Pursuant to Section 12(b), the lessee shall remit payment in thirty days.",
              "This agreement is governed by the laws of Delaware.",
              "The covenant constitutes a material breach of contract.",
              "Whereas the plaintiff alleges damages exceeding the statutory minimum."],
    "medical": ["The patient presented with acute myocardial infarction and elevated troponin.",
                "Administer 500mg amoxicillin every eight hours for infection.",
                "Differential diagnosis includes pneumonia and pulmonary embolism.",
                "The MRI revealed a lesion in the left temporal lobe.",
                "Hypertension was managed with an ACE inhibitor and diuretic."],
    "poetry": ["The moon hung low, a silver coin upon the velvet night.",
               "Whispers of autumn drift through amber trees like memories.",
               "In gardens deep where shadows pool, the nightingale sings.",
               "Beneath the willow's weeping arc, the river dreams of sea.",
               "Dawn spills its gold across the hills in quiet reverie."],
    "math": ["The integral of x squared with respect to x equals x cubed over three plus C.",
             "By the Pythagorean theorem, a squared plus b squared equals c squared.",
             "The derivative of sine x is cosine x.",
             "A prime number has exactly two distinct positive divisors.",
             "The determinant of a two by two matrix is ad minus bc."],
    "recipe": ["Preheat the oven to 350 degrees and grease a nine inch pan.",
               "Whisk together the flour, baking soda, and a pinch of salt.",
               "Simmer the sauce over low heat until it thickens slightly.",
               "Fold the egg whites gently into the batter until just combined.",
               "Season the chicken and sear it skin-side down until golden."],
}


def load_synthetic_corpora(names: List[str], n_train: int = 400, n_test: int = 100,
                           quiet: bool = False) -> Dict[str, dict]:
    """The degenerate five-sentence domains, for reproducing the synthetic-vs-real comparison ONLY.

    There is no honest held-out split of five sentences, so `test` repeats the same lines. Any
    acquisition number measured on it is a memorisation score, which is exactly the defect the
    paper documents; the analysis therefore ignores acquisition for this family.
    """
    if not quiet:
        print("[data] WARNING: synthetic corpora are degenerate (5 unique lines per domain, "
              "repeated). They exaggerate collapse and format breakdown. Use for the "
              "synthetic-vs-real comparison only.", flush=True)
    out = {}
    for n in names:
        if n not in _SYNTHETIC:
            continue
        base = _SYNTHETIC[n]
        reps = max(1, n_train // len(base))
        out[n] = dict(train=base * reps, test=list(base))
    return out


def load_corpora(cfg, quiet: bool = False) -> Dict[str, dict]:
    """Dispatch on cfg.corpus_family."""
    if cfg.corpus_family == "synthetic":
        return load_synthetic_corpora(cfg.corpora, cfg.corpus_train, cfg.corpus_test, quiet)
    return load_real_corpora(cfg.corpora, cfg.corpus_train, cfg.corpus_test, quiet)


# --------------------------------------------------------------------------- classification stream
def load_cl_stream(n_train: int = 400, n_eval: int = 100, quiet: bool = False) -> dict:
    """AG News -> DBpedia -> Yelp -> TREC, a genuinely non-SFT task-incremental stream.

    Each task: {labels, train: [(prompt, label_str)], eval: [(prompt, gold_idx)]}.
    """
    from datasets import load_dataset

    def log(m):
        if not quiet:
            print(f"[data] {m}", flush=True)

    def mk(head, text, labels):
        return (f"{head}\nText: {text}\nAnswer with one label from "
                f"{{{', '.join(labels)}}}.\nLabel:")

    T: dict = {}
    try:
        labels = ["World", "Sports", "Business", "Technology"]
        ds = load_dataset("ag_news", split=f"train[:{n_train + n_eval}]")
        rows = [(mk("Classify the news topic.", ex["text"][:400], labels), ex["label"]) for ex in ds]
        T["agnews"] = dict(labels=labels, train=[(p, labels[y]) for p, y in rows[:n_train]],
                           eval=[(p, y) for p, y in rows[n_train:n_train + n_eval]])
    except Exception as e:
        log(f"WARN agnews: {e}")
    try:
        keep = {0: "Company", 2: "Artist", 3: "Athlete", 12: "Film"}
        labels = list(keep.values()); li = {l: i for i, l in enumerate(labels)}
        ds = load_dataset("dbpedia_14", split="train[:6000]")
        rows = []
        for ex in ds:
            if ex["label"] in keep:
                rows.append((mk("Classify the entity type.", ex["content"][:400], labels),
                             keep[ex["label"]]))
            if len(rows) >= n_train + n_eval:
                break
        T["dbpedia"] = dict(labels=labels, train=[(p, l) for p, l in rows[:n_train]],
                            eval=[(p, li[l]) for p, l in rows[n_train:n_train + n_eval]])
    except Exception as e:
        log(f"WARN dbpedia: {e}")
    try:
        labels = ["Negative", "Positive"]
        ds = load_dataset("yelp_polarity", split=f"train[:{n_train + n_eval}]")
        rows = [(mk("Classify the review sentiment.", ex["text"][:400], labels), ex["label"])
                for ex in ds]
        T["yelp"] = dict(labels=labels, train=[(p, labels[y]) for p, y in rows[:n_train]],
                         eval=[(p, y) for p, y in rows[n_train:n_train + n_eval]])
    except Exception as e:
        log(f"WARN yelp: {e}")
    try:
        labels = ["Abbreviation", "Entity", "Description", "Human", "Location", "Number"]
        ds = load_dataset("trec", split=f"train[:{n_train + n_eval}]")
        key = ("coarse_label" if "coarse_label" in ds.column_names
               else "label-coarse" if "label-coarse" in ds.column_names else "label")
        rows = [(mk("Classify the question type.", ex["text"][:300], labels), ex[key]) for ex in ds]
        T["trec"] = dict(labels=labels, train=[(p, labels[y]) for p, y in rows[:n_train]],
                         eval=[(p, y) for p, y in rows[n_train:n_train + n_eval]])
    except Exception as e:
        log(f"WARN trec: {e}")
    log(f"CL stream tasks: {list(T)}")
    return T


def rehearsal_strings(benches: Dict[str, List[dict]], held: int = 8) -> List[str]:
    """The reserved tail of each benchmark, as text, for protection methods and replay.

    These items are excluded from evaluation in EVERY condition (see eval.eval_bench), so no
    condition is scored on data it rehearsed.
    """
    return [it["prompt"] + " " + str(it["answer"])
            for items in benches.values() for it in items[-held:]]
