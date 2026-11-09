# Running

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run a short end-to-end check:

```bash
python -m ffcl smoke --out-dir results-smoke
```

Run the full experiment suite:

```bash
scripts/run_shard.sh 0 1
```

Use multiple shards on available machines:

```bash
runner/launch.sh --backend local --shards 4 --jobs 2
```

Generate tables and figures from run outputs:

```bash
scripts/collect.sh
```
