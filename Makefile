# Reproduction entry points. `make help` lists them.
#
# Every target here is a thin alias for something you can also run by hand; the
# Makefile exists so there is one obvious command per question ("does this work?",
# "do the numbers match the paper?", "run the whole thing").

SHELL := /bin/bash
.DEFAULT_GOAL := help

RESULTS_DIR ?= ./results
SHARDS      ?= 1
BACKEND     ?= local
export RESULTS_DIR

.PHONY: help install check smoke run collect figures clean distclean

help: ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies into the current environment
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

check: ## Verify the environment and that RESULTS_DIR is writable (do this first)
	python scripts/check_env.py

smoke: ## Minutes, not hours: tiny grid end-to-end, to prove the pipeline runs
	scripts/smoke_test.sh

run: ## Reproduce the paper. SHARDS=N BACKEND={local,docker,slurm,ssh}
	runner/launch.sh --backend $(BACKEND) --shards $(SHARDS)

collect: ## Merge shard outputs, then regenerate numbers and figures
	scripts/collect.sh

figures: ## Regenerate the paper's figures from existing results
	python scripts/make_figures.py

clean: ## Remove generated figures and derived tables, keep raw results
	rm -rf figures/generated derived
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

distclean: clean ## Also remove raw results. Destructive: re-running costs GPU-hours.
	@read -p "Delete everything under $(RESULTS_DIR)? [y/N] " a; [[ $$a == y ]] && rm -rf "$(RESULTS_DIR)" || echo "kept"
