#!/usr/bin/env bash
# Rebuild every prompt of both experiments with the mock backend (no model, nothing written) and report how many
# scheduled outputs are still to do; with the stored outputs in place every count is zero.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONUTF8=1 PYTHONWARNINGS=ignore
for cells in b2a b2c; do python -m round2.crossover run --cells $cells --pair-with phaseB --backend mock --dry-run | grep "todo="; done
python -m round2.crossover run --cells p16 --max-tokens 2048 --backend mock --dry-run | grep "todo="
# template C was generated in four shards (<cell>.s<k>of4.jsonl), so it is checked shard by shard
for k in 0 1 2 3; do python -m round2.crossover run --cells p15 --max-tokens 2048 --shard $k/4 --backend mock --dry-run | grep "todo="; done
python -m round2.branched run --backend mock --dry-run | grep -i "todo" || true
python -m round2.commitment run --backend mock --dry-run | grep -i "todo" || true
python -m round2.attribution run --backend mock --dry-run | grep "todo="
