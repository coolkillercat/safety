#!/usr/bin/env bash
# Rebuild every result table from the stored model outputs (terminal-answer rule first, then the strict parser),
# stamp and check the table schemas, and redraw the paper's figures. CPU only; run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONUTF8=1 PYTHONWARNINGS=ignore
mkdir -p results/round2/fig
for parser in "--final" ""; do
  for experiment in crossover branched commitment attribution; do
    echo "== $experiment ${parser:-(strict)}"
    python -m round2.$experiment analyze $parser > /dev/null
  done
done
python scripts/write_schemas.py
python scripts/check_schema.py
python scripts/make_figures.py
