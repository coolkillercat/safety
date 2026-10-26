"""Write `<table>.schema.json` next to every CSV in results/round2 (round2.schemas registry).

  PYTHONPATH=src python scripts/write_schemas.py
"""
from pathlib import Path

from round2.schemas import stamp

ROOT = Path(__file__).resolve().parents[1] / "results" / "round2"
n = 0
for p in sorted(ROOT.glob("*.csv")):
    stamp(p)
    n += 1
print(f"[schemas] stamped {n} tables in {ROOT}")
