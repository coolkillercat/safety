"""Linter for the result tables: every CSV in a results directory has a sidecar schema
with population / unit / weights / missing / support, written no earlier than the table, whose column list matches
the table's header — AND the tables satisfy the semantic invariants a schema sidecar cannot express:

  monotone-denominators   scheduled ≥ eligible ≥ parsed in every row (and the longer chains of the commitment and
                          attribution cell tables)
  F+U=final-parsed        commitment: n_initially_correct + n_initially_wrong == n_final_parsed; attribution: the two
                          baseline groups' n_group sum to n_baseline_parsed
  support<=cell           every contrast row's stated support (pairs) ≤ the smaller involved cell's parsed count
                          (attribution_contrasts / attribution_package_diffs against the matching attribution_qwen table)
  one-order-cells         crossover_cells: template-B / B2 cells (order 0 only) have scheduled = items, i.e. one third of
                          the three-order templates' scheduled count in the same stratum

Exit code 1 on any failure; each message names the table and the rule.

  PYTHONPATH=src python scripts/check_schema.py [DIR]      (default: results/round2)
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from round2.schemas import REQUIRED  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "round2"


def _mono(bad, name, df, chain, rule="monotone-denominators"):
    cols = [c for c in chain if c in df.columns]
    for a, b in zip(cols, cols[1:]):
        viol = df[df[a] < df[b]]
        if len(viol):
            r = viol.iloc[0]
            bad.append(f"{name}: {rule}: {a} ({r[a]}) < {b} ({r[b]}) in {len(viol)} row(s), first: "
                       f"{', '.join(f'{k}={r[k]}' for k in ('cell', 'stratum', 'prior', 'advice', 'baseline') if k in df.columns)}")


def semantic(p: Path, df: pd.DataFrame, bad: list):
    name = p.name
    # 1 — monotone denominators
    _mono(bad, name, df, ["scheduled", "eligible", "parsed"])
    _mono(bad, name, df, ["scheduled", "n_evidence_eligible", "n_baseline_parsed", "n_final_parsed"])
    _mono(bad, name, df, ["scheduled", "n_baseline_parsed", "n_group", "n_rows_present", "n_eligible", "n_final_parsed"])
    # 2 — F + U = final-parsed (parallel groups)
    if {"n_initially_correct", "n_initially_wrong", "n_final_parsed"} <= set(df.columns):
        viol = df[df.n_initially_correct + df.n_initially_wrong != df.n_final_parsed]
        if len(viol):
            r = viol.iloc[0]
            bad.append(f"{name}: F+U=final-parsed: n_initially_correct + n_initially_wrong != n_final_parsed in "
                       f"{len(viol)} row(s), first: {r.get('stratum')}/{r.get('prior')}/{r.get('advice')} "
                       f"({r.n_initially_correct} + {r.n_initially_wrong} != {r.n_final_parsed})")
    if name.startswith("attribution_qwen") and {"n_group", "n_baseline_parsed", "baseline"} <= set(df.columns):
        g = df.groupby(["stratum", "prior", "advice"])
        for key, grp in g:
            if grp.baseline.nunique() == 2 and grp.drop_duplicates("baseline").n_group.sum() != grp.n_baseline_parsed.iloc[0]:
                bad.append(f"{name}: F+U=final-parsed: the two baseline groups' n_group do not sum to n_baseline_parsed "
                           f"for {'/'.join(key)}")
                break
    # 3 — contrast support ≤ the smaller cell
    if name.startswith(("attribution_contrasts", "attribution_package_diffs")) and "pairs" in df.columns:
        suf = re.sub(r"^attribution_(contrasts|package_diffs)", "", p.stem)
        cells_p = p.with_name(f"attribution_qwen3.5-27b{suf}.csv")
        if not cells_p.exists():
            bad.append(f"{name}: support<=cell: companion cell table {cells_p.name} missing — support cannot be checked")
        else:
            cells = pd.read_csv(cells_p)
            cells = cells[cells.baseline == "correct"].set_index(["stratum", "prior", "advice"])["n_final_parsed"]
            cells_w = pd.read_csv(cells_p)
            cells_w = cells_w[cells_w.baseline == "wrong"].set_index(["stratum", "prior", "advice"])["n_final_parsed"]
            for _, r in df.iterrows():
                m = re.match(r"^(?:F|advice effect|error) (\S+) − (\S+)(?: \(initially wrong\))?$", str(r.contrast))
                if not m or not r.pairs:
                    continue
                A, B = m.groups()
                arms = [r.advice, "control"] if str(r.contrast).startswith("advice effect") else [r.advice]
                src = cells_w if str(r.contrast).endswith("(initially wrong)") else cells
                counts = [src.get((r.stratum, pk, arm), None) for pk in (A, B) for arm in arms]
                counts = [c for c in counts if c is not None]
                if counts and r.pairs > min(counts):
                    bad.append(f"{name}: support<=cell: '{r.contrast}' ({r.stratum}, {r.advice}) states support {int(r.pairs)} "
                               f"> the smaller cell's parsed count {int(min(counts))}")
                    break
    # 4 — one-order cells
    if name.startswith("crossover_cells") and {"template", "scheduled", "stratum"} <= set(df.columns):
        for s, grp in df.groupby("stratum"):
            three = grp[grp.template.isin(["A", "C"])].scheduled
            one = grp[grp.template.isin(["B", "B2"])]
            if three.empty or one.empty:
                continue
            if three.nunique() != 1:
                bad.append(f"{name}: one-order-cells: three-order templates disagree on scheduled in {s}: {sorted(three.unique())}")
                continue
            want = int(three.iloc[0]) // 3
            viol = one[one.scheduled != want]
            if len(viol):
                r = viol.iloc[0]
                bad.append(f"{name}: one-order-cells: {len(viol)} template-{r.template} row(s) have scheduled "
                           f"{int(r.scheduled)} != items {want} (one order only), first: {r.cell}")


def main() -> int:
    bad = []
    tables = sorted(ROOT.glob("*.csv"))
    for p in tables:
        sc = p.with_name(p.name + ".schema.json")
        if not sc.exists():
            bad.append(f"{p.name}: no schema")
            continue
        s = json.loads(sc.read_text(encoding="utf-8"))
        miss = [k for k in REQUIRED if not s.get(k)]
        if miss:
            bad.append(f"{p.name}: schema lacks {miss}")
        if sc.stat().st_mtime < p.stat().st_mtime:
            bad.append(f"{p.name}: schema older than the table")
        df = pd.read_csv(p)
        if s.get("columns") != list(df.columns):
            bad.append(f"{p.name}: schema columns differ from the header")
        try:
            semantic(p, df, bad)
        except Exception as e:  # a malformed table is a failure, not a crash
            bad.append(f"{p.name}: semantic check raised {type(e).__name__}: {e}")
    print(f"[check_schema] {ROOT}: {len(tables)} tables, {len(bad)} problems")
    for b in bad:
        print("  ", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
