"""Descriptive length ratios of the replay experiment prior-turn packages relative to the model's own baseline answer.

Reads only existing records (no model calls):
  results/round2/responses/qwen3.5-27b/cot/baseline.jsonl          the baseline chain of thought per (item, order)
  results/round2/responses/qwen3.5-27b/attribution/a_filler_control.jsonl   the filler assistant turn (prompt[2])
  data/round2/own_wrong_texts.jsonl                                  the Gemma-written wrong worked solutions
Writes results/round2/package_length_ratios.md (markdown only, so the CSV schema linter is untouched).

Run from the repo root:  PYTHONUTF8=1 python scripts/package_length_ratios.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "round2"
BASE = RES / "responses" / "qwen3.5-27b" / "cot" / "baseline.jsonl"
FILLER = RES / "responses" / "qwen3.5-27b" / "attribution" / "a_filler_control.jsonl"
WRONG = ROOT / "data" / "round2" / "own_wrong_texts.jsonl"
OUT = RES / "package_length_ratios.md"


def rows(p: Path):
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def quartiles(xs):
    xs = sorted(xs)
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return q[0], statistics.median(xs), q[2], min(xs), max(xs)


def main():
    base = {(r["item_id"], str(r["order"])): r["text"] for r in rows(BASE)}
    out = ["# Length of the replay-experiment prior-turn packages relative to the baseline answer (descriptive)", "",
           "Computed from the stored records only (`scripts/package_length_ratios.py`); no generation. Ratio = package text / "
           "the same (item, order)'s baseline chain of thought. Quartiles are inclusive sample quartiles.", "",
           "| package | n | unit | Q1 | median | Q3 | min | max |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]

    def report(name, pairs):
        for unit, fn in (("characters", len), ("words", lambda s: len(s.split()))):
            rs = [fn(t) / fn(b) for t, b in pairs if fn(b) > 0]
            q1, med, q3, lo, hi = quartiles(rs)
            out.append(f"| {name} | {len(rs)} | {unit} | {q1:.2f} | {med:.2f} | {q3:.2f} | {lo:.2f} | {hi:.2f} |")

    filler = []
    n_noprompt = [0]
    for r in rows(FILLER):
        msgs = r.get("prompt")
        if isinstance(msgs, str):
            import ast
            msgs = ast.literal_eval(msgs)
        if not msgs:
            n_noprompt[0] += 1
            continue
        turn = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        b = base.get((r["item_id"], str(r["order"])))
        if turn and b:
            filler.append((turn, b))
    report("filler (deliberation sentences, character-matched)", filler)
    out.append("")
    out.append(f"filler: {len(filler)} records with a stored prompt; {n_noprompt[0]} records without a stored prompt were skipped.")
    out.append("")
    out.extend(["| package | n | unit | Q1 | median | Q3 | min | max |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"])

    wrong = []
    n_ok = n_all = 0
    for r in rows(WRONG):
        b = base.get((r["item_id"], str(r["order"])))
        n_all += 1
        n_ok += bool(r.get("ok")) and str(r.get("ok")) != "False"
        if b:
            wrong.append((r["text"], b))
    report("own-wrong (Gemma-3-27B worked solution for Y)", wrong)
    out.append("")
    out.append(f"own-wrong texts: {n_all} written; {n_ok} end on `Answer: Y` (`ok`), {n_all - n_ok} do not and are kept, flagged.")
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
