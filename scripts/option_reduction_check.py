"""Check the four-option reduction of MMLU-Pro items: could reducing an item to four options have shifted its key?

`load_mmlu_pro` filters empty / "N/A" option strings *before* indexing with the original `answer_index`, and `reduce_to_four`
samples and shuffles option strings. The check compares every item we used with its original TIGER-Lab/MMLU-Pro record:

  (1) the stored key letter must point at exactly the original key text (`options[answer_index]` of the original record);
  (2) count original records whose empty / "N/A" options sit *before* the key (the only way the pre-filter could shift it);
  (3) the original `answer` letter must agree with `answer_index`;
  (4) list stored options whose meaning depends on the removed options ("all / none of the above", "both... and...").

Reads data/items.jsonl and data/round2/items_*.jsonl (union by id), downloads the MMLU-Pro test split once via `datasets`
(HF cache), and writes results/round2/option_reduction_check.md. No model calls.

  PYTHONUTF8=1 python scripts/option_reduction_check.py
"""
from __future__ import annotations

import glob
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "round2" / "option_reduction_check.md"
LETTERS = "ABCDEFGHIJ"
DEGENERATE = re.compile(r"\b(all|none|neither|both|either)\b.*\b(above|of these|of the following|options?)\b|\b(a|b|c|d) and (a|b|c|d)\b", re.I)


def rows(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    files = [ROOT / "data" / "items.jsonl"] + [Path(p) for p in sorted(glob.glob(str(ROOT / "data" / "round2" / "items_*.jsonl")))]
    used = {}
    for p in files:
        for r in rows(p):
            if not r["id"].startswith("mmlupro-"):
                continue
            used.setdefault(r["id"], {"files": [], "rec": r})["files"].append(p.name)
    anchor = {r["id"] for r in rows(ROOT / "data" / "round2" / "items_qwen3.5-27b.jsonl")}

    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    orig = {f"mmlupro-{r['question_id']}": r for r in ds}

    problems, shifted_risk, degenerate, letter_mismatch = [], [], [], []
    n_checked = 0
    for iid, u in used.items():
        o = orig.get(iid)
        if o is None:
            problems.append((iid, "no original record"))
            continue
        n_checked += 1
        opts_full = o["options"]
        ai = o["answer_index"]
        key_text = opts_full[ai]
        if LETTERS[ai] != o["answer"]:
            letter_mismatch.append((iid, o["answer"], ai))
        before = [i for i, s in enumerate(opts_full[:ai]) if not s or not s.strip() or s.strip().upper() == "N/A"]
        if before:
            shifted_risk.append((iid, before))
        rec = u["rec"]
        stored_key_text = rec["options"][LETTERS.index(rec["key"])]
        if stored_key_text != key_text:
            problems.append((iid, f"stored key text {stored_key_text!r} != original {key_text!r}"))
        if any(opt not in opts_full for opt in rec["options"]):
            problems.append((iid, "a stored option is not among the original options"))
        for opt in rec["options"]:
            if DEGENERATE.search(opt):
                degenerate.append((iid, opt, iid in anchor))

    n_empty_any = sum(1 for o in orig.values() if any((not s or not s.strip() or s.strip().upper() == "N/A") for s in o["options"]))
    n_empty_before = sum(1 for o in orig.values() if any((not s or not s.strip() or s.strip().upper() == "N/A") for s in o["options"][:o["answer_index"]]))

    out = ["# Option-reduction check against the original MMLU-Pro records", "",
           f"Items used in any `items*.jsonl`: {len(used)} (anchor Qwen3.5-27B: {len(anchor)}); matched to an original record: {n_checked}.",
           f"Original test split: {len(orig)} records; records with any empty / N/A option string: {n_empty_any}; with such a string "
           f"*before* the key (the only configuration under which the pre-filter shifts `answer_index`): {n_empty_before}.", "",
           f"- (1) stored key letter points at the original key text: **{n_checked - len([p for p in problems if 'key text' in p[1]])} of {n_checked}** "
           f"({len([p for p in problems if 'key text' in p[1]])} mismatches).",
           f"- (2) used items with an empty / N/A option before the key: **{len(shifted_risk)}**.",
           f"- (3) original `answer` letter disagrees with `answer_index`: {len(letter_mismatch)} items.",
           f"- (4) stored options whose meaning depends on the removed options (regex: all/none/both/either ... above/of these; 'A and B'): "
           f"{len(degenerate)} option strings in {len({d[0] for d in degenerate})} items ({len({d[0] for d in degenerate if d[2]})} anchor items).", ""]
    if problems:
        out += ["## Mismatches", ""] + [f"- {i}: {m}" for i, m in problems] + [""]
    if shifted_risk:
        out += ["## Items with an empty / N/A option before the key", ""] + [f"- {i}: positions {b}" for i, b in shifted_risk] + [""]
    if letter_mismatch:
        out += ["## answer letter vs answer_index", ""] + [f"- {i}: answer {a}, index {ai}" for i, a, ai in letter_mismatch] + [""]
    if degenerate:
        out += ["## Stored options whose meaning may depend on the removed options", "",
                "| item | anchor | option text |", "| --- | --- | --- |"] + [f"| {i} | {'yes' if a else ''} | {t.replace('|', '/')} |" for i, t, a in degenerate] + [""]
    out += ["Files read: " + ", ".join(p.name for p in files) + "."]
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
