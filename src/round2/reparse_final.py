"""Add `parsed_final` and `unresolved` (terminal-answer rule, common2.parse_final) to every CoT record of
the anchor: baseline run (responses/<anchor>/cot), crossover runs incl. template B and its .cap768 backups (crossover),
branched run (branched), commitment run (commitment), and any later cell family under responses/<anchor>/. The strict field
(`parsed_strict`, or `a1` for branched/commitment rows) is left as it is. Writes results/round2/reparse_final.md.

  PYTHONPATH=src python -m round2.reparse_final
"""
from __future__ import annotations

from collections import Counter

from common import read_jsonl, write_jsonl
from round2.common2 import final_fields
from round2.config2 import ANCHOR, RES2

CHECK = {  # answer line, then reopened reasoning, then the cap — must come out unresolved (1-based rows)
    "xb-system-organiser-assertive-none.jsonl": [116, 128, 185, 232, 241],
    "xb-user-organiser-assertive-none.jsonl": [8, 130, 161, 185],
    "xb-userpre-organiser-assertive-none.jsonl": [33, 54, 158, 204, 259, 281, 294],
}


def main():
    base = RES2 / "responses" / ANCHOR
    lines, ok = ["# Terminal-answer re-parse", "",
                 "`parsed_final` = the strict answer only when `Answer: <letter>` is the last non-empty line; an answer "
                 "line followed by substantive text is `unresolved`. Strict letters are unchanged.", "",
                 "| family | records | strict answers | final answers | unresolved (strict answer lost) | of which length-capped |",
                 "| --- | ---: | ---: | ---: | ---: | ---: |"], True
    for fam in sorted(p.name for p in base.iterdir() if p.is_dir()):
        tot = Counter()
        for f in sorted((base / fam).glob("*.jsonl")):
            rows = read_jsonl(f)
            for r in rows:
                strict = r.get("parsed_strict", r.get("a1")) if "parsed_strict" in r or "a1" in r else None
                r.update(final_fields(r.get("text"), strict, r.get("finish_reason")))
                tot["n"] += 1
                tot["strict"] += strict is not None
                tot["final"] += r["parsed_final"] is not None
                lost = strict is not None and r["parsed_final"] is None
                tot["lost"] += lost
                tot["lost_cap"] += lost and (r.get("finish_reason") == "length" or bool(r.get("likely_truncated")))
            write_jsonl(f, rows)
            if f.name in CHECK:
                got = [rows[i - 1]["unresolved"] and rows[i - 1]["parsed_final"] is None for i in CHECK[f.name]]
                ok &= all(got)
                print(f"[reparse_final] {f.name}: V3 rows unresolved {sum(got)}/{len(got)}")
        lines.append(f"| {fam} | {tot['n']} | {tot['strict']} | {tot['final']} | {tot['lost']} | {tot['lost_cap']} |")
    lines += ["", f"V3's 16 template-B rows unresolved: {'yes' if ok else 'NO'}."]
    (RES2 / "reparse_final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
