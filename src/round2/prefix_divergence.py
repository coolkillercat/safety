"""How the selectively regenerated template-B rows relate to the rows first generated at 768 tokens.

For every template-B cell with a .cap768.jsonl backup, compare each row's current text with its first-run text:
`changed` (text differs), `prefix_divergent` (the new text is not an exact extension of the old one) and
`divergence_index` (first differing character, None when it is an extension). Written into the current rows;
summary in results/round2/prefix_divergence.md.

  PYTHONPATH=src python -m round2.prefix_divergence
"""
from __future__ import annotations

from common import read_jsonl, write_jsonl
from round2.config2 import ANCHOR, RES2

OUT = RES2 / "responses" / ANCHOR / "crossover"


def first_diff(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    tot = dict(rows=0, changed=0, divergent=0, recapped=0)
    lines = ["# Template B: regenerated vs first-run texts", "",
             "| cell | rows | changed | not a prefix extension | median first-difference index (divergent) |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for f in sorted(OUT.glob("xb-*.jsonl")):
        if f.name.endswith(".cap768.jsonl"):
            continue
        b = f.with_name(f.stem + ".cap768.jsonl")
        rows = read_jsonl(f)
        old = {(r["item_id"], r["order"]): r for r in read_jsonl(b)} if b.exists() else {}
        ch = dv = 0
        idx = []
        for r in rows:
            o = old.get((r["item_id"], r["order"]))
            if o is None:
                continue
            changed = r["text"] != o["text"]
            ext = r["text"].startswith(o["text"])
            r["changed_vs_cap768"] = changed
            r["prefix_divergent"] = changed and not ext
            r["divergence_index"] = first_diff(o["text"], r["text"]) if changed and not ext else None
            ch += changed
            dv += r["prefix_divergent"]
            if r["prefix_divergent"]:
                idx.append(r["divergence_index"])
        write_jsonl(f, rows)
        tot["rows"] += len(rows)
        tot["changed"] += ch
        tot["divergent"] += dv
        tot["recapped"] += sum(bool(r.get("recapped")) for r in rows)
        med = sorted(idx)[len(idx) // 2] if idx else None
        lines.append(f"| {f.stem} | {len(rows)} | {ch} | {dv} | {med if med is not None else '—'} |")
    lines += ["", f"Total: {tot['rows']} template-B rows; {tot['recapped']} regenerated; **{tot['changed']} texts changed, "
              f"{tot['divergent']} not exact-prefix extensions** of the first-run text. Selective regeneration is "
              "therefore its own retry protocol, not equivalent to a run at 2,048 tokens; the fixed-budget run is "
              "`xb2-*`."]
    (RES2 / "prefix_divergence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-1:]))


if __name__ == "__main__":
    main()
