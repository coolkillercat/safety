"""How often the legacy `order_for` repeated a permutation within an item, and persist the anchor's orders.

Screening (tag "screen"): 5 letter orders over the 3,000-item pool (the first 2 are also the CoT orders).
Probes (tag "probe"): 3 orders over the anchor's 400 strata items. Each non-canonical order is drawn
independently from the 23 non-identity permutations, so repeats are expected (P(no repeat among 4) ~ 0.75).
Writes results/round2/order_dups.md and data/round2/orders_qwen3.5-27b.jsonl (scheme "legacy").

  PYTHONPATH=src python -m round2.order_dups
"""
from __future__ import annotations

from collections import Counter

from common import read_jsonl
from round2.common2 import order_for, write_orders
from round2.config2 import ANCHOR, DATA2, RES2, SCREEN_LETTER_ORDERS


def dup_stats(ids, n, tag):
    distinct = Counter()
    for i in ids:
        ords = [tuple(order_for(i, k, tag) or (0, 1, 2, 3)) for k in range(n)]
        distinct[len(set(ords))] += 1
    return distinct


def main():
    pool = read_jsonl(DATA2 / "pool.jsonl")
    items = read_jsonl(DATA2 / f"items_{ANCHOR}.jsonl")
    lines = ["# Option-order duplicates under the legacy scheme", "",
             "`order_for` drew each non-canonical order independently, so an item could see the same permutation "
             "twice. New runs use `distinct_orders` (without replacement); the crossover and branched runs keep the legacy "
             "orders so they pair with the baseline run.", ""]
    for name, ids, n, tag in (("screening letter, 5 orders", [r["id"] for r in pool], SCREEN_LETTER_ORDERS, "screen"),
                              ("screening CoT, 2 orders", [r["id"] for r in pool], 2, "screen"),
                              ("probe, 3 orders (anchor items)", [r["id"] for r in items], 3, "probe")):
        d = dup_stats(ids, n, tag)
        tot = sum(d.values())
        rep = tot - d[n]
        lines += [f"## {name}", "", f"Items {tot}; with a repeated order {rep} ({rep / tot:.1%}).", "",
                  "| distinct orders | items |", "| ---: | ---: |"]
        lines += [f"| {k} | {d[k]} |" for k in sorted(d, reverse=True)] + [""]
    by_stratum = {s: [r["id"] for r in items if r["stratum"] == s] for s in ("known", "boundary")}
    lines += ["## probe repeats by stratum (anchor)", ""]
    for s, ids in by_stratum.items():
        d = dup_stats(ids, 3, "probe")
        lines.append(f"- {s}: {len(ids) - d[3]}/{len(ids)} items have two identical probe orders")
    (RES2 / "order_dups.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_orders(ANCHOR, "screen", "legacy", pool, SCREEN_LETTER_ORDERS)
    p = write_orders(ANCHOR, "probe", "legacy", items, 3)
    print("\n".join(lines))
    print("->", RES2 / "order_dups.md", p)


if __name__ == "__main__":
    main()
