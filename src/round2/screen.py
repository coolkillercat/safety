"""Screen the 3,000-item pool for one model.

letter-only T=0 under 5 option orders + CoT (greedy) under 2 orders; every probe is stored with its
answer mapped back to the canonical letter. Idempotent per (item, mode, order).

  PYTHONPATH=src python -m round2.screen --model qwen3.5-27b
  PYTHONPATH=src python -m round2.screen --model gemma3-12b --backend mock --n 60
"""
from __future__ import annotations

import argparse

from common import LETTERS, read_jsonl, write_jsonl
from round2.common2 import (BASE_STOP, build_pool, distinct_orders, generate_meta, is_base, letter_full,
                            load_model, order_for, parse_answer, write_orders)
from round2.config2 import CELL2, COT_MAX_TOKENS, DATA2, MODELS, SCREEN_COT_ORDERS, SCREEN_LETTER_ORDERS
from round2.frames import build


def screen_path(model: str, tag: str | None = None):
    return DATA2 / (f"screen_{model}.{tag}.jsonl" if tag else f"screen_{model}.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--backend", default="vllm", choices=["vllm", "mock"])
    ap.add_argument("--n", type=int, default=None, help="first n pool items (debug)")
    ap.add_argument("--batch", type=int, default=512)
    # /: new screens use orders sampled without replacement (persisted in orders_<model>.jsonl),
    # strict answer parsing, and format-failure quarantine; --tag writes a separate file (e.g. a rescreen).
    ap.add_argument("--tag", default=None, help="output screen_<model>.<tag>.jsonl")
    ap.add_argument("--order-scheme", default="wor", choices=["wor", "legacy"])
    args = ap.parse_args()
    mock = args.backend == "mock"
    pool = build_pool()[: args.n] if args.n else build_pool()
    out = screen_path(args.model, args.tag)
    n_ord = max(SCREEN_LETTER_ORDERS, SCREEN_COT_ORDERS)
    otag = f"screen.{args.tag}" if args.tag else "screen"
    if args.order_scheme == "wor":
        ords = {it["id"]: distinct_orders(it["id"], n_ord, otag) for it in pool}
    else:
        ords = {it["id"]: [order_for(it["id"], k, "screen") for k in range(n_ord)] for it in pool}
    if not mock:
        write_orders(args.model, otag, args.order_scheme, pool, n_ord)
    done = {(r["item_id"], r["mode"], r["order"]) for r in read_jsonl(out)}
    jobs = [(it, "letter", k) for it in pool for k in range(SCREEN_LETTER_ORDERS)]
    jobs += [(it, "cot", k) for it in pool for k in range(SCREEN_COT_ORDERS)]
    jobs = [j for j in jobs if (j[0]["id"], j[1], j[2]) not in done]
    print(f"[screen] model={args.model} pool={len(pool)} todo={len(jobs)} done={len(done)}", flush=True)
    if not jobs:
        return
    be = load_model(args.model, mock=mock)
    fam = MODELS[args.model]["family"]
    cell = CELL2["baseline"]
    for mode in ("letter", "cot"):
        mj = [j for j in jobs if j[1] == mode]
        for i in range(0, len(mj), args.batch):
            chunk = mj[i:i + args.batch]
            built = []
            for it, _, k in chunk:
                x_dummy = dict(it, X=next(L for L in LETTERS if L != it["key"]))   # baseline ignores X
                built.append(build(x_dummy, cell, None, ords[it["id"]][k], mode, fam, mock=mock))
            msgs = [b["messages"] for b in built]
            rows = []
            if mode == "letter":
                for (it, _, k), b, p in zip(chunk, built, letter_full(be, msgs)):
                    probs = {b["d2c"][L]: v for L, v in p["probs"].items()} if p["probs"] else None
                    rows.append({"item_id": it["id"], "mode": mode, "order": k, "order_perm": ords[it["id"]][k],
                                 "probs": probs, "answer": max(probs, key=probs.get) if probs else None,
                                 "format_failure": p["format_failure"], "first_token": p["first_token"],
                                 "letter_mass": p["letter_mass"], "top5": p["top5"], "key": it["key"]})
            else:
                stop = BASE_STOP if is_base(args.model) else None
                for (it, _, k), b, g in zip(chunk, built, generate_meta(be, msgs, max_tokens=COT_MAX_TOKENS,
                                                                         temperature=0.0, stop=stop)):
                    a = parse_answer(g["text"])
                    rows.append({"item_id": it["id"], "mode": mode, "order": k, "order_perm": ords[it["id"]][k],
                                 "text": g["text"], "answer": b["d2c"][a] if a else None, "parser": "strict",
                                 "finish_reason": g["finish_reason"], "completion_tokens": g["completion_tokens"],
                                 "key": it["key"]})
            write_jsonl(out, rows, append=True)
            print(f"[screen] {mode} {min(i + args.batch, len(mj))}/{len(mj)}", flush=True)
    print("->", out)


if __name__ == "__main__":
    main()
