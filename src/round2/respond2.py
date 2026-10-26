"""Run the 29 cells of the baseline run (the `baseline` cell defines each trial's first answer) for one model.

CoT probe (default): reason, then "Answer: <letter>", greedy, 3 option orders (local) / 1 (API).
Letter probe (--probe letter): T=0 letter probabilities, 1 order (the anchor's method comparison).
Rows go to results/round2/responses/<model>/<probe>/<cell>.jsonl, idempotent per (cell, item, order).

  PYTHONPATH=src python -m round2.respond2 --model qwen3.5-27b
  PYTHONPATH=src python -m round2.respond2 --model qwen3.5-27b --probe letter --cells peer
  PYTHONPATH=src python -m round2.respond2 --model qwen3.5-27b --cells baseline system-none --strata known

Crossover cells (`x-<placement>-<speaker>-<certainty>-<tier>`, built by round2.crossover) run on
any model as SHARED cells: a non-anchor model needs --manifest (data/round2/items_shared.jsonl: the anchor's
items with X = X_qwen) or --items-from qwen3.5-27b, and refuses otherwise. Rows -> responses/<model>/crossover/.
Every new row stores prompt_sha / order_scheme / X / max_tokens / model_rev; resuming a cell made under a
different protocol (e.g. another --max-tokens) is refused. rows store parsed_final / unresolved.
"""
from __future__ import annotations

import argparse

from common import read_jsonl, write_jsonl
from round2.common2 import (BASE_STOP, check_resume, distinct_orders, final_fields, generate_meta, identity, is_base,
                            letter_full, load_model, model_revision, order_for, parse_answer)
from round2.config2 import (ANCHOR, CELL2, CELLS2, COT_MAX_TOKENS, DATA2, LETTER_CELLS, MODELS, PROBE_ORDERS_API,
                            PROBE_ORDERS_LOCAL, RES2)
from round2.frames import build
from round2.strata import items_path

SPEND_STEP = {"claude-sonnet-5.5": "sonnet_run", "deepseek-v4": "deepseek_run", "gemini-3.5-flash": "gemini_run"}


MANIFEST = DATA2 / "items_shared.jsonl"


class XCell:
    """A crossover runs crossover cell run through respond2 (shared cells for cross-model runs)."""
    def __init__(self, name: str):
        self.name = name
        _, self.role, self.speaker, self.certainty, self.tier = name.split("-")
        self.source, self.headcount = "crossover", None


def out_path(model: str, probe: str, cell: str):
    if cell.startswith("x-"):
        probe = "crossover"
    return RES2 / "responses" / model / probe / f"{cell}.jsonl"


def select_cells(spec: list[str] | None, probe: str) -> list:
    if not spec:
        return [CELL2[c] for c in LETTER_CELLS] if probe == "letter" else CELLS2
    out = []
    for s in spec:   # cell names, source names ("peer", "system",...) or crossover cells (x-...)
        out += [XCell(s)] if s.startswith("x-") else [CELL2[s]] if s in CELL2 else [c for c in CELLS2 if c.source == s]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--probe", default="cot", choices=["cot", "letter"])
    ap.add_argument("--cells", nargs="*", default=None, help="cell or source names (default: all 29 / 10 peer)")
    ap.add_argument("--strata", nargs="*", default=["known", "boundary"])
    ap.add_argument("--orders", type=int, default=None, help="default 3 local / 1 API; letter probe 1")
    ap.add_argument("--max-items", type=int, default=None, help="API reduced grid: first N per stratum")
    ap.add_argument("--items-from", default=None, help="model whose item set to use (API runs)")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "mock"])
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--pair-with", default=None, choices=["phaseB"],
                    help="the baseline run's legacy probe orders (pairs with the baseline run); default distinct orders")
    ap.add_argument("--dry-run", action="store_true", help="print the order scheme and job count, run nothing")
    ap.add_argument("--manifest", nargs="?", const=str(MANIFEST), default=None,
                    help="shared-X manifest (default data/round2/items_shared.jsonl)")
    ap.add_argument("--max-tokens", type=int, default=COT_MAX_TOKENS)
    args = ap.parse_args()
    cfg = MODELS[args.model]
    mock = args.backend == "mock"
    n_orders = args.orders or (1 if args.probe == "letter" else
                               PROBE_ORDERS_API if cfg["kind"] == "api" else PROBE_ORDERS_LOCAL)

    cells = select_cells(args.cells, args.probe)
    shared = [c.name for c in cells if isinstance(c, XCell)]
    if shared and args.model != ANCHOR and not (args.manifest or args.items_from == ANCHOR):
        raise SystemExit(f"[respond2] {args.model}: shared cells {shared[:3]} need --manifest or --items-from {ANCHOR} "
                         f"(the anchor's items and X); refusing to run them on the model's own strata")
    if args.manifest:
        man = read_jsonl(args.manifest)
        assert all(r["X"] and r["X"] != r["key"] and r["X"] == r["X_qwen"] for r in man), "manifest X check failed"
        items = [dict(r, stratum=r["stratum_qwen"]) for r in man if r["stratum_qwen"] in args.strata]
        print(f"[respond2] manifest {args.manifest}: {len(items)} items (X = X_qwen), strata {args.strata}; "
              f"known for {args.model} under v3: {sum(args.model in r['models_known'] for r in items)}", flush=True)
    else:
        items = [r for r in read_jsonl(items_path(args.items_from or args.model)) if r["stratum"] in args.strata]
    if args.max_items:
        keep, seen = [], {}
        for r in items:
            seen[r["stratum"]] = seen.get(r["stratum"], 0) + 1
            if seen[r["stratum"]] <= args.max_items:
                keep.append(r)
        items = keep
    ev = {r["id"]: r for r in read_jsonl(DATA2 / "evidence.jsonl")}
    needs_ev = any(c.tier and c.tier != "none" for c in cells)
    missing = [r["id"] for r in items if f"{r['id']}|{r['X']}" not in ev]
    if needs_ev and missing:
        raise SystemExit(f"{len(missing)} items lack evidence for their (item, X) pair; run the evidence step")

    if args.pair_with == "phaseB":
        print("[respond2] orders: legacy (paired)", flush=True)
        scheme, order_of = "legacy", (lambda iid, k: order_for(iid, k, "probe"))
    else:
        print("[respond2] orders: distinct", flush=True)
        scheme, order_of = "distinct", (lambda iid, k: distinct_orders(iid, n_orders, "probe")[k])
    fam = cfg["family"]
    by_id = {it["id"]: it for it in items}

    def make(c, it, k):
        if isinstance(c, XCell):
            from round2.crossover import build_x
            return build_x(it, c.name, ev.get(f"{it['id']}|{it['X']}"), order_of(it["id"], k))
        return build(it, c, ev.get(f"{it['id']}|{it['X']}"), order_of(it["id"], k), args.probe, fam, mock=mock)

    rev = model_revision(args.model)
    jobs = []
    for c in cells:
        have = read_jsonl(out_path(args.model, args.probe, c.name))
        if args.probe == "cot":
            check_resume(have, f"{args.model}/{c.name}", scheme, args.max_tokens, rev,
                         x_of={i: it["X"] for i, it in by_id.items()},
                         rebuild=lambda r, c=c: (make(c, by_id[r["item_id"]], r["order"])["messages"]
                                                 if r["item_id"] in by_id else None))
        done = {(r["item_id"], r["order"]) for r in have}
        jobs += [(c, it, k) for it in items for k in range(n_orders) if (it["id"], k) not in done]
    print(f"[respond2] model={args.model} probe={args.probe} cells={len(cells)} items={len(items)} "
          f"orders={n_orders} max_tokens={args.max_tokens} model_rev={rev} todo={len(jobs)}", flush=True)
    if args.dry_run:
        return
    if not jobs:
        return
    be = load_model(args.model, mock=mock)
    step = SPEND_STEP.get(args.model)
    for i in range(0, len(jobs), args.batch):
        chunk = jobs[i:i + args.batch]
        built = [make(c, it, k) for c, it, k in chunk]
        msgs = [b["messages"] for b in built]
        if step:
            import spend
            spend.check(cfg["api"], step)
        if args.probe == "letter":
            outs = letter_full(be, msgs)
        else:
            outs = generate_meta(be, msgs, max_tokens=args.max_tokens, temperature=0.0,
                                 stop=BASE_STOP if is_base(args.model) else None)
        if step:
            spend.log(cfg["api"], cfg["model"], step, *be.pop_usage(), note=f"{len(msgs)} calls")
        by_cell = {}
        for (c, it, k), b, o in zip(chunk, built, outs):
            row = {"model": args.model, "probe": args.probe, "cell": c.name, "source": c.source, "tier": c.tier,
                   "headcount": c.headcount, "item_id": it["id"], "stratum": it["stratum"],
                   "subject": it["subject"], "order": k, "order_perm": order_of(it["id"], k), "key": it["key"], "X": it["X"], "pushed": b["pushed"]}
            if args.probe == "letter":
                probs = {b["d2c"][L]: v for L, v in o["probs"].items()} if o["probs"] else None
                ans = max(probs, key=probs.get) if probs else None
                row.update(probs=probs, p_key=probs[it["key"]] if probs else None,
                           p_X=probs[it["X"]] if probs else None, format_failure=o["format_failure"],
                           first_token=o["first_token"], letter_mass=o["letter_mass"])
            else:
                a = parse_answer(o["text"])           # strict since B'
                ans = b["d2c"][a] if a else None
                row.update(text=o["text"], parser="strict", parsed_strict=ans, finish_reason=o["finish_reason"],
                           completion_tokens=o["completion_tokens"], **final_fields(o["text"], ans),
                           **identity(b["messages"], scheme, it["X"], args.max_tokens, rev))
            row.update(answer=ans, parsed=ans is not None,
                       yielded=None if ans is None else ans != it["key"],
                       flip_to_X=None if ans is None else ans == it["X"])
            if k == 0:
                row["prompt"] = b["messages"]
            by_cell.setdefault(c.name, []).append(row)
        for name, rows in by_cell.items():
            write_jsonl(out_path(args.model, args.probe, name), rows, append=True)
        print(f"[respond2] {min(i + args.batch, len(jobs))}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
