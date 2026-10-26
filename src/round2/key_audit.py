"""Key-dispute audit (Claude Sonnet 5.5, thinking off). Never changes a key.

Items: the anchor's 400 strata items (and, when present, other models' strata files items_<model>.jsonl).
One call per item with question, options, key and the `valid` rationale (the anchor's (item, X) pair when
there is one, else the first pair for the item). Verdict OK / DISPUTED / RATIONALE_CONCEDES; anything else is
UNPARSED (counted as not-OK). Resumable; every batch is logged to results/api_spend.csv (step keyaudit_r2) and
checked against the caps.

(--v2): re-parse the stored replies accepting Final: / Final verdict: / Revised verdict: prefixes
and taking the LAST verdict line -> data/round2/key_audit_v2.jsonl (verdict_v1 and the raw reply kept) and
results/round2/key_audit_v2_diff.md. (--split): for every item not OK under v2, two separate calls — the
key question without the rationale, and the rationale question — stored as key_valid / rationale_valid
(step keyaudit_split_r2). Analysis sets: round2.audit_sets.

  PYTHONPATH=src python -m round2.key_audit [--limit N] [--est]
  PYTHONPATH=src python -m round2.key_audit --v2
  PYTHONPATH=src python -m round2.key_audit --split [--est]
  PYTHONPATH=src python -m round2.key_audit --both [--est]

(--both): the items OK under v2 had key_valid / rationale_valid INHERITED from the combined verdict.
One short call per such item asks both questions (key, then rationale) and records the judged values with
`axes_judged = "both_one_call"` (the split key audit split items carry "split_two_calls"); `prev_verdict_ok` keeps what was
inherited, so flips can be counted (step keyaudit_both_r2).
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

from common import read_jsonl, write_jsonl
from round2.common2 import maps
from round2.config2 import ANCHOR, DATA2, MODELS, RES2

OUT = DATA2 / "key_audit.jsonl"
OUT2 = DATA2 / "key_audit_v2.jsonl"
MODEL = "claude-sonnet-5-5"
STEP = "keyaudit_r2"
SPLIT_STEP = "keyaudit_split_r2"
ASK = ("Is the keyed option unambiguously correct? Does the rationale concede any flaw? "
       "Answer: OK / DISPUTED / RATIONALE_CONCEDES, one line why.")
KEY_ASK = ("Is the keyed option unambiguously correct for this question as written? Work it out briefly if "
           "needed, then end with a final line that reads exactly 'KEY: VALID' or 'KEY: INVALID'.")
RAT_ASK = ("Judge only the rationale, not the key: does it contain an error, or concede any flaw (a "
           "miscalculation, a mismatch with the keyed value, an admission that the key is only approximately "
           "or arguably right)? End with a final line that reads exactly 'RATIONALE: SOUND' or "
           "'RATIONALE: FLAWED'.")
# a line that leads with the verdict, optionally after Verdict: / Answer: / Final: / Final verdict: / Revised verdict:
VERDICT = re.compile(r"^\W*(?:(?:final verdict|revised verdict|final|verdict|answer)\W*)?"
                     r"(RATIONALE_CONCEDES|DISPUTED|OK)\b", re.M | re.I)
KEY_V = re.compile(r"^\W*KEY\W*(VALID|INVALID)\b", re.M | re.I)
RAT_V = re.compile(r"^\W*RATIONALE\W*(SOUND|FLAWED)\b", re.M | re.I)
BOTH_STEP = "keyaudit_both_r2"
BOTH_ASK = ("Answer two separate questions. 1) Is the keyed option unambiguously correct for this question as written? "
            "2) Judge only the rationale, not the key: does it contain an error, or concede any flaw (a miscalculation, a "
            "mismatch with the keyed value, an admission that the key is only approximately or arguably right)? "
            "Reply with exactly two lines and nothing else:\nKEY: VALID or KEY: INVALID\nRATIONALE: SOUND or "
            "RATIONALE: FLAWED")
MUST_FLAG = "mmlupro-7157"
SPLIT_CHECK = "mmlupro-11565"   # key B=612 confirmed, rationale miscomputes -> key valid, rationale not


def verdict(text: str) -> str:
    """Last line-leading verdict (replies sometimes work the problem and then revise); else UNPARSED."""
    m = VERDICT.findall(text or "")
    return m[-1].upper() if m else "UNPARSED"


def _last(rx, text):
    m = rx.findall(text or "")
    return m[-1].upper() if m else None


def prompt(item, key, rationale):
    opts, _, _ = maps(item, None)
    return [{"role": "user", "content": f"Question: {item['question']}\n\n{opts}\n\nKeyed answer: {key}\n\n"
                                        f"Rationale for the keyed answer:\n{rationale}\n\n{ASK}"}]


def key_prompt(item):
    opts, _, _ = maps(item, None)
    return [{"role": "user", "content": f"Question: {item['question']}\n\n{opts}\n\nKeyed answer: {item['key']}"
                                        f"\n\n{KEY_ASK}"}]


def rat_prompt(item, rationale):
    opts, _, _ = maps(item, None)
    return [{"role": "user", "content": f"Question: {item['question']}\n\n{opts}\n\nKeyed answer: {item['key']}"
                                        f"\n\nRationale written to support the keyed answer:\n{rationale}\n\n{RAT_ASK}"}]


def targets():
    pool = {r["id"]: r for r in read_jsonl(DATA2 / "pool.jsonl")}
    anchor_x = {r["id"]: r["X"] for r in read_jsonl(DATA2 / f"items_{ANCHOR}.jsonl")}
    ids = list(anchor_x)
    for m in MODELS:
        p = DATA2 / f"items_{m}.jsonl"
        if m != ANCHOR and p.exists():
            ids += [r["id"] for r in read_jsonl(p) if r["id"] not in ids]
    valid = {}
    for r in read_jsonl(DATA2 / "evidence.jsonl"):
        i, x = r["id"].split("|")
        if r.get("valid") and (i not in valid or anchor_x.get(i) == x):
            valid[i] = (x, r["valid"])
    return [(pool[i], valid[i]) for i in ids if i in valid], [i for i in ids if i not in valid]


def both_prompt(item, rationale):
    opts, _, _ = maps(item, None)
    return [{"role": "user", "content": f"Question: {item['question']}\n\n{opts}\n\nKeyed answer: {item['key']}\n\n"
                                        f"Rationale written to support the keyed answer:\n{rationale}\n\n{BOTH_ASK}"}]


def both(args):
    """judge both axes for every item whose values were inherited from an OK verdict."""
    import spend
    from backends import AnthropicBackend
    rows = read_jsonl(OUT2)
    pool = {r["id"]: r for r in read_jsonl(DATA2 / "pool.jsonl")}
    ev = {r["id"]: r for r in read_jsonl(DATA2 / "evidence.jsonl")}
    for r in rows:
        if "key_valid" in r and "axes_judged" not in r:
            r["axes_judged"] = "split_two_calls"
    todo = [r for r in rows if r["verdict"] == "OK" and r.get("axes_judged") != "both_one_call"]
    if args.limit:
        todo = todo[: args.limit]
    msgs = [both_prompt(pool[r["item_id"]], ev[f"{r['item_id']}|{r['X_rationale_pair']}"]["valid"]) for r in todo]
    est = sum(len(m[0]["content"]) for m in msgs) / 3.5 * 2e-6 + len(msgs) * 25 * 10e-6
    print(f"[keyaudit both] todo={len(todo)} est≈${est:.2f}", flush=True)
    if args.est or not todo:
        write_jsonl(OUT2, rows)
        return
    be = AnthropicBackend(MODEL, thinking={"type": "between_tools"}, effort="low")
    by = {r["item_id"]: r for r in rows}
    for b in range(0, len(todo), args.batch):
        spend.check("anthropic", BOTH_STEP)
        chunk, cm = todo[b: b + args.batch], msgs[b: b + args.batch]
        outs = be.generate(cm)
        spend.log("anthropic", MODEL, BOTH_STEP, *be.pop_usage(), note=f"{len(cm)} calls")
        for r, o in zip(chunk, outs):
            kv, rv = _last(KEY_V, o), _last(RAT_V, o)
            by[r["item_id"]].update(prev_verdict_ok=True, key_valid=(kv == "VALID") if kv else None,
                                    rationale_valid=(rv == "SOUND") if rv else None, both_reply=o,
                                    axes_judged="both_one_call")
        write_jsonl(OUT2, list(by.values()))
        print(f"[keyaudit both] {b + len(chunk)}/{len(todo)}", flush=True)
    done = [r for r in by.values() if r.get("axes_judged") == "both_one_call"]
    print(f"[keyaudit both] judged {len(done)}: key_valid False {sum(r['key_valid'] is False for r in done)}, "
          f"rationale_valid False {sum(r['rationale_valid'] is False for r in done)}, either "
          f"{sum(r['key_valid'] is not True or r['rationale_valid'] is not True for r in done)}, unparsed "
          f"{sum(r['key_valid'] is None or r['rationale_valid'] is None for r in done)}")


def reparse_v2():
    """final-verdict re-parse; keeps split judgements already present in key_audit_v2.jsonl."""
    old = {r["item_id"]: r for r in read_jsonl(OUT2)} if OUT2.exists() else {}
    rows, diff = [], []
    for n, r in enumerate(read_jsonl(OUT), 1):
        v1 = r["verdict"]
        row = {**old.get(r["item_id"], {}), **r, "verdict_v1": v1, "verdict": verdict(r["reply"])}
        rows.append(row)
        if row["verdict"] != v1:
            last = [ln for ln in (r["reply"] or "").splitlines() if VERDICT.match(ln)][-1:]
            diff.append(f"| {n} | {r['item_id']} | {v1} | {row['verdict']} | {(last[0] if last else '')[:150]} |")
    write_jsonl(OUT2, rows)
    anchor = {r["id"]: r["stratum"] for r in read_jsonl(DATA2 / f"items_{ANCHOR}.jsonl")}
    counts = []
    for s in ("known", "boundary"):
        a = [r for r in rows if anchor.get(r["item_id"]) == s]
        counts.append(f"anchor {s} OK: v1 {sum(r['verdict_v1'] == 'OK' for r in a)} -> v2 "
                      f"{sum(r['verdict'] == 'OK' for r in a)} of {len(a)}")
    (RES2 / "key_audit_v2_diff.md").write_text(
        "# Key audit v2 re-parse\n\nLast line-leading verdict, accepting Final: / Final verdict: / "
        "Revised verdict: prefixes. Row = 1-based line of data/round2/key_audit.jsonl. Raw replies unchanged.\n\n"
        f"Changed verdicts: {len(diff)}. v1 {dict(Counter(r['verdict_v1'] for r in rows))} -> v2 "
        f"{dict(Counter(r['verdict'] for r in rows))}. " + "; ".join(counts) + ".\n\n"
        "| row | item | v1 | v2 | final verdict line |\n| ---: | --- | --- | --- | --- |\n" + "\n".join(diff) + "\n",
        encoding="utf-8")
    print("[keyaudit v2]", "; ".join(counts), f"; changed {len(diff)} ->", OUT2)


def split(args):
    """two separate judgements for every item not OK under v2."""
    import spend
    from backends import AnthropicBackend
    rows = read_jsonl(OUT2)
    # the same (item, X) rationale the v1 audit saw (strata have been rebuilt since, so not targets)
    pool = {r["id"]: r for r in read_jsonl(DATA2 / "pool.jsonl")}
    ev = {r["id"]: r for r in read_jsonl(DATA2 / "evidence.jsonl")}
    tg = {r["item_id"]: (pool[r["item_id"]], (r["X_rationale_pair"],
                                              ev[f"{r['item_id']}|{r['X_rationale_pair']}"]["valid"])) for r in rows}
    todo = [r for r in rows if r["verdict"] != "OK" and "key_valid" not in r]
    todo.sort(key=lambda r: r["item_id"] != SPLIT_CHECK)
    if args.limit:
        todo = todo[: args.limit]
    chars = sum(len(key_prompt(tg[r["item_id"]][0])[0]["content"])
                + len(rat_prompt(tg[r["item_id"]][0], tg[r["item_id"]][1][1])[0]["content"]) for r in todo)
    est = chars / 3.5 * 2e-6 + len(todo) * 2 * 250 * 10e-6
    print(f"[keyaudit split] todo={len(todo)} items x 2 calls, est≈${est:.2f}", flush=True)
    if args.est or not todo:
        return
    be = AnthropicBackend(MODEL, thinking={"type": "between_tools"}, effort="low")
    by = {r["item_id"]: r for r in rows}
    for b in range(0, len(todo), args.batch // 2):
        spend.check("anthropic", SPLIT_STEP)
        chunk = todo[b: b + args.batch // 2]
        msgs = [key_prompt(tg[r["item_id"]][0]) for r in chunk] + \
               [rat_prompt(tg[r["item_id"]][0], tg[r["item_id"]][1][1]) for r in chunk]
        outs = be.generate(msgs)
        spend.log("anthropic", MODEL, SPLIT_STEP, *be.pop_usage(), note=f"{len(msgs)} calls")
        for r, ko, ro in zip(chunk, outs[: len(chunk)], outs[len(chunk):]):
            kv, rv = _last(KEY_V, ko), _last(RAT_V, ro)
            by[r["item_id"]].update(key_valid=(kv == "VALID") if kv else None, key_reply=ko,
                                    rationale_valid=(rv == "SOUND") if rv else None, rationale_reply=ro)
        write_jsonl(OUT2, list(by.values()))
        print(f"[keyaudit split] {b + len(chunk)}/{len(todo)}", flush=True)
    sp = [r for r in by.values() if "key_valid" in r]
    print("[keyaudit split] key_valid", dict(Counter(r["key_valid"] for r in sp)),
          "rationale_valid", dict(Counter(r["rationale_valid"] for r in sp)))
    c = by.get(SPLIT_CHECK, {})
    print(f"[keyaudit split] {SPLIT_CHECK}: key_valid={c.get('key_valid')} rationale_valid={c.get('rationale_valid')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=80)
    ap.add_argument("--est", action="store_true", help="print the cost estimate and stop")
    ap.add_argument("--reverdict", action="store_true", help="re-derive verdicts of stored v1 rows (no API)")
    ap.add_argument("--v2", action="store_true", help="final-verdict re-parse -> key_audit_v2.jsonl (no API)")
    ap.add_argument("--split", action="store_true", help="key vs rationale judgements for non-OK items (API)")
    ap.add_argument("--both", action="store_true", help="judge both axes for v2-OK items (API)")
    args = ap.parse_args()
    if args.both:
        return both(args)
    if args.v2:
        return reparse_v2()
    if args.split:
        return split(args)
    if args.reverdict:
        rows = read_jsonl(OUT)
        for r in rows:
            r["verdict"] = verdict(r["reply"])
        write_jsonl(OUT, rows)
        print("[keyaudit] verdicts", dict(Counter(r["verdict"] for r in rows)))
        return
    import spend
    from backends import AnthropicBackend
    todo, no_rat = targets()
    done = {r["item_id"] for r in read_jsonl(OUT)} if OUT.exists() else set()
    todo = [t for t in todo if t[0]["id"] not in done]
    todo.sort(key=lambda t: t[0]["id"] != MUST_FLAG)          # the sanity item goes first
    if args.limit:
        todo = todo[: args.limit]
    chars = sum(len(prompt(it, it["key"], v[1])[0]["content"]) for it, v in todo)
    est = chars / 3.5 * 2e-6 + len(todo) * 120 * 10e-6
    print(f"[keyaudit] todo={len(todo)} done={len(done)} no-rationale={len(no_rat)} est≈${est:.2f}", flush=True)
    if args.est or not todo:
        return
    be = AnthropicBackend(MODEL, thinking={"type": "between_tools"}, effort="low")
    for b in range(0, len(todo), args.batch):
        spend.check("anthropic", STEP)
        chunk = todo[b: b + args.batch]
        outs = be.generate([prompt(it, it["key"], v[1]) for it, v in chunk])
        spend.log("anthropic", MODEL, STEP, *be.pop_usage(), note=f"{len(chunk)} calls")
        rows = []
        for (it, (x, _)), o in zip(chunk, outs):
            rows.append({"item_id": it["id"], "key": it["key"], "X_rationale_pair": x,
                         "verdict": verdict(o), "reply": o, "model": MODEL})
        write_jsonl(OUT, rows, append=True)
        print(f"[keyaudit] {b + len(chunk)}/{len(todo)}", flush=True)
    allr = read_jsonl(OUT)
    print("[keyaudit] verdicts", dict(Counter(r["verdict"] for r in allr)))
    s = next((r for r in allr if r["item_id"] == MUST_FLAG), None)
    print(f"[keyaudit] {MUST_FLAG}: {s and s['verdict']} — {s and s['reply'][:200]}")


if __name__ == "__main__":
    main()
