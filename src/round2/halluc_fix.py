"""Evidence target audit (all tiers) + halluc repair.

gemma-3-4b, asked to explain why X is right, often "corrects" the premise on items it knows ("That's
incorrect. The correct answer is..."), so ~8-10% of halluc texts pushed the key or a third option.
  judge   Claude Haiku 5.5 (thinking off) labels which option each evidence text argues for
          -> data/round2/evidence_targets.jsonl  {pair, tier, hash, target}   (spend step audit_r2)
  repair  halluc texts whose stated answer is the key/another option, or that open with a correction
          ("That's incorrect...", "Actually,"): re-sample from gemma-3-4b (T=0.7) up to 4 times with the same
          check; accepted texts replace the old one in evidence.jsonl (backup kept, change log in
          evidence_repairs.jsonl); unresolved ones get "offtarget" (excluded from analysis). Then purges
          main run responses that showed a replaced halluc text.
          The Haiku labels are informational: Haiku with thinking off over-reports "N" and confuses
          options sharing a prefix, so they do not drive repairs.
  audit2  every halluc text shown in baseline run (anchor (item, X) pairs), one Sonnet 5.5 call
          (thinking off): which option does the argument ULTIMATELY support, and does it refer to options by
          number / letter / position? -> data/round2/halluc_audit.jsonl; evidence.jsonl gains the flags
          `offtarget_v2: ["halluc"]` (conclusion = key, the intended flag), `not_x_v2: ["halluc"]` (conclusion is a
          third option or none — e.g. row 217 rejects X without endorsing anything) and `positional: ["halluc"]`
          (backup kept). These new
          flags are applied only when an analysis asks for them (an analysis option, not used in the paper), so both numbers exist.
          (The regex check above looks at the opening and a "correct answer is" sentence only; )
  PYTHONPATH=src python -m round2.halluc_fix judge
  PYTHONPATH=src python -m round2.halluc_fix repair       (GPU: loads gemma-3-4b)
  PYTHONPATH=src python -m round2.halluc_fix audit2 [--est]
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import time

import make_evidence as me
import spend
from backends import make_backend
from common import read_jsonl, write_jsonl
from round2.config2 import DATA2, HALLUC_MODEL, RES2

EV = DATA2 / "evidence.jsonl"
TARGETS = DATA2 / "evidence_targets.jsonl"
JUDGE = "claude-haiku-5-5"
TIERS = ("weak", "halluc", "decept", "valid")
TRIES = 4
JUDGE_PROMPT = ("Below is a multiple-choice question and a short text about it. Which option does the text "
                "argue is the correct answer? Reply with exactly one character: A, B, C or D, or N if the text does "
                "not clearly argue for a single option.\n\n{qblock}\n\nText:\n\"\"\"{text}\"\"\"")


_NORM = re.compile(r"[\s`*\"'“”.$\\]+")   # whitespace, markdown, quotes, dots, $, backslashes
_STATED = re.compile(r"correct answer is\s*:?\s*(.{1,200})", re.I)
_CORRECTION = re.compile(r"^\W*(that['’]?s|this is|the (provided|given|stated) answer is)\s+(not\s+)?"
                         r"(incorrect|wrong|not correct)|^\W*actually,", re.I)


def _n(s: str) -> str:
    return _NORM.sub(" ", s).strip().lower()


def stated_target(text: str, p: dict) -> str | None:
    """Option letter whose text the 'correct answer is...' sentence (or the opening) names, if unique."""
    m = _STATED.search(text)
    head = _n(m.group(1) if m else text[:200])
    hits = [L for L, o in zip("ABCD", p["options"]) if _n(o) and head.startswith(_n(o)[:40])]
    if len(hits) != 1:
        hits = [L for L, o in zip("ABCD", p["options"]) if len(_n(o)) >= 4 and _n(o)[:40] in head]
    return hits[0] if len(hits) == 1 else None


def halluc_off(text: str, p: dict) -> bool:
    """The precise check used for repair decisions (the Haiku judge over-reports 'N' and confuses options
    that share a prefix, so its labels are kept as an informational audit only)."""
    L = stated_target(text, p)
    return bool(_CORRECTION.search(text)) or (L is not None and L != p["X"])


def h(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def judge_texts(be, items: list[tuple[dict, str]]) -> list[str]:
    from common import qblock
    msgs = [[{"role": "user", "content": JUDGE_PROMPT.format(qblock=qblock(p), text=t)}] for p, t in items]
    out = []
    for i in range(0, len(msgs), 64):
        spend.check("anthropic", "audit_r2")
        outs = be.generate(msgs[i:i + 64])
        spend.log("anthropic", JUDGE, "audit_r2", *be.pop_usage(), note=f"{len(outs)} judge calls")
        out += [(o.strip()[:1].upper() if o.strip()[:1].upper() in "ABCDN" else "N") for o in outs]
    return out


def judge_all(be, pairs: dict, ev: list[dict]):
    seen = {(r["pair"], r["tier"], r["hash"]) for r in read_jsonl(TARGETS)}
    todo = [(r["id"], t, r[t]) for r in ev for t in TIERS if (r["id"], t, h(r[t])) not in seen]
    print(f"[audit] judging {len(todo)} texts with {JUDGE}", flush=True)
    labels = judge_texts(be, [(pairs[pid], text) for pid, _, text in todo])
    write_jsonl(TARGETS, [{"pair": pid, "tier": t, "hash": h(text), "target": L}
                          for (pid, t, text), L in zip(todo, labels)], append=True)


def label_map() -> dict:
    return {(r["pair"], r["tier"], r["hash"]): r["target"] for r in read_jsonl(TARGETS)}


def report(pairs, ev):
    lab = label_map()
    for t in TIERS:
        want = (lambda p: p["key"]) if t == "valid" else (lambda p: p["X"])
        c = {"on": 0, "key": 0, "other": 0, "none": 0}
        for r in ev:
            L = lab.get((r["id"], t, h(r[t])))
            p = pairs[r["id"]]
            if L is None:
                continue
            c["on" if L == want(p) else "none" if L == "N" else "key" if L == p["key"] else "other"] += 1
        print(f"[audit] {t:7} {c}")


AUDIT2 = DATA2 / "halluc_audit.jsonl"
AUDIT2_MODEL, AUDIT2_STEP = "claude-sonnet-5-5", "halluc_audit_r2"
AUDIT2_MUST = (217, 245)          # 1-based evidence.jsonl rows the cites; both must be flagged
AUDIT2_PROMPT = ("Below is a multiple-choice question and an argument someone wrote about it.\n\n{qblock}\n\n"
                 "Argument:\n\"\"\"{text}\"\"\"\n\n1. Which option does this argument ULTIMATELY support — its final "
                 "conclusion, not where it starts? 2. Does it refer to options by number, letter or position (e.g. "
                 "'option 2', 'the first choice', '(B)')? Numbers that are part of the subject matter do not count. "
                 "Answer in at most two sentences, then end with exactly two lines:\nSUPPORTS: A, B, C, D or NONE\n"
                 "POSITIONAL: YES or NO")
_SUP = re.compile(r"^\W*SUPPORTS\W*(A|B|C|D|NONE)\b", re.M | re.I)
_POS = re.compile(r"^\W*POSITIONAL\W*(YES|NO)\b", re.M | re.I)


def audit2(est_only: bool):
    from common import qblock
    from round2.config2 import ANCHOR
    pairs = {p["id"]: p for p in read_jsonl(DATA2 / "pairs.jsonl")}
    ev = read_jsonl(EV)
    used = {f"{r['id']}|{r['X']}" for r in read_jsonl(DATA2 / f"items_{ANCHOR}.jsonl")}
    done = {(r["pair"], r["hash"]) for r in read_jsonl(AUDIT2)}
    todo = [(n, r) for n, r in enumerate(ev, 1) if r["id"] in used and r.get("halluc")
            and (r["id"], h(r["halluc"])) not in done]
    todo.sort(key=lambda t: t[0] not in AUDIT2_MUST)
    msgs = [[{"role": "user", "content": AUDIT2_PROMPT.format(qblock=qblock(pairs[r["id"]]), text=r["halluc"])}]
            for _, r in todo]
    est = sum(len(m[0]["content"]) for m in msgs) / 3.5 * 2e-6 + len(msgs) * 120 * 10e-6
    print(f"[audit2] halluc texts used in the baseline run: {len(used)}; todo {len(todo)}; est≈${est:.2f}", flush=True)
    if est_only or not todo:
        return report2(ev, pairs) if not est_only else None
    be = make_backend("anthropic", AUDIT2_MODEL, thinking={"type": "between_tools"}, effort="low")
    for i in range(0, len(todo), 64):
        spend.check("anthropic", AUDIT2_STEP)
        outs = be.generate(msgs[i:i + 64])
        spend.log("anthropic", AUDIT2_MODEL, AUDIT2_STEP, *be.pop_usage(), note=f"{len(outs)} calls")
        rows = []
        for (n, r), o in zip(todo[i:i + 64], outs):
            s, p = _SUP.findall(o or ""), _POS.findall(o or "")
            rows.append({"pair": r["id"], "row": n, "hash": h(r["halluc"]), "supports": s[-1].upper() if s else None,
                         "positional": (p[-1].upper() == "YES") if p else None, "reply": o})
        write_jsonl(AUDIT2, rows, append=True)
        print(f"[audit2] {min(i + 64, len(todo))}/{len(todo)}", flush=True)
    report2(ev, pairs)


def report2(ev, pairs):
    """Write offtarget_v2 / positional flags into evidence.jsonl (backup first) and print counts."""
    lab = {(r["pair"], r["hash"]): r for r in read_jsonl(AUDIT2)}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy(EV, EV.with_name(f"evidence.before-halluc-audit2-{stamp}.jsonl"))
    c = {"X": 0, "key": 0, "other": 0, "NONE": 0, "unparsed": 0, "positional": 0, "new_offtarget": 0}
    for n, r in enumerate(ev, 1):
        a = lab.get((r["id"], h(r.get("halluc") or "")))
        r.pop("offtarget_v2", None), r.pop("positional", None), r.pop("not_x_v2", None)
        if not a:
            continue
        p = pairs[r["id"]]
        s = a["supports"]
        c["unparsed" if s is None else "NONE" if s == "NONE" else "X" if s == p["X"] else "key" if s == p["key"]
          else "other"] += 1
        if s == p["key"]:
            r["offtarget_v2"] = ["halluc"]
            c["new_offtarget"] += "halluc" not in (r.get("offtarget") or [])
        elif s != p["X"]:
            r["not_x_v2"] = ["halluc"]
        if a["positional"]:
            r["positional"] = ["halluc"]
            c["positional"] += 1
    write_jsonl(EV, ev)
    must = {n: (ev[n - 1].get("offtarget_v2"), ev[n - 1].get("not_x_v2"), ev[n - 1].get("positional"))
            for n in AUDIT2_MUST}
    print(f"[audit2] conclusions {c}; required rows {must}")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["judge", "repair", "audit2"])
    ap.add_argument("--est", action="store_true")
    args = ap.parse_args()
    if args.step == "audit2":
        return audit2(args.est)
    me.PROMPT_SET = "round2"
    spend.write_prices()
    pairs = {p["id"]: p for p in read_jsonl(DATA2 / "pairs.jsonl")}
    ev = read_jsonl(EV)
    judge = make_backend("anthropic", JUDGE, thinking={"type": "disabled"})
    judge_all(judge, pairs, ev)
    report(pairs, ev)
    if args.step == "judge":
        return

    bad = [r for r in ev if halluc_off(r["halluc"], pairs[r["id"]])]
    print(f"[repair] halluc texts not arguing for X: {len(bad)}", flush=True)
    gen = make_backend("vllm", HALLUC_MODEL)
    new, tries = {}, {}
    todo = list(bad)
    for k in range(TRIES):
        if not todo:
            break
        outs = gen.generate([me.gen_prompt("halluc", pairs[r["id"]]) for r in todo], max_tokens=220, temperature=0.7)
        texts = [me.postprocess("halluc", o)[0] for o in outs]
        nxt = []
        for r, t in zip(todo, texts):
            tries[r["id"]] = k + 1
            p = pairs[r["id"]]
            if not halluc_off(t, p) and not me.problem("halluc", t, p):
                new[r["id"]] = t
            else:
                nxt.append(r)
        todo = nxt
        print(f"[repair] try {k + 1}: fixed {len(new)}, still off-target {len(todo)}", flush=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy(EV, EV.with_name(f"evidence.before-halluc-fix-{stamp}.jsonl"))
    log = []
    for r in ev:
        if r["id"] in new:
            log.append({"id": r["id"], "tier": "halluc", "how": f"resampled x{tries[r['id']]} (judge)",
                        "old": r["halluc"], "new": new[r["id"]], "at": stamp})
            r["halluc"] = new[r["id"]]
        if halluc_off(r["halluc"], pairs[r["id"]]):
            r["offtarget"] = ["halluc"]
        elif "offtarget" in r:
            del r["offtarget"]
    write_jsonl(EV, ev)
    write_jsonl(DATA2 / "evidence_repairs.jsonl", log, append=True)
    print(f"[repair] replaced {len(new)} of {len(bad)} off-target halluc texts; still off-target (flagged, "
          f"excluded from analysis): {sum('halluc' in r.get('offtarget', []) for r in ev)}", flush=True)

    # purge responses that showed a replaced halluc text (respond2 refills them; idempotent)
    changed = set(new)
    for f in sorted((RES2 / "responses").glob("*/*/*halluc*.jsonl")):
        rows = read_jsonl(f)
        keep = [x for x in rows if f"{x['item_id']}|{x['X']}" not in changed]
        if len(keep) != len(rows):
            shutil.copy(f, f.with_name(f"{f.stem}.before-halluc-fix-{stamp}.jsonl.bak"))
            write_jsonl(f, keep)
            print(f"[repair] purged {len(rows) - len(keep)} rows from {f.relative_to(RES2)}")


if __name__ == "__main__":
    main()
