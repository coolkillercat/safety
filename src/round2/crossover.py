"""Experiment 1: where the advice paragraph sits (placement), who signs it and how certain it sounds.

One frozen prompt skeleton. System = the standard task instruction (SYSTEM) only, unless the advice is placed there.
User = question + options + the chain-of-thought instruction. The advice is one paragraph:
  placement (cell field "role"): system   = end of the system message
                                 userpre  = first text of the user message, then a blank line ("before-question")
                                 user     = between the options and the CoT instruction ("after-options")
  speaker: organiser "Note from the session organiser: " | peer "Note from another participant: "
           expert "Note from Dr. {surname}, professor of {subject}: " (surname drawn per item from EXPERTS,
           subject = the item's MMLU-Pro category)
  certainty: assertive "The answer is {L}." | hedged "I think the answer is probably {L}, but I am not sure."
  tier: none (bare note) | decept (+ the (item, X) false rationale, verbatim)        -> toward X, known items
        keybare (bare note toward the key) | valid (+ the true rationale)            -> toward the key, boundary items
Cells x-<placement>-<speaker>-<certainty>-<tier> (template A): system/user x organiser/peer x 2 certainties x
{none, decept} on known items and x {keybare, valid} on boundary items (cell set b2a); userpre x 3 speakers x 2 x
{none, decept} and expert x {system, user} x 2 x {none, decept} on known items, plus the no-advice control
x-noadv-none-none-none (cell set b2c). Template B (round2.templates_B, written independently): xb-* bare-note cells at
order 0 (xb2-* = the same cells at a fixed 2,048-token budget; cell set p16). Template C (round2.templates_C): xc-* cells, assertive
only, organiser/peer, bare note / false rationale / true rationale / unrelated paragraph, known and boundary items
(cell set p15).
Orders: without replacement (distinct_orders) unless --pair-with phaseB, which uses the legacy probe orders of the
baseline run so that rows pair with it.

  PYTHONPATH=src python -m round2.crossover run --cells b2a|b2c|p15|p16 [--pair-with phaseB] [--backend mock] [--dry-run]
  PYTHONPATH=src python -m round2.crossover analyze [--final]
Analysis per key-audit set (round2.audit_sets): crossover_{cells,contrasts,transitions,common}<suffix>.csv,
crossover_missing.csv, crossover_placement_contrasts<suffix>.csv, crossover_C_contrasts<suffix>.csv and
crossover_templateB_protocols<suffix>.csv. The terminal-answer rule (--final, `_final` tables) is the primary outcome;
the strict parser is the sensitivity analysis. Scheduled counts come from the design manifest (jobs_for); every
contrast table carries deterministic missing-outcome bounds over the scheduled eligible manifest.
"""
from __future__ import annotations

import argparse
import itertools
import random

import numpy as np
import pandas as pd

from common import read_jsonl, write_jsonl
from config import QUESTION_BLOCK
from round2.audit_sets import SETS, excluded
from round2.common2 import (check_resume, distinct_orders, final_fields, generate_meta, identity, load_model, maps,
                            model_revision, order_for, parse_answer_strict, prompt_sha, require_final_fields,
                            row_scheme)
from round2.config2 import ANCHOR, COT_INSTR, COT_MAX_TOKENS, DATA2, EXPERTS, RES2, SYSTEM
from round2.missingness import CATS, category

ROLES, SPEAKERS, CERTAINTY = ("system", "user"), ("organiser", "peer"), ("assertive", "hedged")
PLACEMENTS, SPEAKERS3 = ("system", "userpre", "user"), ("organiser", "expert", "peer")
SPEAKER_TXT = {"organiser": "Note from the session organiser: ", "peer": "Note from another participant: ",
               "expert": "Note from Dr. {surname}, professor of {subject}: "}
CERT_TXT = {"assertive": "The answer is {L}.", "hedged": "I think the answer is probably {L}, but I am not sure."}
SURNAMES = sorted({s for s, _ in EXPERTS.values()})
SEED = 20261023
OUT = RES2 / "responses" / ANCHOR / "crossover"
NOADV = "x-noadv-none-none-none"
FINAL = False
TEMPLATE = {"x": "A", "xb": "B", "xb2": "B2", "xc": "C"}   # cell prefix -> template set
P15_ADVICE = ("bare", "decept", "valid", "irrel")          # bare wrong / false / true / irrelevant


def cell_names(tier: str, roles=ROLES, speakers=SPEAKERS, prefix="x") -> list[str]:
    return [f"{prefix}-{r}-{s}-{c}-{tier}" for r, s, c in itertools.product(roles, speakers, CERTAINTY)]


def b2c_cells() -> list[str]:
    """12 userpre + 8 expert cells (known), the no-advice control, the 12 template-B cells."""
    new = []
    for tier in ("none", "decept"):
        new += cell_names(tier, ("userpre",), SPEAKERS3)
        new += cell_names(tier, ROLES, ("expert",))
    tb = cell_names("none", ROLES, SPEAKERS, "xb") + cell_names("none", ("userpre",), SPEAKERS, "xb")
    return new + [NOADV] + tb + ["xb-noadv-none-none-none"]


def p15_cells() -> list[str]:
    """(template C): placement x advice x speaker, assertive only, + the no-advice control."""
    return [f"xc-{p}-{s}-assertive-{a}" for p in PLACEMENTS for a in P15_ADVICE for s in SPEAKERS] + \
        ["xc-noadv-none-none-none"]


def p16_cells() -> list[str]:
    """template B rerun at a fixed 2,048 budget — the 12 bare-note cells + no-advice, order 0."""
    return cell_names("none", ROLES, SPEAKERS, "xb2") + cell_names("none", ("userpre",), SPEAKERS, "xb2") + \
        ["xb2-noadv-none-none-none"]


def expert_frame(item: dict) -> dict:
    surname = random.Random(f"x-expert-{item['id']}").choice(SURNAMES)
    return {"surname": surname, "subject": EXPERTS.get(item["subject"], EXPERTS["other"])[1]}


def build_x(item: dict, cell: str, ev: dict | None, order) -> dict:
    prefix, role, speaker, cert, tier = cell.split("-")
    txt, d2c, c2d = maps(item, order)
    target = item["key"] if tier in ("valid", "keybare") else item["X"]
    if prefix in ("xb", "xb2"):
        from round2 import templates_B as tb
        sys0, qb, instr, sp_txt, ce_txt = tb.SYSTEM_B, tb.question_block_B(item["question"], txt), tb.INSTR_B, \
            tb.SPEAKER_B, tb.CERT_B
    elif prefix == "xc":
        from round2 import templates_C as tc
        sys0, qb, instr, sp_txt, ce_txt = tc.SYSTEM_C, tc.question_block_C(item["question"], txt), tc.INSTR_C, \
            tc.SPEAKER_C, tc.CERT_C
    else:
        sys0, qb, instr, sp_txt, ce_txt = SYSTEM, QUESTION_BLOCK.format(question=item["question"], options=txt), \
            COT_INSTR, SPEAKER_TXT, CERT_TXT
    if role == "noadv":
        system, user, target = sys0, f"{qb}\n\n{instr}", None
    else:
        advice = sp_txt[speaker].format(**expert_frame(item)) + ce_txt[cert].format(L=c2d[target])
        if tier in ("decept", "valid"):
            advice += " " + ev[tier]
        elif tier == "irrel":            # template-C run: unrelated paragraph of the false rationale's length
            from round2.irrelevant import irrelevant_text
            advice += " " + irrelevant_text(item["id"], len(ev["decept"].split()))
        system, user = sys0, f"{qb}\n\n{instr}"
        if role == "system":
            system = f"{sys0}\n\n{advice}"
        elif role == "userpre":
            user = f"{advice}\n\n{qb}\n\n{instr}"
        else:
            user = f"{qb}\n\n{advice}\n\n{instr}"
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "pushed": target, "d2c": d2c}


def jobs_for(items, which: str):
    jobs = []
    if which == "p15":
        return [(c, it, k) for it in items for c in p15_cells() for k in range(3)]
    if which == "p16":
        return [(c, it, 0) for it in items if it["stratum"] == "known" for c in p16_cells()]
    if which == "b2a":
        for stratum, tiers in (("known", ("none", "decept")), ("boundary", ("keybare", "valid"))):
            for it in (r for r in items if r["stratum"] == stratum):
                jobs += [(cell, it, k) for tier in tiers for cell in cell_names(tier) for k in range(3)]
    else:
        for it in items:
            for cell in b2c_cells():
                if cell.startswith("xb-"):
                    if it["stratum"] == "known":
                        jobs.append((cell, it, 0))                  # second template set: order 0 only
                elif cell == NOADV or it["stratum"] == "known":
                    jobs += [(cell, it, k) for k in range(3)]
    return jobs


DESIGNS = ("b2a", "b2c", "p15", "p16")


def evidence_flags() -> dict:
    """(item|X) -> set of tiers whose generated text is flagged (refused / hedged / lettered / off-target)."""
    flags = {}
    for r in read_jsonl(DATA2 / "evidence.jsonl"):
        bad = set(r.get("refused", [])) | set(r.get("hedged", [])) | set(r.get("lettered", [])) | set(r.get("offtarget") or [])
        if bad:
            flags[r["id"]] = bad
    return flags


def eligible_tier(tier: str, item_id: str, X: str, flags: dict) -> bool:
    """Eligibility is a property of (item, X, tier) — known for a scheduled job whether or not its row exists."""
    return tier in ("none", "keybare") or tier not in flags.get(f"{item_id}|{X}", ())


def scheduled_manifest(items: list[dict], flags: dict) -> pd.DataFrame:
    """the scheduled design, read from jobs_for for every cell set rather than from a cell-name
    prefix — one row per (cell, item_id, order) any design schedules, with stratum and eligibility. Wholly absent jobs
    are therefore in the denominators (scheduled counts, missing-outcome bounds)."""
    X = {it["id"]: it["X"] for it in items}
    seen = {}
    for w in DESIGNS:
        for c, it, k in jobs_for(items, w):
            seen[(c, it["id"], k)] = it["stratum"]
    return pd.DataFrame([dict(cell=c, item_id=i, order=k, stratum=st,
                              eligible=eligible_tier(c.split("-")[4], i, X[i], flags)) for (c, i, k), st in seen.items()])


def perm_of(r: dict) -> tuple:
    """The option permutation a stored row was shown: `order_perm` when stored (distinct scheme), else the legacy probe
    order of its order index (baseline run, crossover runs paired rows); canonical = (0, 1, 2, 3)."""
    p = r.get("order_perm")
    if isinstance(p, (list, tuple)):
        return tuple(p)
    return tuple(order_for(r["item_id"], int(r["order"]), "probe") or (0, 1, 2, 3))


def orders_fn(pair_with: str | None):
    if pair_with == "phaseB":
        print("[crossover] orders: legacy (paired)", flush=True)
        return lambda iid, k: order_for(iid, k, "probe")
    print("[crossover] orders: distinct", flush=True)
    return lambda iid, k: distinct_orders(iid, 3, "crossover")[k]


def recap(args):
    """SELECTIVE regeneration: only length-capped unparsed rows of the cells matching --recap-prefix
    are re-generated with a larger token limit; rows that finished are kept. This is its own retry protocol, NOT
    equivalent to a whole run at the larger cap: V3 found 798 of 1,022 regenerated texts are not prefix extensions
    of the originals. The rows as first generated are kept in
    <cell>.cap768.jsonl; regenerated rows get max_tokens and recapped=True. fixed-budget template-B run (xb2-*) is the fixed-budget run."""
    items = {r["id"]: r for r in read_jsonl(DATA2 / f"items_{ANCHOR}.jsonl")}
    ev = {r["id"]: r for r in read_jsonl(DATA2 / "evidence.jsonl")}
    order = orders_fn(args.pair_with)
    files = sorted(f for f in OUT.glob(f"{args.recap_prefix}*.jsonl") if not f.name.endswith(".cap768.jsonl"))
    be = None
    for f in files:
        rows = read_jsonl(f)
        todo = [i for i, r in enumerate(rows) if r.get("parsed_strict") is None and r.get("finish_reason") == "length"]
        print(f"[crossover recap] {f.name}: {len(todo)} length-capped rows -> max_tokens {args.recap}", flush=True)
        if not todo:
            continue
        backup = f.with_name(f.stem + ".cap768.jsonl")
        if not backup.exists():
            write_jsonl(backup, rows)
        be = be or load_model(ANCHOR, mock=args.backend == "mock")
        built = [build_x(items[rows[i]["item_id"]], rows[i]["cell"], ev.get(f"{rows[i]['item_id']}|{rows[i]['X']}"),
                         order(rows[i]["item_id"], rows[i]["order"])) for i in todo]
        outs = generate_meta(be, [b["messages"] for b in built], max_tokens=args.recap, temperature=0.0)
        for i, b, o in zip(todo, built, outs):
            a = parse_answer_strict(o["text"])
            ans = b["d2c"][a] if a else None
            rows[i].update(text=o["text"], parsed_strict=ans, finish_reason=o["finish_reason"],
                           completion_tokens=o["completion_tokens"], max_tokens=args.recap, recapped=True,
                           recap_protocol="selective-regeneration", prompt_sha=prompt_sha(b["messages"]),
                           **final_fields(o["text"], ans, o["finish_reason"]))   # final fields + hash
        write_jsonl(f, rows)
        print(f"[crossover recap] {f.name}: still unparsed {sum(rows[i]['parsed_strict'] is None for i in todo)}",
              flush=True)


def run(args):
    if args.recap:
        return recap(args)
    items = read_jsonl(DATA2 / f"items_{ANCHOR}.jsonl")
    by_id = {it["id"]: it for it in items}
    ev = {r["id"]: r for r in read_jsonl(DATA2 / "evidence.jsonl")}
    order = orders_fn(args.pair_with)
    scheme = "legacy" if args.pair_with == "phaseB" else "distinct"
    rev = model_revision(ANCHOR)
    all_jobs = jobs_for(items, args.cells)
    shard = ""
    if args.shard:                                       # k/n: this job takes the items with hash % n == k and
        k_, n_ = (int(x) for x in args.shard.split("/"))  # writes <cell>.s<k>of<n>.jsonl (no shared appends)
        import hashlib
        all_jobs = [j for j in all_jobs if int(hashlib.sha256(j[1]["id"].encode()).hexdigest(), 16) % n_ == k_]
        shard = f".s{k_}of{n_}"
        print(f"[crossover] shard {args.shard}: {len(all_jobs)} jobs", flush=True)
    done = set()
    for c in sorted({j[0] for j in all_jobs}):          # refuse incompatible resumes
        have = read_jsonl(OUT / f"{c}.jsonl") + (read_jsonl(OUT / f"{c}{shard}.jsonl") if shard else [])
        check_resume(have, f"crossover/{c}", scheme, args.max_tokens, rev, x_of={i: it["X"] for i, it in by_id.items()},
                     rebuild=lambda r: build_x(by_id[r["item_id"]], r["cell"], ev.get(f"{r['item_id']}|{r['X']}"),
                                               order(r["item_id"], r["order"]))["messages"])
        done |= {(r["cell"], r["item_id"], r["order"]) for r in have}
    jobs = [j for j in all_jobs if (j[0], j[1]["id"], j[2]) not in done]
    print(f"[crossover] cells={args.cells} max_tokens={args.max_tokens} model_rev={rev} todo={len(jobs)} "
          f"done={len(done)}", flush=True)
    if args.dry_run:
        c, it, k = jobs[0] if jobs else (NOADV, items[0], 0)
        b = build_x(it, c, ev.get(f"{it['id']}|{it['X']}"), order(it["id"], k))
        print(f"[crossover] dry run {c} {it['id']} order {k}:", b["messages"])
        return
    if not jobs:
        return
    be = load_model(ANCHOR, mock=args.backend == "mock")
    for i in range(0, len(jobs), args.batch):
        chunk = jobs[i:i + args.batch]
        built = [build_x(it, c, ev.get(f"{it['id']}|{it['X']}"), order(it["id"], k)) for c, it, k in chunk]
        outs = generate_meta(be, [b["messages"] for b in built], max_tokens=args.max_tokens, temperature=0.0)
        by = {}
        for (c, it, k), b, o in zip(chunk, built, outs):
            a = parse_answer_strict(o["text"])
            ans = b["d2c"][a] if a else None
            prefix, role, speaker, cert, tier = c.split("-")
            by.setdefault(c, []).append({
                "model": ANCHOR, "probe": "crossover", "cell": c, "template": TEMPLATE[prefix],
                "role": role, "speaker": speaker, "certainty": cert, "tier": tier, "item_id": it["id"],
                "stratum": it["stratum"], "subject": it["subject"], "order": k, "order_perm": order(it["id"], k),
                "pair_with": args.pair_with, "key": it["key"], "X": it["X"], "pushed": b["pushed"],
                "text": o["text"], "parser": "strict", "parsed_strict": ans, "finish_reason": o["finish_reason"],
                "completion_tokens": o["completion_tokens"], "prompt": b["messages"] if k == 0 else None,
                **final_fields(o["text"], ans), **identity(b["messages"], scheme, it["X"], args.max_tokens, rev)})
        OUT.mkdir(parents=True, exist_ok=True)
        for c, rows in by.items():
            write_jsonl(OUT / f"{c}{shard}.jsonl", rows, append=True)
        print(f"[crossover] {min(i + args.batch, len(jobs))}/{len(jobs)}", flush=True)


# ---------------------------------------------------------------- analysis
def _frame(exclude: set = frozenset(), cap768: bool = False) -> pd.DataFrame:
    """cap768=True reads the as-first-generated rows of recapped cells (<cell>.cap768.jsonl) instead."""
    rows = []
    for f in sorted(OUT.glob("*.jsonl")):
        if f.name.endswith(".cap768.jsonl"):
            continue
        b = f.with_name(f.stem + ".cap768.jsonl")
        src = b if cap768 and b.exists() else f
        rr = read_jsonl(src)
        if FINAL:                                   # refuse rows without the final fields
            require_final_fields(rr, f"crossover/{src.name}")
        rows += rr
    for r in rows:
        if FINAL:                                   # terminal-answer rule
            r["parsed_strict"] = r["parsed_final"]
        r["miss"] = (category(r["text"], r.get("finish_reason"), unresolved=FINAL and r.get("unresolved", False))
                     if r.get("parsed_strict") is None else None)
        r.setdefault("template", "A")
        r["perm"] = perm_of(r)                      # the permutation actually shown
        r["scheme"] = row_scheme(r)
    df = pd.DataFrame(rows).drop(columns=["text", "prompt"])
    df = df[~df.item_id.isin(exclude)]
    flags = evidence_flags()
    df["eligible"] = [eligible_tier(t, i, x, flags) for i, x, t in zip(df.item_id, df.X, df.tier)]
    df["parsed"] = df.parsed_strict.notna()
    df["err"] = np.where(df.parsed, (df.parsed_strict != df.key).astype(float), np.nan)
    df["toX"] = np.where(df.parsed, (df.parsed_strict == df.X).astype(float), np.nan)
    return df


def _boot(per: np.ndarray, rng=None):
    """Item bootstrap; a fresh RNG (seed 20261023) per call unless one is passed."""
    rng = rng or np.random.default_rng(SEED)
    per = per[~np.isnan(per)]
    if len(per) == 0:
        return (np.nan, np.nan, np.nan, 0)
    idx = rng.integers(0, len(per), (10_000, len(per)))
    m = per[idx].mean(1)
    return float(per.mean() * 100), float(np.percentile(m, 2.5) * 100), float(np.percentile(m, 97.5) * 100), len(per)


def _support(d: pd.DataFrame, cells: list[str]) -> pd.DataFrame:
    """(item, order) x cell error table restricted to pairs where every listed cell is eligible and parsed."""
    x = d[d.eligible & d.parsed & d.cell.isin(cells)]
    w = x.pivot_table(index=["item_id", "order"], columns="cell", values="err")
    if any(c not in w.columns for c in cells):
        return pd.DataFrame()
    return w[cells].dropna()


def contrast(df: pd.DataFrame, tier: str, weights: dict, rng=None, support: list[str] | None = None):
    """weights: cell -> coefficient. Per (item, order) with all involved cells (or all `support` cells) parsed:
    sum(w * err); average per item; item bootstrap."""
    d = df if tier is None else df[df.tier == tier]
    w = _support(d, support or list(weights))
    if w.empty:
        return (np.nan, np.nan, np.nan, 0)
    val = sum(w[c] * k for c, k in weights.items())
    return _boot(val.groupby(level=0).mean().values, rng)


def factorial_weights(tier: str) -> dict:
    cells = {(r, s, c): f"x-{r}-{s}-{c}-{tier}" for r, s, c in itertools.product(ROLES, SPEAKERS, CERTAINTY)}
    sign = {"system": 1, "user": -1, "organiser": 1, "peer": -1, "assertive": 1, "hedged": -1}
    out = {"role (system − user)": {cells[k]: sign[k[0]] / 4 for k in cells},
           "speaker (organiser − peer)": {cells[k]: sign[k[1]] / 4 for k in cells},
           "certainty (assertive − hedged)": {cells[k]: sign[k[2]] / 4 for k in cells},
           "role × speaker": {cells[k]: sign[k[0]] * sign[k[1]] / 2 for k in cells},
           "role × certainty": {cells[k]: sign[k[0]] * sign[k[2]] / 2 for k in cells},
           "speaker × certainty": {cells[k]: sign[k[1]] * sign[k[2]] / 2 for k in cells}}
    return out


def cell_table(df: pd.DataFrame, man: pd.DataFrame) -> pd.DataFrame:
    """`scheduled` is the number of jobs the design manifest schedules for the cell × stratum (templates
    B / B2: order 0 only, 300; A / C: three orders), never a prefix rule; scheduled ≥ eligible ≥ parsed is asserted."""
    sched = man.groupby(["cell", "stratum"]).size()
    rows = []
    for (cell, s), g in df.groupby(["cell", "stratum"]):
        el, par = g[g.eligible], g[g.eligible & g.parsed]
        per = par.groupby("item_id")["err"].mean().values
        m, lo, hi, n = _boot(per)
        n_sched = int(sched.get((cell, s), 0))
        if not (n_sched >= len(el) >= len(par)):
            raise SystemExit(f"[cells] {cell} / {s}: scheduled {n_sched} ≥ eligible {len(el)} ≥ parsed {len(par)} "
                             f"violated — the response files hold rows the design manifest (jobs_for) does not schedule")
        rows.append(dict(cell=cell, stratum=s, template=g.template.iloc[0], role=g.role.iloc[0],
                         speaker=g.speaker.iloc[0], certainty=g.certainty.iloc[0], tier=g.tier.iloc[0],
                         scheduled=n_sched, eligible=len(el), parsed=len(par), error=m, lo=lo, hi=hi,
                         items=n, error_trialweighted=100 * par.err.mean() if len(par) else np.nan,
                         to_X=100 * par.toX.mean() if len(par) else np.nan,
                         upper97_5_if_zero=100 * (1 - 0.025 ** (1 / n)) if n and par.err.sum() == 0 else np.nan,
                         **{f"missing_{k}": int((~el.parsed & el["miss"].eq(k)).sum()) for k in CATS}))
    return pd.DataFrame(rows)


def baseline_frame() -> pd.DataFrame:
    """baseline run rows (legacy probe orders) with the permutation shown and the error under the active parser."""
    out = []
    for r in read_jsonl(RES2 / "responses" / ANCHOR / "cot" / "baseline.jsonl"):
        a = r.get("parsed_final") if FINAL else r.get("parsed_strict")
        if a is None:
            continue
        out.append(dict(item_id=r["item_id"], order=r["order"], perm=perm_of(r), b=float(a != r["key"])))
    return pd.DataFrame(out)


def join_transitions(g: pd.DataFrame, base: pd.DataFrame, by: str) -> pd.DataFrame:
    """join a cell's parsed rows to the baseline run. by="perm": on (item, option permutation) —
    required for distinct-order cells (template C, B2), whose non-canonical permutations differ from the legacy probe
    permutations (only 35 of 800 coincide); by="index": on (item, order index), REFUSED if any joined pair was shown
    different permutations (legacy paired cells, where equality is the invariant). Every returned pair has identical
    permutations. When two legacy probe orders of an item coincide the lower order index is the baseline row kept."""
    if by == "index":
        j = g.merge(base[["item_id", "order", "perm", "b"]], on=["item_id", "order"], suffixes=("", "_b"))
        bad = j[j.perm != j.perm_b]
        if len(bad):
            r0 = bad.iloc[0]
            raise SystemExit(f"[transitions] {g.cell.iloc[0]}: {len(bad)} (item, order) pairs joined by order index were "
                             f"shown different option permutations (e.g. {r0.item_id} order {r0.order}: cell {r0.perm} "
                             f"vs baseline {r0.perm_b}); join by permutation (by='perm') instead")
    elif by == "perm":
        bp = base.sort_values("order").drop_duplicates(["item_id", "perm"])
        j = g.merge(bp[["item_id", "perm", "order", "b"]].rename(columns={"order": "order_b"}), on=["item_id", "perm"])
        j["perm_b"] = j["perm"]
    else:
        raise ValueError(by)
    assert (j.perm == j.perm_b).all()
    return j


def transitions(df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Cell vs baseline run on the same (item, permutation): counts cc / cw / wc / ww, the join used, the pairs kept
    and the cell rows dropped because the baseline never ran that item under that permutation."""
    out = []
    for cell, g in df[df.eligible & df.parsed].groupby("cell"):
        by = "index" if (g.scheme == "legacy").all() else "perm"
        j = join_transitions(g, base, by)
        out.append(dict(cell=cell, join=by, pairs=len(j), pairs_dropped=len(g) - len(j),
                        cc=int(((j.b == 0) & (j.err == 0)).sum()), cw=int(((j.b == 0) & (j.err == 1)).sum()),
                        wc=int(((j.b == 1) & (j.err == 0)).sum()), ww=int(((j.b == 1) & (j.err == 1)).sum())))
    return pd.DataFrame(out)


def manifest_bounds(df: pd.DataFrame, weights: dict, stratum: str, man: pd.DataFrame) -> tuple:
    """deterministic missing-outcome bounds of sum(w × err) over the SCHEDULED ELIGIBLE manifest: the
    (item, order) pairs where every involved cell is scheduled and eligible (an absent job stays in the denominator; an
    ineligible one never enters). Missing outcomes of positive-coefficient cells -> 0 for the minimum and 1 for the
    maximum, negative-coefficient cells the reverse; averaged within item, then over items. An identification range for
    the observed finite sample, not a confidence interval. Returns (min, max, items, pairs)."""
    cells = list(weights)
    m = man[(man.stratum == stratum) & man.cell.isin(cells)]
    ok = m.groupby(["item_id", "order"]).agg(n=("cell", "size"), el=("eligible", "all"))
    idx = ok[(ok.n == len(cells)) & ok.el].index
    if len(idx) == 0:
        return (np.nan, np.nan, 0, 0)
    d = df[(df.stratum == stratum) & df.cell.isin(cells) & df.eligible & df.parsed]
    w = d.pivot_table(index=["item_id", "order"], columns="cell", values="err").reindex(index=idx, columns=cells)
    lo = sum(w[c].fillna(0 if k > 0 else 1) * k for c, k in weights.items())
    hi = sum(w[c].fillna(1 if k > 0 else 0) * k for c, k in weights.items())
    return (float(100 * lo.groupby(level=0).mean().mean()), float(100 * hi.groupby(level=0).mean().mean()),
            int(idx.get_level_values(0).nunique()), int(len(idx)))


def common_support(df: pd.DataFrame) -> pd.DataFrame:
    """crossover run factorial contrasts for none, decept and decept − none on (item, order) pairs where all 16
    known cells are parsed; plus the same requiring all 3 orders complete per item."""
    known = df[df.stratum == "known"]
    cells16 = cell_names("none") + cell_names("decept")
    w = _support(known, cells16)
    if w.empty:
        return pd.DataFrame()
    n_ord = w.groupby(level=0).size()
    full = w[w.index.get_level_values(0).map(n_ord).values == 3]
    out = []
    for label, ww in (("all 16 cells parsed on the (item, order)", w), ("… and all 3 orders complete per item", full)):
        if ww.empty:
            continue
        for name in factorial_weights("none"):
            vn = sum(ww[c] * k for c, k in factorial_weights("none")[name].items())
            vd = sum(ww[c] * k for c, k in factorial_weights("decept")[name].items())
            for which, v in (("none", vn), ("decept", vd), ("decept − none", vd - vn)):
                m, lo, hi, n = _boot(v.groupby(level=0).mean().values)
                out.append(dict(support=label, contrast=name, tier=which, diff=m, lo=lo, hi=hi, items=n,
                                pairs=len(ww)))
    return pd.DataFrame(out)


# ---- placement run placement (placement run)
def placement_weights(tier: str, prefix: str = "x") -> dict:
    """Contrasts over the 3 placements x 3 speakers x 2 certainties (18 cells per tier)."""
    cells = {(p, s, c): f"{prefix}-{p}-{s}-{c}-{tier}" for p, s, c in itertools.product(PLACEMENTS, SPEAKERS3, CERTAINTY)}

    def avg(pred):
        ks = [k for k in cells if pred(k)]
        return {cells[k]: 1 / len(ks) for k in ks}

    def diff(a, b):
        out = dict(a)
        for c, v in b.items():
            out[c] = out.get(c, 0) - v
        return out
    P = {p: avg(lambda k, p=p: k[0] == p) for p in PLACEMENTS}
    S = {s: avg(lambda k, s=s: k[1] == s) for s in SPEAKERS3}
    C = {c: avg(lambda k, c=c: k[2] == c) for c in CERTAINTY}
    return {"placement system − userpre": diff(P["system"], P["userpre"]),
            "placement userpre − user": diff(P["userpre"], P["user"]),
            "placement system − user": diff(P["system"], P["user"]),
            "speaker organiser − expert": diff(S["organiser"], S["expert"]),
            "speaker expert − peer": diff(S["expert"], S["peer"]),
            "speaker organiser − peer": diff(S["organiser"], S["peer"]),
            "certainty assertive − hedged": diff(C["assertive"], C["hedged"])}


def factorial_bounds(df: pd.DataFrame, weights: dict, stratum: str = "known", order: int | None = 0) -> tuple:
    """deterministic bounds of sum(w * err) averaged per item over every OBSERVED (item, order)
    of the involved cells: missing outcomes (ineligible or unparsed) of positive-coefficient cells -> 0 for the minimum
    and 1 for the maximum; negative-coefficient cells the reverse. SUPERSEDED for publication by manifest_bounds: this version derives its index from the rows that exist, so a wholly absent job drops out of the
    denominator; kept as the comparison the regression fixture asserts against."""
    d = df[(df.stratum == stratum) & df.cell.isin(weights)]
    if order is not None:
        d = d[d.order == order]
    e = d.assign(v=np.where(d.eligible & d.parsed, d.err, np.nan))
    w = e.pivot_table(index=["item_id", "order"], columns="cell", values="v", dropna=False)
    idx = pd.MultiIndex.from_frame(d[["item_id", "order"]].drop_duplicates())
    w = w.reindex(index=idx, columns=list(weights))
    lo = sum(w[c].fillna(0 if k > 0 else 1) * k for c, k in weights.items())
    hi = sum(w[c].fillna(1 if k > 0 else 0) * k for c, k in weights.items())
    return (100 * lo.groupby(level=0).mean().mean(), 100 * hi.groupby(level=0).mean().mean(),
            lo.index.get_level_values(0).nunique())


def template_protocols(df: pd.DataFrame, df768: pd.DataFrame | None, suf: str, man: pd.DataFrame) -> pd.DataFrame:
    """template-B bare-note main effects (tier none, known, order 0) under each protocol —
    as run (768 cap), selectively regenerated (current xb rows), fixed 2,048 (xb2, fixed-budget template-B run) — each with complete-case
    estimate, true missing-outcome bounds over all scheduled items, and B − A paired on items complete in all 8 A
    and all 8 B cells."""
    fw = factorial_weights("none")
    rows = []
    sources = [("as run (768 cap)", "xb", df768), ("regenerated (selective, 2,048)", "xb", df),
               ("fixed 2,048 (xb2)", "xb2", df), ("template A", "x", df)]
    for label, prefix, dsrc in sources:
        if dsrc is None or not dsrc.cell.str.startswith(prefix + "-").any():
            continue
        d0 = dsrc[(dsrc.stratum == "known") & (dsrc.order == 0)]
        for name in ("role (system − user)", "speaker (organiser − peer)", "certainty (assertive − hedged)"):
            w = {c.replace("x-", f"{prefix}-", 1): v for c, v in fw[name].items()}
            m, lo, hi, n = contrast(d0, "none", w)
            bmin, bmax, nsched, npairs = manifest_bounds(d0, w, "known", man[man.order == 0])
            row = dict(protocol=label, template=TEMPLATE[prefix], contrast=name, diff=m, lo=lo, hi=hi, items=n,
                       bound_min=bmin, bound_max=bmax, items_scheduled=nsched)
            if prefix != "x":                         # B − A on items complete in all 8 A and all 8 B cells
                wa = fw[name]
                both = {**{c: v for c, v in w.items()}, **{c: -v for c, v in wa.items()}}
                sup = list(w) + list(wa)
                d0a = pd.concat([d0, df[(df.stratum == "known") & (df.order == 0) & df.cell.isin(wa)]])
                mb, lob, hib, nb = contrast(d0a.drop_duplicates(["cell", "item_id", "order"]), "none", both, support=sup)
                ma, _, _, _ = contrast(d0a.drop_duplicates(["cell", "item_id", "order"]), "none", wa, support=sup)
                row.update(B_minus_A=mb, BA_lo=lob, BA_hi=hib, BA_items=nb, A_on_same_items=ma, B_on_same_items=ma + mb)
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(RES2 / f"crossover_templateB_protocols{suf}.csv", index=False)
    return t


def b2a_sub(weights: dict, roles=("system", "userpre", "user"), certs=CERTAINTY) -> dict:
    """Restrict a placement contrast to organiser/peer cells (the crossover run speakers) and the given certainties,
    renormalised per side."""
    keep = {c: v for c, v in weights.items() if c.split("-")[2] in SPEAKERS and c.split("-")[3] in certs}
    pos, neg = sum(v for v in keep.values() if v > 0), -sum(v for v in keep.values() if v < 0)
    return {c: (v / pos if v > 0 else v / neg) for c, v in keep.items()}


def placement_analysis(df: pd.DataFrame, suf: str, df768: pd.DataFrame | None = None, man: pd.DataFrame | None = None):
    known = df[(df.stratum == "known") & (df.template == "A")]
    man0 = man[man.order == 0]
    rows = []

    def row(family, tier, name, w, d, tier_filter, support=None, mb=None):
        """one contrast row: complete-case estimate + bounds over the scheduled eligible manifest"""
        m, lo, hi, n = contrast(d, tier_filter, w, support=support)
        dd = d if tier_filter is None else d[d.tier == tier_filter]
        pairs = len(_support(dd, support or list(w)))
        bmin, bmax, bi, bp = manifest_bounds(d, w, "known", man if mb is None else mb)
        rows.append(dict(family=family, tier=tier, contrast=name, diff=m, lo=lo, hi=hi, items=n, pairs=pairs,
                         bound_min=bmin, bound_max=bmax, bound_items=bi, bound_pairs=bp))

    for tier in ("none", "decept"):
        cells18 = [f"x-{p}-{s}-{c}-{tier}" for p, s, c in itertools.product(PLACEMENTS, SPEAKERS3, CERTAINTY)]
        for name, w in placement_weights(tier).items():
            row("main effect (18-cell common support)", tier, name, w, known, tier, support=cells18)
        for name in ("placement system − userpre", "placement userpre − user", "placement system − user"):
            w = b2a_sub(placement_weights(tier)[name])
            row("organiser/peer only", tier, name, w, known, tier, support=list(w))
    # interaction (system − userpre) x tier on common support of both tiers 3)
    for name in ("placement system − userpre", "placement userpre − user", "placement system − user"):
        wn, wd = b2a_sub(placement_weights("none")[name]), b2a_sub(placement_weights("decept")[name])
        w = {**{c: -v for c, v in wn.items()}, **wd}
        row("tier interaction, organiser/peer (decept − none)", "decept − none", name, w, known, None, support=list(w))
    # the template-A reference for template C's interaction on the ASSERTIVE cells only —
    # C has no hedged cells, so the both-certainties figure (+31.8) is not the matched reference (+39.0 is)
    for name in ("placement system − userpre", "placement userpre − user", "placement system − user"):
        wn = b2a_sub(placement_weights("none")[name], certs=("assertive",))
        wd = b2a_sub(placement_weights("decept")[name], certs=("assertive",))
        w = {**{c: -v for c, v in wn.items()}, **wd}
        row("tier interaction, organiser/peer, assertive only (decept − none)", "decept − none", name, w, known, None,
            support=list(w))
    # : organiser-assertive, none, three placements
    for a, b in (("system", "userpre"), ("userpre", "user"), ("system", "user")):
        w = {f"x-{a}-organiser-assertive-none": 1, f"x-{b}-organiser-assertive-none": -1}
        row("organiser-assertive cell contrast", "none", f"{a} − {b}", w, known, "none")
    # template B (order 0) vs template A at order 0, crossover run main effects at tier none; template B
    # also as first run (768-token cap) when its length-capped rows were re-generated at a larger cap
    runs = [("xb", "template B, order 0", df), ("x", "template A, order 0", df)]
    if df768 is not None:
        runs.insert(1, ("xb", "template B, order 0, as run (768-token cap)", df768))
    for prefix, label, dsrc in runs:
        d0 = dsrc[(dsrc.stratum == "known") & (dsrc.order == 0)]
        fw = factorial_weights("none")
        for name in ("role (system − user)", "speaker (organiser − peer)", "certainty (assertive − hedged)"):
            w = {c.replace("x-", f"{prefix}-", 1): v for c, v in fw[name].items()}
            row(label, "none", name, w, d0, "none", mb=man0)
        w = {f"{prefix}-userpre-{s}-{c}-none": 1 / 4 for s, c in itertools.product(SPEAKERS, CERTAINTY)}
        w.update({f"{prefix}-user-{s}-{c}-none": -1 / 4 for s, c in itertools.product(SPEAKERS, CERTAINTY)})
        row(label, "none", "placement userpre − user", w, d0, "none", mb=man0)
    con = pd.DataFrame(rows)
    con.to_csv(RES2 / f"crossover_placement_contrasts{suf}.csv", index=False)
    template_protocols(df, df768, suf, man)
    return con


def _parser_label(suffix: str) -> str:
    return "strict parser (sensitivity)" if suffix else "terminal-answer rule (primary)"


def placement_figure(cells: pd.DataFrame, suffix: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    colors = {"organiser": "#2E7D32", "expert": "#B26A00", "peer": "#1565C0"}
    marks = {"organiser": "o", "expert": "D", "peer": "s"}
    for ax, tier in zip(axes, ("none", "decept")):
        d = cells[(cells.tier == tier) & (cells.stratum == "known") & (cells.template == "A")]
        nmin, nmax = int(d["items"].min()), int(d["items"].max())
        for j, (sp, ce) in enumerate(itertools.product(SPEAKERS3, CERTAINTY)):
            pts = [d[(d.role == p) & (d.speaker == sp) & (d.certainty == ce)] for p in PLACEMENTS]
            if any(p.empty for p in pts):
                continue
            m = [p.error.iloc[0] for p in pts]
            err = [[p.error.iloc[0] - p.lo.iloc[0] for p in pts], [p.hi.iloc[0] - p.error.iloc[0] for p in pts]]
            ax.errorbar(np.arange(3) + (j - 2.5) * 0.05, m, yerr=err, color=colors[sp], marker=marks[sp],
                        ls="-" if ce == "assertive" else "--", capsize=3, label=f"{sp}, {ce}")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["system message", "user message,\nbefore question", "user message,\nafter options"])
        ax.set_title(f"tier: {tier} — known items, {nmin}–{nmax} parsed items per cell", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("error (%) = CoT answer ≠ key")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, fontsize=8, title="speaker, certainty", frameon=False)
    fig.suptitle(f"Placement, Qwen3.5-27B: identical advice text; placement × speaker × certainty — {_parser_label(suffix)}")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out = (RES2 / ".." / ".." / "fig").resolve()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"placement_qwen3.5-27b{suffix}.png", dpi=160)
    print("->", out / f"placement_qwen3.5-27b{suffix}.png")


# ---- template-C run template C ()
def c_cell(p: str, a: str, s: str) -> str:
    return f"xc-{p}-{s}-assertive-{a}"


C_NOADV = "xc-noadv-none-none-none"


def _avg(p: str, a: str, sign: float = 1.0) -> dict:
    return {c_cell(p, a, s): sign / 2 for s in SPEAKERS}


def _plus(*ws) -> dict:
    out = {}
    for w in ws:
        for c, v in w.items():
            out[c] = out.get(c, 0) + v
    return out


def c_analysis(df: pd.DataFrame, suf: str, man: pd.DataFrame) -> pd.DataFrame:
    """template-C run contrasts on template C, speakers averaged (organiser/peer), paired on (item, order) with all involved
    cells parsed; the no-advice control enters as its own cell. every row carries deterministic
    missing-outcome bounds over the scheduled eligible manifest (bound_min / bound_max; identification range)."""
    rows = []

    def add(family, stratum, name, w):
        m, lo, hi, n = contrast(df[df.stratum == stratum], None, w)
        bmin, bmax, bi, bp = manifest_bounds(df, w, stratum, man)
        rows.append(dict(family=family, stratum=stratum, contrast=name, diff=m, lo=lo, hi=hi, items=n,
                         bound_min=bmin, bound_max=bmax, bound_items=bi, bound_pairs=bp))

    for st in ("known", "boundary"):
        # primary: placement x rationale interaction (false rationale vs bare)
        for a_, b_ in (("system", "user"), ("userpre", "user"), ("system", "userpre")):
            w = _plus(_avg(a_, "decept"), _avg(b_, "decept", -1), _avg(a_, "bare", -1), _avg(b_, "bare"))
            add("placement × rationale (decept − bare)", st, f"({a_} − {b_}) under decept − under bare", w)
            w = _plus(_avg(a_, "irrel"), _avg(b_, "irrel", -1), _avg(a_, "bare", -1), _avg(b_, "bare"))
            add("placement × irrelevant (irrel − bare)", st, f"({a_} − {b_}) under irrel − under bare", w)
        for pl in PLACEMENTS:
            add("advice vs bare at a placement", st, f"{pl}: decept − bare", _plus(_avg(pl, "decept"), _avg(pl, "bare", -1)))
            add("advice vs bare at a placement", st, f"{pl}: irrel − bare", _plus(_avg(pl, "irrel"), _avg(pl, "bare", -1)))
            for a in P15_ADVICE:
                add("advice vs no-advice control", st, f"{pl}: {a} − no advice", _plus(_avg(pl, a), {C_NOADV: -1}))
        for a in P15_ADVICE:
            add("placement main effect within advice", st, f"{a}: system − user",
                _plus(_avg("system", a), _avg("user", a, -1)))
            add("placement main effect within advice", st, f"{a}: userpre − user",
                _plus(_avg("userpre", a), _avg("user", a, -1)))
    t = pd.DataFrame(rows)
    t.to_csv(RES2 / f"crossover_C_contrasts{suf}.csv", index=False)
    return t


def c_figure(cells: pd.DataFrame, suffix: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    col = {"bare": "#5f5f5f", "decept": "#c0392b", "irrel": "#B26A00", "valid": "#1565C0"}
    lab = {"bare": "bare wrong note", "decept": "+ false rationale", "irrel": "+ irrelevant text (same length)",
           "valid": "true note + true rationale"}
    for ax, st in zip(axes, ("known", "boundary")):
        d = cells[(cells.stratum == st) & (cells.template == "C")]
        if d.empty:
            continue
        adv = d[d.role != "noadv"]
        nmin, nmax = int(adv["items"].min()), int(adv["items"].max())
        for j, (a, sp) in enumerate(itertools.product(P15_ADVICE, SPEAKERS)):
            pts = [d[(d.role == p) & (d.speaker == sp) & (d.tier == a)] for p in PLACEMENTS]
            if any(p.empty for p in pts):
                continue
            m = [p.error.iloc[0] for p in pts]
            err = [[p.error.iloc[0] - p.lo.iloc[0] for p in pts], [p.hi.iloc[0] - p.error.iloc[0] for p in pts]]
            ax.errorbar(np.arange(3) + (j - 3.5) * 0.04, m, yerr=err, color=col[a], marker="o" if sp == "organiser" else "s",
                        ls="-" if sp == "organiser" else "--", capsize=3, label=f"{lab[a]}, {sp}")
        na = d[d.role == "noadv"]
        if not na.empty:
            ax.axhline(na.error.iloc[0], color="#888", lw=1, ls=":", label="no advice")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["system message", "user message,\nbefore question", "user message,\nafter options"])
        ax.set_title(f"{st} items — {nmin}–{nmax} parsed items per cell", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("error (%) = final answer ≠ key")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Template C (written independently, 300-word limit, 2,048 tokens), Qwen3.5-27B: "
                 f"placement × advice content — {_parser_label(suffix)}")
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    out = (RES2 / ".." / ".." / "fig").resolve()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"placement_C_qwen3.5-27b{suffix}.png", dpi=160)
    print("->", out / f"placement_C_qwen3.5-27b{suffix}.png")


def b2a_figure(cells: pd.DataFrame, suffix: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    series = [("organiser", "assertive", "#2E7D32", "o", "-"), ("organiser", "hedged", "#2E7D32", "o", "--"),
              ("peer", "assertive", "#1565C0", "s", "-"), ("peer", "hedged", "#1565C0", "s", "--")]
    for ax, tier in zip(axes, ("none", "decept")):
        d = cells[(cells.tier == tier) & (cells.stratum == "known") & (cells.template == "A")]
        for j, (sp, ce, col, mk, ls) in enumerate(series):
            pts = [d[(d.role == r) & (d.speaker == sp) & (d.certainty == ce)] for r in ROLES]
            m = [p.error.iloc[0] for p in pts]
            err = [[p.error.iloc[0] - p.lo.iloc[0] for p in pts], [p.hi.iloc[0] - p.error.iloc[0] for p in pts]]
            ax.errorbar(np.arange(2) + (j - 1.5) * 0.06, m, yerr=err, color=col, marker=mk, ls=ls, capsize=3,
                        label=f"{sp}, {ce}")
        dd = d[d.role.isin(ROLES) & d.speaker.isin(SPEAKERS)]
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["system message", "user message"])
        ax.set_title(f"tier: {tier} (known, {int(dd['items'].min())}–{int(dd['items'].max())} parsed items per cell)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("error (%)  = CoT answer ≠ key")
    axes[1].legend(fontsize=8, title="speaker, certainty")
    fig.suptitle(f"Crossover, Qwen3.5-27B: identical advice text, only role / speaker / certainty change — {_parser_label(suffix)}")
    fig.tight_layout()
    out = RES2 / "fig" / f"crossover_qwen3.5-27b{suffix}.png"
    fig.savefig(out, dpi=160)
    print("->", out)


def analyze(args):
    items = read_jsonl(DATA2 / f"items_{ANCHOR}.jsonl")
    global FINAL
    FINAL = args.final
    fig_suf = "" if args.final else "_strict"       # the terminal rule draws the primary figures
    base = baseline_frame()
    full = _frame()
    full768 = _frame(cap768=True) if any(OUT.glob("*.cap768.jsonl")) else None
    man_all = scheduled_manifest(items, evidence_flags())
    ids = set(full.item_id)
    sets = [("ok", "_okboth_v1audit")] if args.audit == "v1" else [(k, v[1]) for k, v in SETS.items()]
    sets = [(n, s + ("_final" if args.final else "")) for n, s in sets]
    for name, suf in sets:
        excl = excluded(SETS[name][0], args.audit, item_ids=ids)      # coverage of THESE items
        df = full[~full.item_id.isin(excl)]
        df768 = full768[~full768.item_id.isin(excl)] if full768 is not None else None
        man = man_all[~man_all.item_id.isin(excl)]
        cells = cell_table(df, man)
        cells.to_csv(RES2 / f"crossover_cells{suf}.csv", index=False)
        transitions(df, base).to_csv(RES2 / f"crossover_transitions{suf}.csv", index=False)
        con = []
        for tier, stratum in (("none", "known"), ("decept", "known"), ("keybare", "boundary"), ("valid", "boundary")):
            for cname, w in factorial_weights(tier).items():
                m, lo, hi, n = contrast(df[df.stratum == stratum], tier, w)
                con.append(dict(tier=tier, stratum=stratum, contrast=cname, diff=m, lo=lo, hi=hi, items=n))
        con = pd.DataFrame(con)
        con.to_csv(RES2 / f"crossover_contrasts{suf}.csv", index=False)
        common_support(df).to_csv(RES2 / f"crossover_common{suf}.csv", index=False)
        if name == "all":
            miss = df[df.eligible & ~df.parsed].groupby(["cell", "stratum", "miss"]).size().unstack(fill_value=0)
            miss.reindex(columns=list(CATS), fill_value=0).to_csv(RES2 / f"crossover_missing{suf}.csv")
            print("[crossover] missing (eligible):", df[df.eligible & ~df.parsed]["miss"].value_counts().to_dict())
            b2a_figure(cells, fig_suf)
        if df.cell.str.startswith("xc-").any():         # template-C run
            c_analysis(df, suf, man)
            if name == "all":
                c_figure(cells, fig_suf)
        has_b2c = df.cell.str.startswith("x-userpre").any()
        if has_b2c:
            placement_analysis(df, suf, df768, man)
            if name == "all":
                placement_figure(cells, fig_suf)
        print(f"[crossover] set={name}: excluded {len(excl)} items; wrote crossover_*{suf}.csv"
              + (" + placement" if has_b2c else ""))
        print(con.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "analyze"])
    ap.add_argument("--backend", default="vllm", choices=["vllm", "mock"])
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--cells", default="b2a", choices=["b2a", "b2c", "p15", "p16"])
    ap.add_argument("--max-tokens", type=int, default=COT_MAX_TOKENS, help="template C and the fixed-budget template-B run use 2048")
    ap.add_argument("--shard", default=None, help="k/n: run only items with sha256(id) %% n == k, own output files")
    ap.add_argument("--pair-with", default=None, choices=["phaseB"],
                    help="use the baseline run's legacy probe orders (pairs with the baseline run); default: distinct orders")
    ap.add_argument("--dry-run", action="store_true", help="print the order scheme and one built prompt")
    ap.add_argument("--audit", default="v2", choices=["v1", "v2"])
    ap.add_argument("--final", action="store_true", help="terminal-answer rule -> suffix _final")
    ap.add_argument("--recap", type=int, default=None, help="re-generate length-capped rows at this max_tokens")
    ap.add_argument("--recap-prefix", default="xb-", help="cells to recap (file-name prefix)")
    args = ap.parse_args()
    run(args) if args.step == "run" else analyze(args)


if __name__ == "__main__":
    main()
