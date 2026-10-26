"""Strata per model, shared-known set, (item, X) pairs for evidence.

known    = correct in 7/7 screening probes (5 letter orders + 2 CoT orders)  -> sample 300
boundary = correct in 3-5 of 7                                               -> sample 100
X        = most likely wrong option by mean letter probability over the 5 orders.
Sampling is coordinated across models (one fixed random rank per item, subject round-robin), so items
known by several models land in all their samples. Re-running is safe; a model is (re)written only
when its screening is complete.

  PYTHONPATH=src python -m round2.strata --models qwen3.5-27b       # one model
  PYTHONPATH=src python -m round2.strata                            # every screened model + shared + pairs

(--rules v3, non-anchor models, v2 screens only): a letter row is compliant only if its first token is
A-D AND its A-D mass >= 0.5 (anything else is a format failure). known = all 5 letter rows compliant and correct
AND both CoT rows correct; boundary = >= 3 compliant letter rows and 3/7 <= correct share <= 5/7 over the
compliant letter rows + 2 CoT rows. Own X needs a non-degenerate wrong-option ordering: the top wrong option's
probability summed over the compliant orders > 1e-4; otherwise X = None (the item has no own X and is used only in
shared-X cells; it gets no evidence pair). The previous items file is kept as items_<model>.pre-v3.jsonl.
"""
from __future__ import annotations

import argparse
from collections import Counter

from common import LETTERS, read_jsonl, write_jsonl
from round2.common2 import build_pool, rank
from round2.config2 import (BOUNDARY_RANGE, DATA2, LOCAL_MODELS, N_BOUNDARY, N_KNOWN, SCREEN_COT_ORDERS,
                            SCREEN_LETTER_ORDERS)
from round2.screen import screen_path

N_PROBES = SCREEN_LETTER_ORDERS + SCREEN_COT_ORDERS


def items_path(model: str):
    return DATA2 / f"items_{model}.jsonl"


def score(model: str, pool: list[dict], tag: str | None = None) -> dict | None:
    rows = read_jsonl(screen_path(model, tag))
    by = {}
    for r in rows:
        by.setdefault(r["item_id"], {})[(r["mode"], r["order"])] = r
    complete = sum(len(v) >= N_PROBES for v in by.values())
    if complete < len(pool):
        print(f"[strata] {model}: screening incomplete ({complete}/{len(pool)} items) - skipped")
        return None
    out = {}
    for it in pool:
        v = by[it["id"]]
        letters = [v[("letter", k)] for k in range(SCREEN_LETTER_ORDERS)]
        cots = [v[("cot", k)] for k in range(SCREEN_COT_ORDERS)]
        # : a letter probe with no A-D token in the top-k is a format failure, never an answer. Older
        # screens (before the fix) stored those as exactly uniform probabilities; treat those the same way.
        ff = [r for r in letters if r.get("format_failure") or (r["probs"] and set(r["probs"].values()) == {0.25})]
        valid = [r for r in letters if r not in ff]
        n_correct = sum(r["answer"] == it["key"] for r in valid + cots)
        pmean = ({L: sum(r["probs"][L] for r in valid) / len(valid) for L in LETTERS} if valid
                 else {L: 0.25 for L in LETTERS})
        X = max((L for L in LETTERS if L != it["key"]), key=lambda L: pmean[L])
        out[it["id"]] = dict(it, X=X, n_correct=n_correct, p_mean=pmean, format_failures=len(ff),
                             cot_unparsed=sum(r["answer"] is None for r in cots))
    return out


MASS_MIN, X_MASS_MIN = 0.5, 1e-4


def score_v3(model: str, pool: list[dict], tag: str) -> dict | None:
    rows = read_jsonl(screen_path(model, tag))
    if any("letter_mass" not in r for r in rows if r["mode"] == "letter"):
        raise SystemExit(f"{model}: screen has no letter_mass / first_token (rescreen with the v2 screener)")
    by = {}
    for r in rows:
        by.setdefault(r["item_id"], {})[(r["mode"], r["order"])] = r
    if sum(len(v) >= N_PROBES for v in by.values()) < len(pool):
        print(f"[strata v3] {model}: screening incomplete - skipped")
        return None
    out = {}
    for it in pool:
        v = by[it["id"]]
        letters = [v[("letter", k)] for k in range(SCREEN_LETTER_ORDERS)]
        cots = [v[("cot", k)] for k in range(SCREEN_COT_ORDERS)]
        comp = [r for r in letters if not r.get("format_failure") and r.get("probs")
                and (r.get("first_token") or "").strip() in LETTERS and (r.get("letter_mass") or 0) >= MASS_MIN]
        let_ok = sum(r["answer"] == it["key"] for r in comp)
        cot_ok = sum(r["answer"] == it["key"] for r in cots)
        share = (let_ok + cot_ok) / (len(comp) + len(cots))
        wrong = {L: sum(r["probs"][L] for r in comp) for L in LETTERS if L != it["key"]}
        top = max(wrong, key=lambda L: (wrong[L], -LETTERS.index(L))) if wrong else None
        own_x = bool(comp) and wrong[top] > X_MASS_MIN
        stratum = ("known" if len(comp) == SCREEN_LETTER_ORDERS and let_ok == len(comp) and cot_ok == len(cots)
                   else "boundary" if len(comp) >= 3 and 3 / 7 <= share <= 5 / 7 else None)
        out[it["id"]] = dict(it, X=top if own_x else None, own_X=own_x, x_mass=wrong.get(top, 0.0) if comp else 0.0,
                             x_tie=sum(abs(wrong[L] - wrong[top]) < 1e-12 for L in wrong) > 1 if comp else False,
                             n_compliant=len(comp), n_correct=let_ok + cot_ok, correct_share=share,
                             stratum_v3=stratum, cot_unparsed=sum(r["answer"] is None for r in cots))
    return out


def build_shared_manifest(pool: list[dict]) -> list[dict]:
    """the shared-X manifest is the ANCHOR's items with the anchor's X (X_qwen), one row per anchor
    item. `models_known` / `models_boundary` list the models for which the item is screening-stable (known) or
    boundary — the anchor by its own strata, every other model by the v3 rule on its v2 screen (all 3,000 pool
    items, not only its sampled 400). Asserts X present and != key on every row.
    (The old code flattened every model's rows into one dict, so the last model's X — often None — won.)"""
    from round2.config2 import ANCHOR
    anchor = read_jsonl(items_path(ANCHOR))
    status = {}
    for m in LOCAL_MODELS:
        if m == ANCHOR:
            continue
        if not screen_path(m, "v2").exists():
            print(f"[shared] {m}: no v2 screen, left out of models_known")
            continue
        sc = score_v3(m, pool, "v2")
        if sc:
            status[m] = {i: r["stratum_v3"] for i, r in sc.items()}
    rows = []
    for r in anchor:
        known = [ANCHOR] if r["stratum"] == "known" else []
        bnd = [ANCHOR] if r["stratum"] == "boundary" else []
        known += [m for m, st in status.items() if st.get(r["id"]) == "known"]
        bnd += [m for m, st in status.items() if st.get(r["id"]) == "boundary"]
        rows.append({**{k: r[k] for k in ("id", "subject", "question", "options", "key")},
                     "X": r["X"], "X_qwen": r["X"], "stratum_qwen": r["stratum"], "models_known": known,
                     "models_boundary": bnd, "models_screened": [ANCHOR] + sorted(status)})
    bad = [r["id"] for r in rows if not r["X"] or r["X"] == r["key"]]
    assert not bad, f"shared manifest: X missing or equal to key for {bad[:5]}"
    write_jsonl(DATA2 / "items_shared.jsonl", rows)
    allk = sum(len(r["models_known"]) == len(r["models_screened"]) for r in rows)
    print(f"[shared] manifest {len(rows)} anchor items (X = X_qwen); known for all {len(rows[0]['models_screened'])} "
          f"screened models: {allk}")
    return rows


def sample(scored: list[dict], n: int) -> list[dict]:
    """Subject round-robin in fixed rank order (coordinated across models)."""
    by_subj = {}
    for it in sorted(scored, key=lambda r: rank(r["id"])):
        by_subj.setdefault(it["subject"], []).append(it)
    picked, i = [], 0
    while len(picked) < n and any(by_subj.values()):
        for s in sorted(by_subj):
            if by_subj[s] and len(picked) < n:
                picked.append(by_subj[s].pop(0))
        i += 1
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=None, help="first n pool items (debug, matches screen --n)")
    ap.add_argument("--force", action="store_true", help="rewrite existing items_<model>.jsonl (old -> .v1)")
    ap.add_argument("--screen-tag", default=None, help="read screen_<model>.<tag>.jsonl (e.g. a rescreen)")
    ap.add_argument("--rules", default="v2", choices=["v2", "v3"], help="v3 = stricter compliance rules")
    ap.add_argument("--manifest-only", action="store_true", help="rebuild items_shared.jsonl only")
    args = ap.parse_args()
    pool = build_pool()[: args.n] if args.n else build_pool()
    if args.manifest_only:
        build_shared_manifest(pool)
        return
    models = args.models or LOCAL_MODELS
    for m in (models if args.rules == "v3" else []):
        from round2.config2 import ANCHOR
        if m == ANCHOR:
            print(f"[strata v3] {m}: anchor strata are not rebuilt")
            continue
        sc = score_v3(m, pool, args.screen_tag or "v2")
        if sc is None:
            continue
        known = [r for r in sc.values() if r["stratum_v3"] == "known"]
        boundary = [r for r in sc.values() if r["stratum_v3"] == "boundary"]
        rows = [dict(r, stratum="known") for r in sample(known, N_KNOWN)]
        rows += [dict(r, stratum="boundary") for r in sample(boundary, N_BOUNDARY)]
        if items_path(m).exists():
            import shutil
            shutil.copy(items_path(m), items_path(m).with_name(f"items_{m}.pre-v3.jsonl"))
        write_jsonl(items_path(m), rows)
        noncomp = sum(r["n_compliant"] < SCREEN_LETTER_ORDERS for r in sc.values())
        print(f"[strata v3] {m}: items with a non-compliant letter row {noncomp}/{len(sc)}; known {len(known)} -> "
              f"{sum(r['stratum'] == 'known' for r in rows)}, boundary {len(boundary)} -> "
              f"{sum(r['stratum'] == 'boundary' for r in rows)}; selected without own X "
              f"{sum(not r['own_X'] for r in rows)} (ties {sum(r['x_tie'] for r in rows)})")
    for m in (models if args.rules == "v2" else []):
        if items_path(m).exists() and not args.force:
            print(f"[strata] {m}: {items_path(m).name} exists (deterministic; --force to rewrite)")
            continue
        sc = score(m, pool, args.screen_tag)
        if sc is None:
            continue
        ok = [r for r in sc.values() if r["format_failures"] == 0]     # quarantine items with any failure
        known = [r for r in ok if r["n_correct"] == N_PROBES]
        boundary = [r for r in ok if BOUNDARY_RANGE[0] <= r["n_correct"] <= BOUNDARY_RANGE[1]]
        rows = [dict(r, stratum="known") for r in sample(known, N_KNOWN)]
        rows += [dict(r, stratum="boundary") for r in sample(boundary, N_BOUNDARY)]
        if items_path(m).exists():
            v1 = items_path(m).with_name(f"items_{m}.v1.jsonl")
            if not v1.exists():
                import shutil
                shutil.copy(items_path(m), v1)
        write_jsonl(items_path(m), rows)
        print(f"[strata] {m}: {len(sc) - len(ok)} items quarantined (format failures in screening)")
        dist = Counter(r["n_correct"] for r in sc.values())
        print(f"[strata] {m}: known {len(known)} -> {sum(r['stratum'] == 'known' for r in rows)}, "
              f"boundary {len(boundary)} -> {sum(r['stratum'] == 'boundary' for r in rows)}; "
              f"n_correct dist {dict(sorted(dist.items()))}; CoT unparsed "
              f"{sum(r['cot_unparsed'] for r in sc.values())}/{len(sc) * SCREEN_COT_ORDERS}")

    # shared-known (among models whose items exist) and the evidence pairs (union over models)
    have = [m for m in LOCAL_MODELS if items_path(m).exists()]
    per = {m: read_jsonl(items_path(m)) for m in have}
    if per:
        build_shared_manifest(pool)
        # never drop a pair that may already have evidence (a rescreen can change a model's items)
        pairs = {p["id"]: p for p in read_jsonl(DATA2 / "pairs.jsonl")}
        for m, rows in per.items():
            for r in rows:
                if r.get("X") is None:          # v3: no own X -> shared-X cells only, no pair of its own
                    continue
                pid = f"{r['id']}|{r['X']}"
                p = pairs.setdefault(pid, {k: r[k] for k in ("subject", "question", "options", "key", "X")})
                p["id"], p["item_id"] = pid, r["id"]
                if m not in p.setdefault("models", []):
                    p["models"].append(m)
        write_jsonl(DATA2 / "pairs.jsonl", sorted(pairs.values(), key=lambda p: p["id"]))
        print(f"[strata] evidence pairs (item, X) over {have}: {len(pairs)}")


if __name__ == "__main__":
    main()
