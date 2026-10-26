"""Key-audit analysis sets, shared by the crossover, branched, commitment and replay analyses.

v1 = data/round2/key_audit.jsonl, one verdict per item (superseded; not released).
v2 = data/round2/key_audit_v2.jsonl: final verdict re-parsed (Final / Revised verdict lines); key_valid (is the keyed
     option unambiguously correct?) and rationale_valid (does the supplied valid rationale avoid errors and
     concessions?) are JUDGED for every item — two separate calls for the 197 items not OK, (one
     two-question call) for the 630 OK items; `axes_judged` records which. 
Sets: all | key-valid (drop key_valid False) | rationale-valid (drop rationale_valid False) | ok (drop either).
Emitting an audited set asserts 100% judged coverage of the items being analysed, so a non-anchor table can never be labelled
key-valid without judged rows; an unparsed judgement counts as invalid.
"""
from __future__ import annotations

from common import read_jsonl
from round2.config2 import DATA2

V1 = DATA2 / "key_audit.jsonl"
V2 = DATA2 / "key_audit_v2.jsonl"
# set name -> (exclusion kind, file suffix)
SETS = {"all": ("none", ""), "key-valid": ("disputed-key", "_keyvalid"),
        "rationale-valid": ("bad-rationale", "_ratvalid"), "ok": ("either", "_okboth")}


def coverage(item_ids) -> tuple[int, int]:
    """how many of these items have JUDGED key and rationale verdicts (not inherited from an OK)."""
    judged = {r["item_id"] for r in read_jsonl(V2) if r.get("axes_judged") in ("both_one_call", "split_two_calls")
              and r.get("key_valid") is not None and r.get("rationale_valid") is not None}
    ids = set(item_ids)
    return len(ids & judged), len(ids)


def excluded(kind: str, version: str = "v2", item_ids=None) -> set:
    """Items to drop for an audited set. `item_ids`: the IDs of the table being analysed — coverage is
    asserted against THEM; without them, against the anchor's item list."""
    if kind == "none":
        return set()
    if version == "v1":
        return {r["item_id"] for r in read_jsonl(V1) if r.get("verdict") != "OK"}
    if item_ids is None:
        from round2.config2 import ANCHOR
        item_ids = [r["id"] for r in read_jsonl(DATA2 / f"items_{ANCHOR}.jsonl")]
    n, tot = coverage(item_ids)
    if n != tot:
        raise SystemExit(f"[audit_sets] judged coverage {n}/{tot} of the items being analysed: {tot - n} item(s) have no "
                         f"judged key/rationale verdict — run key_audit --both for them before emitting {kind} sets")
    out = set()
    for r in read_jsonl(V2):
        kv = r.get("key_valid", r["verdict"] == "OK")
        rv = r.get("rationale_valid", r["verdict"] == "OK")
        if (kind in ("disputed-key", "either") and kv is not True) or \
           (kind in ("bad-rationale", "either") and rv is not True):
            out.add(r["item_id"])
    return out
