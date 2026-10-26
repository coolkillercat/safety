"""Shared helpers: pool, option orders, model loading, answer parsing, coordinated ranks."""
from __future__ import annotations

import hashlib
import random
import re

from backends import make_backend
from common import LETTERS, options_text, read_jsonl
from config import DATA
from round2.config2 import COT_INSTR, DATA2, LETTER_INSTR, MODELS, POOL_SIZE, SEED2, SYSTEM

POOL = DATA2 / "pool.jsonl"
ANS = re.compile(r"\banswer\s*[:：]\s*\**\s*\(?([A-D])\)?(?![A-Za-z])", re.I)   # legacy parser (baseline run)
# strict: the final answer is the last line that is exactly "Answer: X" (optional parentheses/period);
# nothing else counts. "answer" is case-insensitive, the letter is not.
ANS_STRICT = re.compile(r"^\s*(?i:answer):\s*\(?([A-D])\)?\s*\.?\s*$", re.M)


def build_pool(n: int = POOL_SIZE) -> list[dict]:
    """3,000 subject-stratified items from the item reduction's 4-option reductions (same seed, same options/key).

    data/items_all_scored.jsonl holds every usable MMLU-Pro test item reduced exactly as the item reduction's
    select_items.reduce_to_four did, so no dataset download is needed here."""
    if POOL.exists():
        return read_jsonl(POOL)
    rows = read_jsonl(DATA / "items_all_scored.jsonl")
    keep = ("id", "subject", "question", "options", "key")
    by_subj: dict[str, list] = {}
    for r in rows:
        by_subj.setdefault(r["subject"], []).append({k: r[k] for k in keep})
    rng = random.Random(SEED2)
    for s in sorted(by_subj):
        rng.shuffle(by_subj[s])
    pool: list[dict] = []
    while len(pool) < n and any(by_subj.values()):
        for s in sorted(by_subj):
            if by_subj[s] and len(pool) < n:
                pool.append(by_subj[s].pop())
    from common import write_jsonl
    write_jsonl(POOL, pool)
    return pool


def order_for(item_id: str, k: int, tag: str) -> list[int] | None:
    """LEGACY option order k (k=0: canonical), as used by screening screening and baseline run probes.
    Each k is drawn independently, so non-canonical orders can repeat (counts how often).
    Kept so crossover and branched runs can pair with baseline run on exactly the same orders."""
    if k == 0:
        return None
    rng = random.Random(f"{SEED2}-{tag}-{item_id}-{k}")
    order = list(range(4))
    while order == [0, 1, 2, 3]:
        rng.shuffle(order)
    return order


def distinct_orders(item_id: str, n: int, tag: str) -> list[list[int] | None]:
    """: n option orders sampled WITHOUT replacement: canonical first, then distinct non-identity
    permutations. Use for new runs (and persist them with write_orders)."""
    import itertools
    perms = [list(p) for p in itertools.permutations(range(4)) if list(p) != [0, 1, 2, 3]]
    random.Random(f"{SEED2}-{tag}-{item_id}-wor").shuffle(perms)
    return [None] + perms[: n - 1]


def write_orders(model: str, tag: str, scheme: str, items: list[dict], n: int):
    """Persist the permutation list per item: data/round2/orders_<model>.jsonl (one row per item+tag)."""
    from common import write_jsonl
    path = DATA2 / f"orders_{model}.jsonl"
    have = {(r["item_id"], r["tag"]) for r in read_jsonl(path)}
    rows = []
    for it in items:
        if (it["id"], tag) in have:
            continue
        orders = (distinct_orders(it["id"], n, tag) if scheme == "wor"
                  else [order_for(it["id"], k, tag) for k in range(n)])
        rows.append({"item_id": it["id"], "tag": tag, "scheme": scheme,
                     "orders": [o if o else [0, 1, 2, 3] for o in orders]})
    write_jsonl(path, rows, append=True)
    return path


def maps(item: dict, order: list[int] | None) -> tuple[str, dict, dict]:
    """options text, display->canonical letter, canonical->display letter."""
    txt, m = options_text(item["options"], order)
    d2c = {L: LETTERS[idx] for L, idx in m.items()}
    return txt, d2c, {c: d for d, c in d2c.items()}


def parse_answer_legacy(text: str) -> str | None:
    """baseline run parser: last inline 'answer: <letter>' anywhere in the text. Can pick up a quoted
    answer ("the professor's answer: C") - an audit found two such false conformity events."""
    m = ANS.findall(text or "")
    return m[-1].upper() if m else None


def parse_answer_strict(text: str) -> str | None:
    """: last line that is exactly 'Answer: <letter>'; otherwise missing."""
    m = ANS_STRICT.findall(text or "")
    return m[-1] if m else None


parse_answer = parse_answer_strict      # default for every run from B' on

_SUBSTANTIVE = re.compile(r"[A-Za-z0-9]")


def parse_final(text: str | None, finish_reason: str | None = None) -> tuple[str | None, bool]:
    """terminal-answer rule -> (displayed letter or None, unresolved).
    The answer is 'Answer: <letter>' on the last non-empty line (a line with any letter or digit); an answer line
    followed by further substantive text is unresolved, not an answer. finish_reason is accepted for the record
    (a capped output whose last line is an answer line still counts; one that reopened the question does not)."""
    lines = [ln for ln in (text or "").splitlines() if _SUBSTANTIVE.search(ln)]
    if not lines:
        return None, False
    m = ANS_STRICT.match(lines[-1])
    if m:
        return m.group(1), False
    return None, bool(ANS_STRICT.search(text or ""))


def final_fields(text: str | None, strict_canonical: str | None, finish_reason: str | None = None) -> dict:
    """Record fields for the terminal-answer rule. The final answer, when there is one, is the same line the strict
    parser takes (the last matching line), so parsed_final is either the strict canonical letter or None."""
    letter, unresolved = parse_final(text, finish_reason)
    return {"parsed_final": strict_canonical if letter else None, "unresolved": unresolved}


FINAL_FIELDS = ("parsed_final", "unresolved")


def require_final_fields(rows: list, label: str):
    """an analysis under the terminal-answer rule refuses a file in which any row lacks the final
    fields, naming the file and the first offending row, instead of failing later with a KeyError."""
    for i, r in enumerate(rows, 1):
        missing = [k for k in FINAL_FIELDS if k not in r]
        if missing:
            raise SystemExit(f"[final] {label}: row {i} (cell {r.get('cell')}, item {r.get('item_id')}, order "
                             f"{r.get('order')}) lacks {missing} — run `python -m round2.reparse_final` before "
                             f"analysing with --final")


def prompt_sha(messages) -> str:
    import json
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def model_revision(name: str) -> str:
    """HF snapshot hash the local cache resolves 'main' to (first 12 chars); API models report their model id."""
    import os
    from pathlib import Path
    cfg = MODELS[name]
    if "hf" not in cfg:
        return cfg.get("model", "api")
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    ref = home / "hub" / f"models--{cfg['hf'].replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip()[:12] if ref.exists() else "unknown"


def identity(messages, order_scheme: str, X: str | None, max_tokens: int, model_rev: str) -> dict:
    """fields stored on every new record so a resume can tell whether it would mix protocols."""
    import os
    return {"prompt_sha": prompt_sha(messages), "order_scheme": order_scheme, "X": X, "max_tokens": max_tokens,
            "model_rev": model_rev, "gpu": os.environ.get("DIVERSITY_GPU_NAME", "unknown"),
            "tp": int(os.environ.get("DIVERSITY_TP") or 0) or None}


def row_scheme(r: dict) -> str:
    """Order scheme of a stored row; rows written before used the legacy probe orders unless marked."""
    if r.get("order_scheme"):
        return r["order_scheme"]
    return "legacy" if r.get("pair_with") == "phaseB" or "order_perm" not in r else "distinct"


def check_resume(rows: list[dict], label: str, scheme: str, max_tokens: int, model_rev: str,
                 x_of: dict | None = None, rebuild=None):
    """Refuse to append to a cell whose existing rows were made under a different protocol. Rows without a stored
    field are checked on what they have: cap defaults to the Round 2 768, scheme as row_scheme, X against the
    item's current X, and order-0 rows (which store their prompt) against the prompt that would be built now.
    a selectively regenerated row (`recapped`, ) is its own protocol, so only its
    token cap may differ from the run being resumed — and that cap must be recorded; X, order scheme, model revision
    and the prompt are validated for it like for any other row (the former blanket exemption is gone)."""
    bad = []
    for r in rows:
        if r.get("recapped"):
            if r.get("max_tokens") is None:
                bad.append(f"recapped row without a recorded max_tokens for {r.get('item_id')} order {r.get('order')}")
        else:
            cap = r.get("max_tokens", COT_MAX_TOKENS_DEFAULT)
            if cap != max_tokens:
                bad.append(f"token cap {cap} != {max_tokens}")
        if row_scheme(r) != scheme:
            bad.append(f"order scheme {row_scheme(r)} != {scheme}")
        if x_of is not None and r["item_id"] in x_of and r.get("X") != x_of[r["item_id"]]:
            bad.append(f"X {r.get('X')} != {x_of[r['item_id']]} for {r['item_id']}")
        rev = r.get("model_rev")
        if rev and rev != "unknown" and model_rev != "unknown" and rev != model_rev:
            bad.append(f"model revision {rev} != {model_rev}")
        if rebuild is not None and r.get("prompt"):
            m = rebuild(r)
            if m is not None and prompt_sha(m) != prompt_sha(r["prompt"]):
                bad.append(f"prompt differs for {r['item_id']} order {r['order']}")
        if r.get("prompt_sha") and rebuild is not None and not r.get("prompt"):
            m = rebuild(r)
            if m is not None and prompt_sha(m) != r["prompt_sha"]:
                bad.append(f"prompt hash differs for {r['item_id']} order {r['order']}")
        if len(bad) >= 5:
            break
    if bad:
        raise SystemExit(f"[resume] {label}: refusing to resume — existing rows were made under a different "
                         f"protocol ({'; '.join(sorted(set(bad))[:5])}). Use a new cell tag or move the old file.")


COT_MAX_TOKENS_DEFAULT = 768


def generate_meta(be, msgs, **kw) -> list[dict]:
    """[{"text", "finish_reason", "completion_tokens"}]; backends without metadata give None fields."""
    try:
        return be.generate(msgs, return_meta=True, **kw)
    except TypeError:
        return [{"text": t, "finish_reason": None, "completion_tokens": None} for t in be.generate(msgs, **kw)]


def letter_full(be, msgs) -> list[dict]:
    """[{"probs" (None on format failure), "format_failure", "first_token", "letter_mass", "top5"}]."""
    if hasattr(be, "letter_probs_full"):
        return be.letter_probs_full(msgs)
    return [{"probs": p, "format_failure": False, "first_token": None, "letter_mass": None, "top5": None}
            for p in be.letter_probs(msgs)]


def rank(item_id: str) -> float:
    """Fixed pseudo-random rank per item (coordinated sampling across models)."""
    return int(hashlib.sha256(f"{SEED2}-rank-{item_id}".encode()).hexdigest()[:12], 16) / 16 ** 12


def load_model(name: str, mock: bool = False, tp: int | None = None):
    cfg = MODELS[name]
    if mock:
        return make_backend("mock", name)
    if cfg["kind"] == "api":
        kind = cfg["api"]
        kw = {"thinking": {"type": "between_tools"}} if kind == "anthropic" else {}
        return make_backend(kind, cfg["model"], **kw)
    import os
    from round2.base_template import render
    tp = tp or int(os.environ.get("DIVERSITY_TP") or cfg["gpus"])    # DIVERSITY_TP: e.g. 4 on 24 GB cards
    return make_backend("vllm", cfg["hf"], tensor_parallel=tp, max_model_len=4096,
                        chat_template_kwargs=cfg.get("ctk"), dtype=cfg.get("dtype", "auto"),
                        completion=render if cfg["kind"] == "base" else None)


def is_base(name: str) -> bool:
    return MODELS[name]["kind"] == "base"


BASE_STOP = ["\nQuestion:", "\n\n\n"]
__all__ = ["COT_INSTR", "LETTER_INSTR", "SYSTEM"]
