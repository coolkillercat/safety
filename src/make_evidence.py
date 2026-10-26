"""Step 2: generate the four evidence texts per item (weak, halluc, decept, valid).

- weak / decept / valid: strong generator (Anthropic API if ANTHROPIC_API_KEY is set,
  else local gemma-3-27b-it via vLLM).
- halluc: weak local model (gemma-3-4b-it) asked sincerely to explain why X is right.

Writes data/evidence.jsonl: {"id", "weak", "halluc", "decept", "valid", "gen": {...}}
Idempotent: skips items already done.

Usage:
  python src/make_evidence.py --phase strong --strong auto --tp 2   # then:
  python src/make_evidence.py --phase halluc --halluc vllm          # merges at the end
  python src/make_evidence.py --strong mock --halluc mock
"""
from __future__ import annotations

import argparse
import re

from backends import have_anthropic_key, make_backend
from common import qblock, read_jsonl, write_jsonl
from config import DATA, EVIDENCE_API_MODEL, EVIDENCE_LOCAL_MODEL, GEN_PROMPTS, HALLUC_MODEL

OUT = DATA / "evidence.jsonl"
# Round 2 switches these from main: --prompt-set round2 (letter-free prompts + letter check),
# --spend-step (log every API batch to results/api_spend.csv and stop at the budget caps).
PROMPT_SET = "round1"
SPEND_STEP: str | None = None


def letterless_qblock(item: dict) -> str:
    """Options without letters: Round 2 shows items under several option orders."""
    opts = "\n".join(f"- {o}" for o in item["options"])
    return f"Question: {item['question']}\n\nOptions:\n{opts}"


def gen_prompt(tier: str, item: dict) -> list[dict]:
    direction_letter = item["key"] if tier == "valid" else item["X"]
    X_text = item["options"]["ABCD".index(direction_letter)]
    if PROMPT_SET == "round2":
        from round2.config2 import GEN_PROMPTS2
        user = GEN_PROMPTS2[tier].format(X_text=X_text, qblock=letterless_qblock(item))
    else:
        user = GEN_PROMPTS[tier].format(X=direction_letter, X_text=X_text, qblock=qblock(item))
    return [{"role": "user", "content": user}]


def clean(s: str) -> str:
    s = s.strip().strip('"').strip()
    # drop leading labels like "Argument:" and keep it short
    for pre in ("Argument:", "Explanation:", "Reason:"):
        if s.startswith(pre):
            s = s[len(pre):].strip()
    return " ".join(s.split())[:900]


STRONG_TIERS = ("weak", "decept", "valid")
REFUSAL = re.compile(r"^(\[refusal|I can(no|'|’)t|I cannot|I won['’]t|I['’]m (not able|unable)|"
                     r"I am (not able|unable)|I apologi[sz]e|Sorry|I['’]m sorry|As an AI|\s*$)", re.I)
# Wrong-direction texts that argue for X and then concede ("However, the actual answer is B")
# would leak the key into the pushing cells. Only checked on tiers that argue for X.
HEDGE = re.compile(r"\b(however|but|note|that said|in reality|in fact|actually|to be clear)\b[^.]{0,80}?"
                   r"\b(correct|actual|right|true|real)\s+(answer|option|choice)\b"
                   r"|\bdisclaimer\b|\bfor (the )?(purposes?|sake) of (this|the) (study|exercise|persuasion|argument)\b"
                   r"|\b(this|the above|my) argument is (deliberately |intentionally )?(misleading|incorrect|flawed|false)\b"
                   r"|\bpersuasion (study|exercise)\b|\b(not|isn['’]t) actually (correct|right)\b", re.I)
# A "valid" text that disputes the key (MMLU-Pro has label noise; Opus sometimes recomputes).
VALID_DOUBT = re.compile(r"\b(cannot|can['’]t|unable to) (fully )?(justify|confirm|verify|reproduce)\b", re.I)
# Opus tends to append a correction to wrong-direction texts: "...feels right. (Note: the correct
# answer is actually D.)". Cut from the first later sentence that opens like that.
_SENT = re.compile(r"(?<=[.!?)\"”])\s+")
_LEAD = r"^[\s(\[*_>\-–—]*"   # markup before the word: "(Note:", "*(Note:", "**Note:**", "--- *Note"
_NOTE_OPEN = re.compile(_LEAD + r"(note|disclaimer)\b", re.I)
_TURN_OPEN = re.compile(_LEAD + r"(however|in reality|actually|that said|to be clear)\b", re.I)
# only a turn that talks about the right answer ("In reality, LiOH forms..." can be the argument itself)
_TURN_BODY = re.compile(r"\b(correct|actual|right|true|real)\s+(answer|option|choice)\b"
                        r"|\b(is|isn['’]t|not)\s+actually\b|\bactually\s+(correct|right)\b", re.I)


def strip_note(text: str) -> tuple[str, bool]:
    parts = _SENT.split(text)
    for i in range(1, len(parts)):
        p = parts[i]
        if _NOTE_OPEN.match(p) or (_TURN_OPEN.match(p) and _TURN_BODY.search(" ".join(parts[i:]))):
            return " ".join(parts[:i]).strip(), True
    return text, False


def postprocess(tier: str, raw: str) -> tuple[str, bool]:
    """clean, then for tiers that argue for X drop an appended correction."""
    text = clean(raw)
    if tier == "valid":
        return text, False
    return strip_note(text)
RETRIES = 2   # re-sample a refused/hedged text this many times, then keep it and flag it
# Round 2 evidence must not point at an option letter (the letters change with the option order).
LETTER_REF = re.compile(r"\b(?i:option|answer|choice)[:\s]*\(?[A-D]\)?(?![\w'’])|(?<![\w(])\([A-D]\)(?!\w)")
FLAG_FIELD = {"refused": "refused", "hedged": "hedged", "lettered": "lettered"}
PART = {"strong": DATA / "evidence_strong.jsonl", "halluc": DATA / "evidence_halluc.jsonl"}


def problem(tier: str, text: str, item: dict) -> str | None:
    """'refused', 'hedged', 'lettered' (Round 2 only) or None for one generated evidence text."""
    if REFUSAL.match(text):
        return "refused"
    if PROMPT_SET == "round2" and LETTER_REF.search(text):
        return "lettered"
    if tier == "valid" and VALID_DOUBT.search(text):
        return "hedged"
    if tier != "valid":
        key = item["key"]
        names_key = re.search(rf"\b(correct|actual|right|true|real)\s+(answer|option|choice)\s+(is|would be|here is)\s+"
                              rf"(option\s+)?\(?{key}\b", text, re.I)
        if names_key or HEDGE.search(text):
            return "hedged"
    return None


def _gen(be, prompts: list) -> list[str]:
    """be.generate, with a budget check before and a spend-log row after each API batch."""
    import spend
    prov = {"AnthropicBackend": "anthropic", "DeepSeekBackend": "deepseek",
            "GeminiBackend": "gemini"}.get(type(be).__name__)
    if SPEND_STEP and prov:
        spend.check(prov, SPEND_STEP)
    outs = be.generate(prompts, max_tokens=220, temperature=0.7)
    if SPEND_STEP and prov:
        spend.log(prov, be.name, SPEND_STEP, *be.pop_usage(), note=f"{len(prompts)} calls")
    return outs


def run_phase(phase: str, todo: list[dict], be, batch: int):
    """Generate one phase's tiers into its own part file (idempotent per item).
    be: one backend for every tier, or a dict tier -> backend (Round 2: per-tier generators)."""
    done = {r["id"] for r in read_jsonl(PART[phase])}
    todo = [it for it in todo if it["id"] not in done]
    tiers = STRONG_TIERS if phase == "strong" else ("halluc",)
    bes = be if isinstance(be, dict) else {t: be for t in tiers}
    gen = {t: bes[t].name for t in tiers} if isinstance(be, dict) else be.name
    print(f"[{phase}] gen={gen} todo={len(todo)}", flush=True)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        rows = [{"id": it["id"], "gen": gen} for it in chunk]
        for tier in tiers:
            be = bes[tier]
            outs = _gen(be, [gen_prompt(tier, it) for it in chunk])
            stripped = [False] * len(rows)
            for k, (r, o) in enumerate(zip(rows, outs)):
                r[tier], stripped[k] = postprocess(tier, o)
            first = [problem(tier, r[tier], it) for r, it in zip(rows, chunk)]
            for _ in range(RETRIES):
                bad = [k for k, (r, it) in enumerate(zip(rows, chunk)) if problem(tier, r[tier], it)]
                if not bad:
                    break
                outs = _gen(be, [gen_prompt(tier, chunk[k]) for k in bad])
                for k, o in zip(bad, outs):
                    rows[k][tier], stripped[k] = postprocess(tier, o)
            for r, s in zip(rows, stripped):
                if s:    # final text had an appended correction cut off
                    r.setdefault("stripped", []).append(tier)
            for r, it, p0 in zip(rows, chunk, first):
                if p0:   # how often the first sample needed a retry (reported per tier by scan_evidence)
                    r.setdefault("retried", []).append(f"{tier}:{p0}")
                p = problem(tier, r[tier], it)
                if p:
                    r.setdefault(FLAG_FIELD[p], []).append(tier)
        write_jsonl(PART[phase], rows, append=True)
        print(f"[{phase}] evidence {min(i + batch, len(todo))}/{len(todo)}", flush=True)


def merge(items: list[dict]):
    """Join the two part files into evidence.jsonl for items that have both."""
    done = {r["id"] for r in read_jsonl(OUT)}
    strong = {r["id"]: r for r in read_jsonl(PART["strong"])}
    hall = {r["id"]: r for r in read_jsonl(PART["halluc"])}
    rows = []
    for it in items:
        i = it["id"]
        if i in done or i not in strong or i not in hall:
            continue
        r = {"id": i, **{t: strong[i][t] for t in STRONG_TIERS}, "halluc": hall[i]["halluc"],
             "gen": {"strong": strong[i]["gen"], "halluc": hall[i]["gen"]},
             **{f: strong[i].get(f, []) + hall[i].get(f, []) for f in ("refused", "hedged", "lettered", "retried", "stripped")}}
        rows.append(r)
    write_jsonl(OUT, rows, append=True)
    print(f"merged {len(rows)} new -> {OUT} (total {len(done) + len(rows)}/{len(items)})")
    for f in ("refused", "hedged", "lettered"):
        n = sum(bool(r[f]) for r in rows)
        if n:
            print(f"WARNING: {n} items still {f} after {RETRIES} retries (see '{f}'; python src/scan_evidence.py)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strong", default="auto", choices=["auto", "anthropic", "vllm", "mock"])
    # The false and true rationales were generated with Claude Opus 5.5 (overrides config's EVIDENCE_API_MODEL).
    ap.add_argument("--api-model", default="claude-opus-5-5",
                    help=f"API generator, recorded per row in 'gen' (config: {EVIDENCE_API_MODEL})")
    ap.add_argument("--halluc", default="vllm", choices=["vllm", "mock"])
    ap.add_argument("--tp", type=int, default=2, help="tensor parallel for local 27B")
    ap.add_argument("--batch", type=int, default=32)
    # Two local vLLM engines do not fit in one process (each grabs 90% of the card),
    # so on the cluster run --phase strong and --phase halluc as separate processes.
    ap.add_argument("--phase", default="all", choices=["all", "strong", "halluc", "merge"])
    # Round 2 (defaults keep the item reduction behaviour)
    ap.add_argument("--items", default=None, help="items file (default data/items.jsonl)")
    ap.add_argument("--out-dir", default=None, help="where evidence*.jsonl go (default data/)")
    ap.add_argument("--prompt-set", default="round1", choices=["round1", "round2"])
    ap.add_argument("--thinking", default="default", choices=["default", "off"],
                    help="off = thinking {type: between_tools} (Sonnet 5.5)")
    ap.add_argument("--spend-step", default=None, help="log API spend under this step name and enforce caps")
    ap.add_argument("--tier-models", default=None,
                    help="per-tier API generators, e.g. weak=claude-sonnet-5-5,decept=claude-opus-5-5,valid=claude-opus-5-5")
    ap.add_argument("--opus-effort", default=None, help="effort for Opus generators (Round 1 used medium)")
    args = ap.parse_args()

    global OUT, PART, PROMPT_SET, SPEND_STEP
    PROMPT_SET, SPEND_STEP = args.prompt_set, args.spend_step
    if args.out_dir:
        from pathlib import Path
        d = Path(args.out_dir)
        OUT = d / "evidence.jsonl"
        PART = {"strong": d / "evidence_strong.jsonl", "halluc": d / "evidence_halluc.jsonl"}
    if SPEND_STEP:
        import spend
        spend.write_prices()
    items = read_jsonl(args.items or DATA / "items.jsonl")
    done = {r["id"] for r in read_jsonl(OUT)}
    todo = [it for it in items if it["id"] not in done]
    print(f"items={len(items)} done={len(done)} todo={len(todo)}")

    if args.phase in ("all", "strong") and todo:
        strong_kind = args.strong
        strong = None
        if strong_kind in ("auto", "anthropic") and (have_anthropic_key() or strong_kind == "anthropic"):
            def api(model):
                sonnet = model.startswith("claude-sonnet-5")
                return make_backend("anthropic", model,
                                    thinking={"type": "between_tools"} if (args.thinking == "off" and sonnet) else None,
                                    effort=args.opus_effort if "opus" in model else None)
            if args.tier_models:
                tm = dict(kv.split("=") for kv in args.tier_models.split(","))
                built = {m: api(m) for m in set(tm.values()) | {args.api_model}}
                strong = {t: built[tm.get(t, args.api_model)] for t in STRONG_TIERS}
            else:
                strong = api(args.api_model)
            try:
                for b in (set(strong.values()) if isinstance(strong, dict) else [strong]):
                    b.ping()
            except Exception as e:
                if strong_kind == "anthropic":
                    raise
                print(f"WARNING: API generator unusable ({type(e).__name__}: {e}); "
                      f"falling back to local {EVIDENCE_LOCAL_MODEL}", flush=True)
                strong = None
        if strong is None and strong_kind in ("auto", "vllm"):
            strong = make_backend("vllm", EVIDENCE_LOCAL_MODEL, tensor_parallel=args.tp)
        elif strong_kind == "mock":
            strong = make_backend("mock", "mock-strong")
        run_phase("strong", todo, strong, args.batch)
    if args.phase in ("all", "halluc") and todo:
        halluc = make_backend("vllm", HALLUC_MODEL) if args.halluc == "vllm" else make_backend("mock", "mock-weak")
        run_phase("halluc", todo, halluc, args.batch)
    merge(items)


if __name__ == "__main__":
    main()
