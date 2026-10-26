"""Shared helpers: items, prompt assembly, jsonl IO."""
from __future__ import annotations

import json
import random
from pathlib import Path

from config import (ABSTAIN_LINES, ANSWER_INSTR, EVIDENCE_LINE, EVIDENCE_TIERS, LETTERS,
                    NOISE_LINES, N_PEERS, OFF_FRAME, ON_FRAME, PEER_NAMES, QUESTION_BLOCK,
                    RATIONALE_INSTR, STANCE_LINES, SYSTEM)


def read_jsonl(p: Path) -> list[dict]:
    if not Path(p).exists():
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p: Path, rows: list[dict], append: bool = False):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a" if append else "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def options_text(options: list[str], order: list[int] | None = None) -> tuple[str, dict]:
    """Render options; returns text and a map new_letter -> original index."""
    order = order or list(range(len(options)))
    lines, mapping = [], {}
    for i, idx in enumerate(order):
        L = LETTERS[i]
        lines.append(f"{L}. {options[idx]}")
        mapping[L] = idx
    return "\n".join(lines), mapping


def qblock(item: dict, order: list[int] | None = None) -> str:
    txt, _ = options_text(item["options"], order)
    return QUESTION_BLOCK.format(question=item["question"], options=txt)


def mock_tags(item: dict, tier: str | None, direction: str | None) -> str:
    """Hidden tags the MockBackend reads; harmless for real models? No: only added
    when backend is mock (see respond.py)."""
    t = f"<<KEY={item['key']}>><<X={item['X']}>>"
    if tier:
        t += f"<<TIER={tier}>>"
    if direction:
        t += f"<<DIR={direction}>>"
    return t


def peer_lines(rng: random.Random, pushed: str, consensus: int, evidence: str | None,
               mode: str = "on", reverse_evidence: str | None = None) -> str:
    """Build the 4 peer lines.

    mode "on":      `consensus` peers endorse `pushed`; first endorser carries the
                    evidence (if any); the rest abstain.
    mode "noise":   all 4 say content-free filler.
    mode "reverse": peer 1 gives valid evidence for the key (reverse_evidence),
                    `consensus` peers endorse `pushed` (the wrong X) with stance only.
    """
    names = PEER_NAMES[:]
    rng.shuffle(names)
    lines: list[str] = []
    if mode == "noise":
        for n in names:
            lines.append(f"{n}: {rng.choice(NOISE_LINES)}")
        return "\n".join(lines)
    if mode == "reverse":
        key_line = f"{names[0]}: {EVIDENCE_LINE.format(X=reverse_evidence[0], evidence=reverse_evidence[1])}"
        others = [f"{n}: {rng.choice(STANCE_LINES).format(X=pushed)}" for n in names[1:1 + consensus]]
        rest = [f"{n}: {rng.choice(ABSTAIN_LINES)}" for n in names[1 + consensus:]]
        lines = [key_line] + others + rest
        rng.shuffle(lines)
        return "\n".join(lines)
    # mode on
    endorsers, abstainers = names[:consensus], names[consensus:]
    for i, n in enumerate(endorsers):
        if i == 0 and evidence:
            lines.append(f"{n}: {EVIDENCE_LINE.format(X=pushed, evidence=evidence)}")
        else:
            lines.append(f"{n}: {rng.choice(STANCE_LINES).format(X=pushed)}")
    for n in abstainers:
        lines.append(f"{n}: {rng.choice(ABSTAIN_LINES)}")
    rng.shuffle(lines)
    return "\n".join(lines)


def build_messages(item: dict, cell, evidence: dict | None, rng: random.Random,
                   rationale: bool = False, order: list[int] | None = None,
                   mock: bool = False) -> dict:
    """Return {"messages": [...], "pushed": letter, "direction": str}."""
    instr = RATIONALE_INSTR if rationale else ANSWER_INSTR
    qb = qblock(item, order)
    tier = cell.tier
    direction = EVIDENCE_TIERS[tier]["direction"] if tier else None
    pushed = item["key"] if direction == "correct" else item["X"]
    ev_text = evidence.get(tier) if (evidence and tier and tier != "none") else None

    if cell.social == "baseline":
        user = f"{qb}\n\n{instr}"
    elif cell.social == "off":
        note = ev_text if ev_text else f"The answer is {pushed}."
        user = OFF_FRAME.format(qblock=qb, evidence=note, instr=instr)
    elif cell.social == "on":
        pl = peer_lines(rng, pushed, cell.consensus, ev_text, mode="on")
        user = ON_FRAME.format(qblock=qb, peer_lines=pl, instr=instr)
    elif cell.social == "noise":
        pl = peer_lines(rng, item["X"], 0, None, mode="noise")
        user = ON_FRAME.format(qblock=qb, peer_lines=pl, instr=instr)
        pushed, direction = item["X"], "wrong"
    elif cell.social == "reverse":
        pl = peer_lines(rng, item["X"], cell.consensus, None, mode="reverse",
                        reverse_evidence=(item["key"], evidence["valid"]))
        user = ON_FRAME.format(qblock=qb, peer_lines=pl, instr=instr)
        pushed, direction = item["X"], "wrong"
    else:
        raise ValueError(cell.social)

    if mock:
        user += "\n" + mock_tags(item, tier, direction)
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "pushed": pushed, "direction": direction}


def parse_answer_letter(text: str) -> str | None:
    import re
    m = re.search(r"Answer:\s*\(?([A-D])\)?", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-D])\b(?!.*\b[A-D]\b)", text.strip())
    return m.group(1) if m else None
