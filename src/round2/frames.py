"""Prompt frames of the baseline run: peer / expert / system sources, headcount 1 or 4, and the controls.

Letters are rendered per option order: the claim ("I think it's B") uses the displayed letter of the
pushed option; evidence texts are letter-free (round2 evidence prompts), so they read the same in
every order. Evidence for the pair (item, X) comes from data/round2/evidence.jsonl.
"""
from __future__ import annotations

import random

from config import QUESTION_BLOCK
from round2.common2 import maps
from round2.config2 import (ABSTAIN_LINES, COT_INSTR, EVIDENCE_LINE, EXPERT_NAME, EXPERTS, LETTER_INSTR,
                            NOISE_LINES, ON_FRAME, PEER_NAMES, STANCE_LINES, SYSTEM, SYSTEM_NOTE,
                            SYSTEM_NOTE_EVIDENCE, UNITY_PREFIX, Cell2)


def expert_label(subject: str) -> str:
    surname, field = EXPERTS.get(subject, EXPERTS["other"])
    return EXPERT_NAME.format(surname=surname, field=field)


def _unity(line: str, family: str) -> str:
    body = line if line.startswith("I'") or line.startswith("I ") else line[0].lower() + line[1:]
    return UNITY_PREFIX.format(family=family) + body


def _group(rng: random.Random, P: str, k: int, ev: str | None, lead: str | None = None,
           family: str | None = None) -> str:
    """4 members; k endorse P (the first carries the evidence, if any), the rest abstain.
    lead: replace the first endorser's name with a credentialed label (expert source)."""
    names = PEER_NAMES[:]
    rng.shuffle(names)
    lines = []
    for i, n in enumerate(names):
        if i < k:
            text = EVIDENCE_LINE.format(X=P, evidence=ev) if (i == 0 and ev) else rng.choice(STANCE_LINES).format(X=P)
            if family:
                text = _unity(text, family)
            lines.append(f"{lead if (i == 0 and lead) else n}: {text}")
        else:
            lines.append(f"{n}: {rng.choice(ABSTAIN_LINES)}")
    rng.shuffle(lines)
    return "\n".join(lines)


def build(item: dict, cell: Cell2, ev: dict | None, order: list[int] | None, probe: str,
          family: str, mock: bool = False) -> dict:
    """-> {"messages", "pushed" (canonical letter or None), "d2c" (display->canonical)}"""
    rng = random.Random(f"r2-{cell.name}-{item['id']}-{order}")
    txt, d2c, c2d = maps(item, order)
    qb = QUESTION_BLOCK.format(question=item["question"], options=txt)
    instr = COT_INSTR if probe == "cot" else LETTER_INSTR
    tier = cell.tier
    pushed = item["key"] if tier == "valid" else item["X"]
    P = c2d[pushed]
    ev_text = ev.get(tier) if (ev and tier and tier != "none") else None
    system = SYSTEM

    if cell.source == "baseline":
        user = f"{qb}\n\n{instr}"
        pushed = None
    elif cell.source == "system":
        note = (SYSTEM_NOTE_EVIDENCE if ev_text else SYSTEM_NOTE).format(X=P, evidence=ev_text)
        system = f"{SYSTEM}\n\n{note}"
        user = f"{qb}\n\n{instr}"
    elif cell.source in ("peer", "expert", "unity"):
        lead = expert_label(item["subject"]) if cell.source == "expert" else None
        fam = family if cell.source == "unity" else None
        user = ON_FRAME.format(qblock=qb, peer_lines=_group(rng, P, cell.headcount, ev_text, lead, fam), instr=instr)
    elif cell.source == "noise":
        names = PEER_NAMES[:]
        rng.shuffle(names)
        user = ON_FRAME.format(qblock=qb, peer_lines="\n".join(f"{n}: {rng.choice(NOISE_LINES)}" for n in names),
                               instr=instr)
        pushed = item["X"]
    elif cell.source == "reverse":
        names = PEER_NAMES[:]
        rng.shuffle(names)
        K = c2d[item["key"]]
        lines = [f"{names[0]}: {EVIDENCE_LINE.format(X=K, evidence=ev['valid'])}"]
        lines += [f"{n}: {rng.choice(STANCE_LINES).format(X=c2d[item['X']])}" for n in names[1:]]
        rng.shuffle(lines)
        user = ON_FRAME.format(qblock=qb, peer_lines="\n".join(lines), instr=instr)
        pushed = item["X"]
    else:
        raise ValueError(cell.source)

    if mock:
        direction = "correct" if tier == "valid" and cell.source != "reverse" else ("wrong" if tier else None)
        user += f"\n<<KEY={c2d[item['key']]}>><<X={c2d[item['X']]}>>"
        if tier and cell.source != "reverse":
            user += f"<<TIER={tier}>><<DIR={direction}>>"
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "pushed": pushed, "d2c": d2c}
