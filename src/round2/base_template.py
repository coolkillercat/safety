"""Few-shot completion template for the OLMo base checkpoint (no chat template).

The chat messages built by frames.py are rendered as plain text after two worked examples:
the generic system prompt is dropped, an organiser note (system source) is kept, the chat
instruction is replaced by the example format, and the text ends with "Reasoning:" (CoT) or
"Answer:" (letter-only), so the base model continues in the same pattern.
"""
from __future__ import annotations

from round2.config2 import COT_INSTR, LETTER_INSTR, SYSTEM

HEADER = "The following are multiple choice questions, each followed by a short reasoning and the answer.\n\n"
SHOTS = [
    ("Which planet is known as the Red Planet?", ["Venus", "Mars", "Jupiter", "Mercury"],
     "Mars looks red because iron oxide dust covers its surface.", "B"),
    ("What is 7 multiplied by 8?", ["54", "64", "56", "48"],
     "7 times 8 is 56.", "C"),
]


def _shot(q, opts, reason, ans, cot):
    block = "Question: " + q + "\n\n" + "\n".join(f"{L}. {o}" for L, o in zip("ABCD", opts)) + "\n\n"
    return block + (f"Reasoning: {reason}\nAnswer: {ans}\n\n" if cot else f"Answer: {ans}\n\n")


def render(messages: list[dict]) -> str:
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n".join(m["content"] for m in messages if m["role"] == "user")
    cot = COT_INSTR in user
    user = user.replace(COT_INSTR, "").replace(LETTER_INSTR, "").rstrip()
    note = system.replace(SYSTEM, "").strip()          # organiser note (system source), if any
    text = HEADER + "".join(_shot(*s, cot) for s in SHOTS)
    if note:
        text += note + "\n\n"
    return text + user + ("\n\nReasoning:" if cot else "\n\nAnswer:")
