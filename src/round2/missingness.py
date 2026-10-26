"""Why an eligible CoT output has no strict answer.

  length_cap          generation stopped at the token cap (finish_reason == "length"; baseline run records predate
                      finish_reason, so their `likely_truncated` flag stands in)
  inline_answer       an answer letter is stated, but not on a final "Answer: X" line (legacy parser finds one)
  explicit_rejection  the output declines all four options ("none of the options", "Answer: None",...)
  other               anything else
Checked in that order. Used by the crossover, branched, commitment and replay analyses.
"""
from __future__ import annotations

import re

from round2.common2 import parse_answer_legacy

CATS = ("unresolved", "length_cap", "inline_answer", "explicit_rejection", "other")
_REJECT = re.compile(
    r"none of the (?:given |provided |listed |above |four |offered )?(?:options|choices|answers|answer choices)"
    r"|no (?:correct|valid|matching) (?:option|answer|choice)"
    r"|(?:is|are) not (?:among|listed among|one of|in) the (?:given |provided |listed )?(?:options|choices)"
    r"|(?:cannot|can't|can not) (?:select|choose|pick) (?:any|one)"
    r"|all (?:four |of the )?(?:options|choices) are (?:incorrect|wrong|flawed)"
    r"|answer:\s*\**\s*(?:none|n/?a|e)\b", re.I)


def category(text: str | None, finish_reason: str | None = None, likely_truncated: bool | None = None,
             unresolved: bool = False) -> str:
    """unresolved: an answer line followed by more substantive text (terminal-answer rule only)."""
    if unresolved:
        return "unresolved"
    if finish_reason == "length" or (finish_reason is None and likely_truncated):
        return "length_cap"
    if parse_answer_legacy(text or ""):
        return "inline_answer"
    if _REJECT.search(text or ""):
        return "explicit_rejection"
    return "other"
