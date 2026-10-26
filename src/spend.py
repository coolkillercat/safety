"""API spend log and hard budget caps.

results/api_spend.csv holds two kinds of rows:
  kind=price  the per-MTok prices used, with where and when they were verified
  kind=spend  one row per API batch: tokens, USD at those prices, running total per provider/step
Budget checks read the file, so caps hold across separate jobs (run API steps one at a time).
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

from config import RESULTS

# DIVERSITY_SPEND lets smoke tests in another root count against the same caps.
SPEND = Path(os.environ.get("DIVERSITY_SPEND", RESULTS / "api_spend.csv"))
FIELDS = ["time", "kind", "provider", "model", "step", "in_tokens", "out_tokens",
          "usd_per_mtok_in", "usd_per_mtok_out", "usd", "provider_total_usd", "note"]

# Verified 2026-10-23 on the providers' official pricing pages (peak rates where they vary).
PRICES = {
    ("anthropic", "claude-sonnet-5-5"): (2.00, 10.00, "platform.claude.com/docs/en/about-claude/pricing"),
    ("anthropic", "claude-opus-5-5"): (4.00, 20.00, "platform.claude.com/docs/en/about-claude/pricing"),
    ("anthropic", "claude-haiku-5-5"): (0.10, 0.50, "platform.claude.com/docs/en/about-claude/pricing (<=100k-token prompts)"),
    ("deepseek", "deepseek-v4-pro"): (1.32, 3.96, "api-docs.deepseek.com/quick_start/pricing (peak; off-peak half)"),
    ("deepseek", "deepseek-flash"): (0.30, 1.20, "api-docs.deepseek.com/quick_start/pricing (peak; off-peak half)"),
    # Not on ai.google.dev/gemini-api/docs/pricing any more (lists 3.6/3.8 Flash at 0.75/3.75);
    # third-party listings give 1.50/9.00. Using the higher figure until confirmed in API-model.
    ("gemini", "gemini-3.5-flash"): (1.50, 9.00, "third-party listings; absent from Google's page 2026-10-23"),
}
# Caps (USD): stop and report at the provider cap, never above the hard cap.
PROVIDER_CAPS = {"anthropic": 85.0, "deepseek": 10.0, "gemini": 25.0}
HARD_CAPS = {"anthropic": 100.0}
STEP_CAPS = {"sonnet_run": 40.0, "keyaudit_r2": 4.8,   # : <= $5 (checked before each ~$0.2 batch)
             "keyaudit_split_r2": 2.1, "halluc_audit_r2": 0.8,   # split key audit: <= $3 together (split overran its estimate)
             "keyaudit_both_r2": 2.0}   # <= $4


class BudgetExceeded(RuntimeError):
    pass


def _rows(path: Path = SPEND) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _append(row: dict, path: Path = SPEND):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def write_prices(path: Path = SPEND):
    """Record the price table once (idempotent)."""
    have = {(r["provider"], r["model"]) for r in _rows(path) if r["kind"] == "price"}
    for (prov, model), (pin, pout, src) in PRICES.items():
        if (prov, model) not in have:
            _append({"time": time.strftime("%Y-%m-%d"), "kind": "price", "provider": prov, "model": model,
                     "usd_per_mtok_in": pin, "usd_per_mtok_out": pout, "note": f"verified 2026-10-23: {src}"}, path)


def totals(path: Path = SPEND) -> tuple[dict, dict]:
    by_prov, by_step = {}, {}
    for r in _rows(path):
        if r["kind"] != "spend":
            continue
        by_prov[r["provider"]] = by_prov.get(r["provider"], 0.0) + float(r["usd"])
        by_step[r["step"]] = by_step.get(r["step"], 0.0) + float(r["usd"])
    return by_prov, by_step


def check(provider: str, step: str, path: Path = SPEND):
    """Raise BudgetExceeded before a batch if the provider or step cap is already reached."""
    by_prov, by_step = totals(path)
    if by_prov.get(provider, 0.0) >= PROVIDER_CAPS.get(provider, float("inf")):
        raise BudgetExceeded(f"{provider} spend ${by_prov[provider]:.2f} >= cap ${PROVIDER_CAPS[provider]:.0f}")
    if by_step.get(step, 0.0) >= STEP_CAPS.get(step, float("inf")):
        raise BudgetExceeded(f"step {step} spend ${by_step[step]:.2f} >= cap ${STEP_CAPS[step]:.0f}")


def log(provider: str, model: str, step: str, in_tokens: int, out_tokens: int, note: str = "",
        path: Path = SPEND) -> float:
    if not in_tokens and not out_tokens:
        return 0.0
    pin, pout, _ = PRICES[(provider, model)]
    usd = in_tokens / 1e6 * pin + out_tokens / 1e6 * pout
    total = totals(path)[0].get(provider, 0.0) + usd
    _append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "spend", "provider": provider, "model": model,
             "step": step, "in_tokens": in_tokens, "out_tokens": out_tokens, "usd_per_mtok_in": pin,
             "usd_per_mtok_out": pout, "usd": f"{usd:.4f}", "provider_total_usd": f"{total:.4f}", "note": note},
            path)
    return usd


if __name__ == "__main__":
    write_prices()
    p, s = totals()
    print("prices recorded ->", SPEND, "| spend by provider:", p, "| by step:", s)
