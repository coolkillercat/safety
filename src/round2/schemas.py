"""Every result table carries its schema (population, unit, weights, missing-outcome policy, contrast support) in a
sidecar `<table>.schema.json`, plus the column list and a one-line summary. Schemas are composed from a per-family base
and the filename suffixes (analysis set, parser). `scripts/write_schemas.py` stamps every CSV; `scripts/check_schema.py`
checks the structure and the semantic invariants (denominator monotonicity, F + U = final-parsed, contrast support no
larger than the smaller cell, one-order cells scheduled = items).
The `_final` tables (terminal-answer rule) are the primary outcome; the unsuffixed tables use the strict parser and are
the sensitivity analysis.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REQUIRED = ("population", "unit", "weights", "missing", "support")

ANCHOR_POP = "Qwen3.5-27B anchor items (known 300 = 7/7 screening-stable, boundary 100 = 3–5/7), 3 option orders"
ITEMW = "item-weighted: mean over an item's parsed orders, then mean over items; 95% item bootstrap (10,000, seed 20261023 reset per estimate)"

BOUNDS = ("bound_min/bound_max = deterministic missing-outcome bounds over the SCHEDULED eligible manifest (absent "
          "jobs stay in the denominator; positive-coefficient cells' missing outcomes filled correct/wrong, "
          "negative-coefficient the reverse) — an identification range for this finite sample, not an interval; "
          "bound_items/bound_pairs = its support")

FAMILIES = {   # filename prefix -> base schema
    "crossover_cells": dict(population=ANCHOR_POP + "; Experiment 1 crossover cells (templates A, B, B2, C)",
                            unit="cell × stratum", weights=ITEMW + "; error_trialweighted pooled",
                            missing="complete cases; missing_* by category; upper97_5_if_zero for zero-event cells; "
                                    "scheduled = jobs the design manifest (crossover.jobs_for) schedules for the cell "
                                    "(templates B / B2: order 0 only = items; A / C: 3 orders), scheduled ≥ eligible ≥ parsed",
                            support="cell-wise; template-B/B2 cells order 0 only"),
    "crossover_contrasts": dict(population=ANCHOR_POP + "; template A system / after-options cells: known (none, decept) and boundary (keybare, valid)",
                                unit="factorial contrast (±1/4 main effects, ±1/2 interactions) × tier",
                                weights=ITEMW, missing="complete cases",
                                support="(item, order) with all 8 cells of the tier parsed"),
    "crossover_common": dict(population=ANCHOR_POP + "; template A system / after-options known cells", unit="factorial contrast × tier, and decept − none",
                             weights=ITEMW, missing="complete cases",
                             support="(item, order) with all 16 known cells (none + decept) parsed; second block also "
                                     "requires all 3 orders complete per item"),
    "crossover_transitions": dict(population=ANCHOR_POP + "; crossover cells vs the baseline run",
                                  unit="cell: counts cc / cw / wc / ww", weights="trial counts",
                                  missing="pairs parsed in both; pairs_dropped = parsed cell rows with no baseline row under "
                                          "the SAME option permutation",
                                  support="(item, permutation) parsed in cell and the baseline run — join = 'perm' (on the "
                                          "actual permutation; distinct-order cells such as template C) or 'index' (on the "
                                          "order index, permutations asserted identical; legacy paired cells)"),
    "crossover_missing": dict(population=ANCHOR_POP + "; all crossover cells", unit="cell × stratum × missing category",
                              weights="counts of eligible unparsed outputs", missing="this table IS the missing outcomes",
                              support="eligible rows"),
    "crossover_placement": dict(population=ANCHOR_POP + "; template A known cells (three placements) + template B order 0",
                                unit="placement / speaker / certainty contrasts by family × tier", weights=ITEMW,
                                missing="diff/lo/hi complete cases; " + BOUNDS,
                                support="per family: the 18-cell common support, the organiser/peer cells involved "
                                        "(the 'assertive only' family: organiser/peer assertive cells — the matched "
                                        "reference for template C), or the 8 template cells at order 0 (see `family`)"),
    "crossover_templateB": dict(population="anchor known items, order 0 (300 scheduled)",
                                unit="template-B bare-note main effect × protocol (as run 768 / selectively "
                                "regenerated / fixed 2,048) and template A", weights=ITEMW,
                                missing="diff = complete cases; " + BOUNDS + " (all 300 scheduled items); B_minus_A on "
                                "items complete in all 8 A and 8 B cells", support="see missing"),
    "crossover_C": dict(population="anchor items, template C, known 300 × 3 and boundary 100 × 3",
                        unit="contrast × stratum", weights=ITEMW, missing="diff/lo/hi complete cases; " + BOUNDS,
                        support="(item, order) with all cells of the contrast parsed"),
    "fu_qwen": dict(population="anchor items with a parsed baseline answer a0 (branched follow-up)",
                    unit="advice × stratum: F = P(a1 ≠ key | a0 = key), U = P(a1 = key | a0 ≠ key)",
                    weights=ITEMW, missing="complete cases; missing_* by category; zero-event upper bounds",
                    support="F/U: rows of the arm; Δ vs reconsider: (item, order) parsed in both"),
    "fu_contrasts": dict(population="anchor items with parsed a0 (branched follow-up)", unit="arm − arm on F or U; within-arm net accuracy",
                         weights=ITEMW + "; net accuracy also trial-weighted", missing="complete cases",
                         support="(item, order) parsed in both arms; net accuracy on pairs complete in all five arms"),
    "fu_missing": dict(population="branched follow-up rows", unit="advice × stratum × missing category", weights="counts",
                       missing="this table IS the missing outcomes", support="eligible rows"),
    "commitment_qwen": dict(population="anchor items × 3 orders (commitment run); F on baseline-correct, U on baseline-wrong rows",
                            unit="prior × advice × stratum", weights="F_item/U_item item-weighted (" + ITEMW +
                            "); F_trial/U_trial pooled over rows",
                            missing="complete cases; denominators scheduled → n_evidence_eligible → n_baseline_parsed → "
                                    "n_final_parsed → {n_initially_correct (F) | n_initially_wrong (U)}: parallel groups "
                                    "that sum to n_final_parsed, not nested stages",
                            support="cell rows; *_minus_floor paired with the same prior's control on (item, order)"),
    "commitment_prior": dict(population="commitment run, baseline-correct rows", unit="F(prior) − F(none) per advice × stratum",
                             weights=ITEMW + " (diff); F_prior/F_none trial-weighted on the paired set",
                             missing="complete cases", support="(item, order) parsed in both"),
    "commitment_fouram": dict(population="commitment run rows with a parsed baseline answer",
                              unit="four-arm floor-adjusted contrast: (e_p,a − e_p,ctl) − (e_none,a − e_none,ctl)",
                              weights=ITEMW, missing="complete cases",
                              support="(item, order) with all four cells eligible and parsed"),
    "attribution_qwen": dict(population="replay run: anchor items × 3 orders with a parsed baseline answer",
                             unit="stratum × baseline correctness × prior × advice: error (F on correct, 1 − U on wrong)",
                             weights="err_item item-weighted (" + ITEMW + "); err_trial pooled",
                             missing="complete cases; missing_* by category; denominators scheduled (stratum items × 3) → "
                                     "n_baseline_parsed (stratum) → n_group (initially correct | wrong: parallel groups "
                                     "summing to n_baseline_parsed) → n_rows_present → n_eligible → n_final_parsed",
                             support="cell rows"),
    "attribution_fouram": dict(population="replay run rows with a parsed baseline answer",
                               unit="four-arm floor-adjusted contrast vs none; raw prior − none beside",
                               weights=ITEMW, missing="complete cases",
                               support="(item, order) with all four cells eligible and parsed"),
    "attribution_contrasts": dict(population="replay run, baseline-correct rows", unit="F(A) − F(B) per advice × stratum",
                                  weights=ITEMW, missing="diff/lo/hi complete cases; " + BOUNDS,
                                  support="(item, order) parsed in both"),
    "attribution_package_diffs": dict(population="replay run, baseline-correct rows",
                                      unit="direct difference of two packages' advice effects: (e_A,advice − e_A,control) − "
                                           "(e_B,advice − e_B,control) per advice × stratum; positive = A's effect larger",
                                      weights=ITEMW, missing="diff/lo/hi complete cases; " + BOUNDS,
                                      support="joint: (item, order) with all four cells eligible and parsed; a pair of "
                                              "packages with no joint support gets an explicit 'no joint support' row"),
}
SUFFIX = {"_keyvalid": "items whose key is judged valid (key audit, both axes judged)",
          "_ratvalid": "items whose valid rationale is judged sound",
          "_okboth": "items valid on both audit axes",
          "_final": "terminal-answer rule (parsed_final): an answer line followed by more text is unresolved — primary "
                    "outcome"}


def schema_for(path: Path) -> dict:
    name = path.stem
    fam = next((k for k in sorted(FAMILIES, key=len, reverse=True) if name.startswith(k)), None)
    if fam is None:
        raise KeyError(f"no schema family for {path.name}")
    s = dict(FAMILIES[fam])
    mods = [v for k, v in SUFFIX.items() if re.search(re.escape(k) + r"(_|$)", name)]
    s["population"] = s["population"] + ("; " + "; ".join(mods) if mods else "; all items (no audit exclusion)")
    s["parser"] = ("terminal-answer (parsed_final) — primary" if "_final" in name
                   else "strict last 'Answer: X' line — sensitivity")
    s["family"] = fam
    s["one_line"] = (f"Schema: {s['population']} | unit: {s['unit']} | weights: {s['weights'].split(';')[0]} | "
                     f"missing: {s['missing'].split(';')[0]} | support: {s['support']}")
    return s


def stamp(path: Path) -> Path:
    import pandas as pd
    s = schema_for(path)
    s["columns"] = list(pd.read_csv(path, nrows=0).columns)
    out = path.with_name(path.name + ".schema.json")
    out.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    return out
