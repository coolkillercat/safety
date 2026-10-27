# Talking a language model out of a correct answer: code and data

Code, items, generated advice texts, stored model outputs and result tables for the paper
**"Position Matters More Than Channel: Talking a Language Model Out of a Correct Answer"** (Xinyi Ni, 2026),
[arXiv:2610.34809](https://arxiv.org/abs/2610.34809). To cite it, see `CITATION.cff`.

The subject model is Qwen3.5-27B on four-option MMLU-Pro items. Experiment 1 moves one advice paragraph between three
positions in the prompt (end of the system message, top of the user message, after the options) and varies who signs
it, how certain it sounds and whether a rationale follows. Experiment 2 varies what stands before the advice: nothing,
the model's own earlier answer, the same answer quoted as someone else's, its conclusion only, or two controls.

## Reproduce the paper's tables and figures (CPU only, about five minutes)

```bash
pip install -r requirements.txt
bash scripts/run_analyses.sh
```

This rebuilds every table in `results/round2/` (with its `.schema.json` sidecar) from the stored model outputs, checks
them with `scripts/check_schema.py`, and redraws the paper's Figures 1, 3 and 4 into `figures/`. The tables in the
repository were produced by exactly this script.

`bash tests/smoke_dry_run.sh` rebuilds every prompt of both experiments with the mock backend (no model, no writes) and
confirms that every scheduled output is present.

## Layout

| path | what it holds |
| --- | --- |
| `src/round2/crossover.py` | Experiment 1 (placement, speaker, certainty, rationale; templates A, B, C) |
| `src/round2/attribution.py` | Experiment 2 (prior packages); it reuses rows of `branched.py` and `commitment.py` |
| `src/round2/branched.py`, `src/round2/commitment.py` | the two runs whose rows Experiment 2 reuses (own full answer; no prior turn), plus conditions not analysed in the paper |
| `src/round2/screen.py`, `strata.py`, `order_dups.py` | item screening, known / boundary strata, the pushed option X, option orders |
| `src/round2/key_audit.py`, `audit_sets.py` | answer-key audit and the audit-passing item sets (`_keyvalid` in table names) |
| `src/round2/respond2.py`, `frames.py` | the baseline run that records each trial's first answer |
| `src/config.py`, `src/round2/config2.py`, `templates_B.py`, `templates_C.py`, `irrelevant.py` | prompt text (template A in `config.py`), the two independently written templates, the unrelated paragraph |
| `src/make_evidence.py`, `src/round2/halluc_fix.py` | generation and checking of the advice texts |
| `src/round2/common2.py` | answer parsing (`parse_answer_strict`, `parse_final`), option orders, model loading |
| `src/round2/schemas.py`, `scripts/write_schemas.py`, `scripts/check_schema.py` | table schemas and the table checker |
| `scripts/make_figures.py` | the paper's figures |
| `data/round2/` | items, screening outputs, option orders, advice texts, the planted wrong solutions, the key audit (see `data/README.md`) |
| `results/round2/responses/qwen3.5-27b/` | every model output the paper uses, one JSON line per output (see `results/README.md`) |
| `results/round2/*.csv` | result tables, each with a `.schema.json` sidecar (see `results/README.md`) |

## Where each part of the paper comes from

| paper | code | tables |
| --- | --- | --- |
| Items, screening, pushed option (Section 3.1, Appendix D) | `screen.py`, `strata.py`, `order_dups.py`, `scripts/option_reduction_check.py` | `results/round2/order_dups.md`, `option_reduction_check.md` |
| Answer-key audit | `key_audit.py`, `audit_sets.py` | `data/round2/key_audit_v2.jsonl`; `_keyvalid` tables |
| Experiment 1 (Sections 3.3, 4.1, 4.2; Figures 1a, 1b, 3; Tables 3, 5–8) | `crossover.py` | `crossover_cells_final.csv`, `crossover_placement_contrasts_final.csv`, `crossover_C_contrasts_final.csv`, `crossover_templateB_protocols_final.csv` |
| Experiment 2 (Sections 3.4, 4.3; Figures 1c, 1d, 4; Tables 9–11) | `attribution.py` | `attribution_qwen3.5-27b_final.csv`, `attribution_fouram_final.csv`, `attribution_package_diffs_final.csv`, `attribution_contrasts_final.csv` |
| Reading the final answer (Section 3.5, Appendix E) | `common2.parse_final`, `reparse_final.py` | `_final` (primary) and unsuffixed (strict) tables |
| Package lengths (Appendix B) | `scripts/package_length_ratios.py` | `results/round2/package_length_ratios.md` |

## Regenerating the model outputs (GPU; not needed to check the paper)

The outputs were generated with `Qwen/Qwen3.5-27B` served by vLLM 0.32.0 (transformers 5.19.0, torch 2.13.0), greedy
decoding, thinking mode off, two GPUs with tensor parallelism (template C: four). Every stored row records the prompt
hash, option-order scheme, token budget and model revision, and a run refuses to resume rows made under a different
protocol. In order:

```bash
export PYTHONPATH=src
python -m round2.screen --model qwen3.5-27b                 # 3,000-item pool, 7 probes per item
python -m round2.strata --models qwen3.5-27b                # known / boundary items and the pushed option X
python src/make_evidence.py --prompt-set round2 ...         # rationales (Claude Opus 5.5; needs ANTHROPIC_API_KEY; see --help)
python -m round2.key_audit                                  # key audit (Claude Sonnet 5.5; see --help)
python -m round2.respond2 --model qwen3.5-27b --cells baseline
python -m round2.crossover run --cells b2a --pair-with phaseB   # template A, system / after-options
python -m round2.crossover run --cells b2c --pair-with phaseB   # template A, before-question and the professor; template B
python -m round2.crossover run --cells p15 --max-tokens 2048 --shard 0/4   # template C, shards 0/4 ... 3/4
python -m round2.crossover run --cells p16 --max-tokens 2048    # template B at a fixed 2,048-token budget
python -m round2.branched run
python -m round2.commitment run
python -m round2.attribution gen-wrong                      # planted wrong solutions (gemma-3-27b)
python -m round2.attribution run
```

Any of these accepts `--backend mock --dry-run` to print the prompts without a model.

## Names used in the code and the tables

| in code and tables | in the paper |
| --- | --- |
| stratum `known` / `boundary` | known items (7 of 7 screening probes correct) / boundary items (3 to 5 of 7) |
| `X`, `pushed` | the pushed option |
| role `system` / `userpre` / `user` | placement: system / before-question / after-options |
| speaker `organiser` / `peer` / `expert` | session organiser / another participant / named professor |
| tier `none` / `decept` / `valid` / `keybare` / `irrel` | bare note / false rationale / true rationale / bare note naming the key / unrelated paragraph |
| cell prefixes `x-`, `xb-`, `xb2-`, `xc-` | template A, template B (as run), template B at 2,048 tokens, template C |
| prior `none`, `own-full`, `own-conclusion`, `quoted-full`, `quoted-conclusion`, `own-wrong-full`, `filler` | the prior packages of Experiment 2 |
| `fourarm_diff` | package effect |
| `baseline` cell, `pair_with = phaseB` | the run that records each trial's first answer; rows paired with its option orders |
| `lo`, `hi` | likely range: 95% bootstrap interval over items (10,000 draws, seed 20261023) |
| `bound_min`, `bound_max` | worst-case bounds over unfinished outputs |
| suffix `_final` / none | terminal-answer rule (primary) / strict parser (sensitivity) |
| suffix `_keyvalid`, `_ratvalid`, `_okboth` | audit-passing items (the auditor accepts the key; the paper's robustness set) / items whose true rationale passes / both |

## Licences

Code: MIT (`LICENSE`). Data, generated texts and result tables: CC BY 4.0 (`LICENSE-DATA`). The question texts come from
MMLU-Pro (TIGER-Lab, MIT licence). Model outputs are those of Qwen3.5-27B, Claude Opus 5.5 / Sonnet 5.5 and
gemma-3-27b / gemma-3-4b and are subject to those models' terms.
