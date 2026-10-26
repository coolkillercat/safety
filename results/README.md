# Results

## Model outputs: `round2/responses/qwen3.5-27b/`

One JSON line per output, with the full prompt (`prompt`), the output text (`text`), the answer under the strict parser
(`parsed_strict`, or `a1` in branched / commitment rows) and under the terminal-answer rule (`parsed_final`,
`unresolved`), the item, its stratum, key and pushed option, the option order, and the protocol fields (`max_tokens`,
`order_scheme`, `prompt_sha`, `model_rev`, `finish_reason`) where recorded.

| folder | rows of |
| --- | --- |
| `cot/baseline.jsonl` | the baseline run: each (item, order) answered under the plain prompt; it defines the first answer of Experiment 2 and the option orders paired with it |
| `crossover/` | Experiment 1. `x-<placement>-<speaker>-<certainty>-<tier>.jsonl` (template A), `xb-*` (template B as run, with `*.cap768.jsonl` holding the rows first generated at 768 tokens), `xb2-*` (template B at a fixed 2,048 tokens), `xc-*` (template C), `*-noadv-none-none-none` (no-advice controls) |
| `branched/` | the model's own recorded answer, then advice (`b-<advice>`); Experiment 2's `own-full` package |
| `commitment/` | no prior turn, own answer control, and two conditions not analysed in the paper (`fake`, `hidden`); Experiment 2's `none` package |
| `attribution/` | Experiment 2's remaining packages (`a_<prior>_<advice>`, `_control` = deleted-advice control) |

## Tables: `round2/*.csv`

Every table has a sidecar `<table>.csv.schema.json` stating its population, unit, weights, handling of unfinished
outputs and contrast support. `scripts/check_schema.py` checks the sidecars and the tables' internal consistency.

File-name suffixes: `_final` = terminal-answer rule (the paper's primary outcome), no suffix = strict parser
(sensitivity); `_keyvalid`, `_ratvalid`, `_okboth` = items whose key, true rationale, or both pass the audit.

| family | contents |
| --- | --- |
| `crossover_cells` | error rate of every Experiment 1 cell, with denominators (scheduled, eligible, parsed), likely range, and unfinished outputs by kind |
| `crossover_placement_contrasts` | placement, channel, speaker and certainty effects and the argument's extra placement effect (template A; template B at order 0) |
| `crossover_C_contrasts` | the same contrasts under template C, known and boundary items |
| `crossover_templateB_protocols` | template B's bare-note effects under each run protocol, and template A on the same items |
| `crossover_contrasts`, `crossover_common`, `crossover_transitions`, `crossover_missing` | factorial contrasts of the system / after-options cells, transitions relative to the baseline answer, unfinished outputs |
| `attribution_qwen3.5-27b` | error rate of every Experiment 2 cell by stratum and first-answer correctness |
| `attribution_fouram` | package effects (`fourarm_diff`): the advice effect under a package minus the advice effect with no prior turn |
| `attribution_package_diffs` | direct comparisons of two packages on their joint support |
| `attribution_contrasts` | package-versus-package differences in error, with worst-case bounds |
| `commitment_*`, `fu_*` | tables of the two runs Experiment 2 reuses |

Columns `lo` / `hi` are the likely range (95% bootstrap interval over items, 10,000 draws, seed 20261023);
`bound_min` / `bound_max` are worst-case bounds (every unfinished output counted against the estimate, then for it).

## Check reports: `round2/*.md`

`order_dups.md` (repeated option orders), `option_reduction_check.md` (the four-option reduction against the original
MMLU-Pro records), `package_length_ratios.md` (Experiment 2 package lengths) and `prefix_divergence.md` (template B
rows regenerated at a larger budget), each written by the script named inside it.
