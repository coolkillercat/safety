# Data

All files are JSON Lines (one record per line). Item ids are `mmlupro-<question_id>` from the MMLU-Pro test split.

| file | records | what it holds |
| --- | ---: | --- |
| `round2/pool.jsonl` | 3,000 | the screening pool: MMLU-Pro test items reduced to four options (the key and three distractors drawn with a fixed seed), subject-stratified; `options` in canonical order, `key` the correct letter |
| `round2/screen_qwen3.5-27b.jsonl` | 21,000 | Qwen3.5-27B screening outputs: 5 letter-only probes (`mode = letter`, letter probabilities `probs`) and 2 chain-of-thought probes (`mode = cot`) per pool item, answers mapped back to the canonical letter |
| `round2/items_qwen3.5-27b.jsonl` | 400 | the items of the study: 300 `known` (7 of 7 probes correct) and 100 `boundary` (3 to 5 of 7); `X` is the pushed option (highest mean letter probability among the wrong options), `n_correct` the number of correct probes |
| `round2/orders_qwen3.5-27b.jsonl` | 3,400 | option orders: `tag = screen` (the 5 screening orders of each pool item) and `tag = probe` (the 3 orders shown in the experiments for each study item); each order is a permutation of the canonical options |
| `round2/pairs.jsonl` | 1,645 | (item, pushed option) pairs for which advice texts were written; `models` lists the screened models whose strata contain the pair (the study uses the pairs of Qwen3.5-27B) |
| `round2/evidence.jsonl` | 1,317 | advice texts per pair `id = <item>\|<X>`: `decept` (false rationale arguing for X) and `valid` (true rationale explaining the key), both by Claude Opus 5.5, written without option letters; `weak` and `halluc` are further texts not used in the paper; `gen` records the generator of each text; `refused` / `hedged` / `lettered` / `stripped` are quality flags |
| `round2/evidence_targets.jsonl` | 5,286 | which option each advice text argues for, as labelled by an automatic judge (informational) |
| `round2/own_wrong_texts.jsonl` | 1,186 | Experiment 2's planted wrong solutions: a worked solution for a wrong option `Y` (not the advised option), written by gemma-3-27b in the style of the model's own first answer; `ok = false` marks the 59 texts that do not end on `Answer: Y` (kept) |
| `round2/key_audit_v2.jsonl` | 827 | answer-key audit by Claude Sonnet 5.5: `key_valid` (is the keyed option unambiguously correct?) and `rationale_valid` (does the true rationale avoid errors and concessions?), with the judge's replies |

The unrelated paragraph of template C is built at run time from the sentences in `src/round2/irrelevant.py`; the filler
of Experiment 2 from the sentences in `src/round2/attribution.py`.
