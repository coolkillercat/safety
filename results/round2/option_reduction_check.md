# Option-reduction check against the original MMLU-Pro records

Items used in any `items*.jsonl`: 1211 (anchor Qwen3.5-27B: 400); matched to an original record: 1211.
Original test split: 12032 records; records with any empty / N/A option string: 0; with such a string *before* the key (the only configuration under which the pre-filter shifts `answer_index`): 0.

- (1) stored key letter points at the original key text: **1211 of 1211** (0 mismatches).
- (2) used items with an empty / N/A option before the key: **0**.
- (3) original `answer` letter disagrees with `answer_index`: 0 items.
- (4) stored options whose meaning depends on the removed options (regex: all/none/both/either ... above/of these; 'A and B'): 16 option strings in 12 items (3 anchor items).

## Stored options whose meaning may depend on the removed options

| item | anchor | option text |
| --- | --- | --- |
| mmlupro-2524 |  | None of the above |
| mmlupro-9736 | yes | None of these |
| mmlupro-9727 |  | Neither of these |
| mmlupro-9727 |  | Both of these |
| mmlupro-9214 |  | Neither of these |
| mmlupro-9214 |  | Both of these |
| mmlupro-10412 |  | All of the above |
| mmlupro-10389 |  | None of the above |
| mmlupro-7937 |  | None of the above |
| mmlupro-11134 | yes | all of the above |
| mmlupro-10414 | yes | All of the above (I, II, III) |
| mmlupro-10552 |  | None of the above |
| mmlupro-3972 |  | b and c are Hermitan |
| mmlupro-7186 |  | The b and c coefficients significant and the a and d coefficients insignificant |
| mmlupro-7186 |  | The a and c coefficients significant |
| mmlupro-7186 |  | The b and d coefficients significant |

Files read: items.jsonl, items_gemma3-12b.jsonl, items_gemma3-12b.pre-v3.jsonl, items_gemma3-27b.jsonl, items_gemma3-27b.pre-v3.jsonl, items_gemma4-31b.jsonl, items_gemma4-31b.pre-v3.jsonl, items_gemma4-31b.v1.jsonl, items_olmo3-32b-base.jsonl, items_olmo3-32b-base.pre-v3.jsonl, items_olmo3.1-32b-dpo.jsonl, items_olmo3.1-32b-dpo.pre-v3.jsonl, items_olmo3.1-32b-instruct.jsonl, items_olmo3.1-32b-instruct.pre-v3.jsonl, items_olmo3.1-32b-sft.jsonl, items_olmo3.1-32b-sft.pre-v3.jsonl, items_olmo3.1-32b-sft.v1.jsonl, items_qwen3.5-27b.jsonl, items_shared.jsonl.
