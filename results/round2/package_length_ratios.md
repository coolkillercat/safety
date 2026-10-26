# Length of the replay-experiment prior-turn packages relative to the baseline answer (descriptive)

Computed from the stored records only (`scripts/package_length_ratios.py`); no generation. Ratio = package text / the same (item, order)'s baseline chain of thought. Quartiles are inclusive sample quartiles.

| package | n | unit | Q1 | median | Q3 | min | max |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| filler (deliberation sentences, character-matched) | 395 | characters | 1.02 | 1.03 | 1.05 | 1.00 | 1.14 |
| filler (deliberation sentences, character-matched) | 395 | words | 1.23 | 1.33 | 1.40 | 0.96 | 1.67 |

filler: 395 records with a stored prompt; 791 records without a stored prompt were skipped.

| package | n | unit | Q1 | median | Q3 | min | max |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| own-wrong (Gemma-3-27B worked solution for Y) | 1186 | characters | 1.74 | 1.99 | 2.39 | 0.56 | 10.84 |
| own-wrong (Gemma-3-27B worked solution for Y) | 1186 | words | 1.81 | 2.07 | 2.48 | 0.55 | 11.55 |

own-wrong texts: 1186 written; 1127 end on `Answer: Y` (`ok`), 59 do not and are kept, flagged.
