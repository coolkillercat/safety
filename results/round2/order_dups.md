# Option-order duplicates under the legacy scheme

`order_for` drew each non-canonical order independently, so an item could see the same permutation twice. New runs use `distinct_orders` (without replacement); the crossover and branched runs keep the legacy orders so they pair with the baseline run.

## screening letter, 5 orders

Items 3000; with a repeated order 691 (23.0%).

| distinct orders | items |
| ---: | ---: |
| 5 | 2309 |
| 4 | 650 |
| 3 | 40 |
| 2 | 1 |

## screening CoT, 2 orders

Items 3000; with a repeated order 0 (0.0%).

| distinct orders | items |
| ---: | ---: |
| 2 | 3000 |

## probe, 3 orders (anchor items)

Items 400; with a repeated order 16 (4.0%).

| distinct orders | items |
| ---: | ---: |
| 3 | 384 |
| 2 | 16 |

## probe repeats by stratum (anchor)

- known: 14/300 items have two identical probe orders
- boundary: 2/100 items have two identical probe orders
