# LLM-decision analysis: `default`

- games: **1** (truncated: 0)
- mean rounds: **55.0**, mean transfer rate: **49.0** $/round
- winners: `LLM_p1`=1

## Decisions

- total decisions logged: **40**
- prefilter PASSes: `cash_floor`=8, `cant_afford`=5 (total 13)
- LLM calls: **27**, buy rate: **1.000**
- parse-path distribution: `last_token_fallback`=27

## Buy rate by colour group

| group | n | n_buy | buy_rate |
|-------|--:|------:|---------:|
| Railroads | 4 | 4 | 1.000 |
| Lightblue | 3 | 3 | 1.000 |
| Pink | 3 | 3 | 1.000 |
| Orange | 3 | 3 | 1.000 |
| Red | 3 | 3 | 1.000 |
| Yellow | 3 | 3 | 1.000 |
| Utilities | 2 | 2 | 1.000 |
| Indigo | 2 | 2 | 1.000 |
| Brown | 2 | 2 | 1.000 |
| Green | 2 | 2 | 1.000 |

**Buy rate by cash bucket**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| 200-500 | 2 | 2 | 1.000 |
| 500-1000 | 16 | 16 | 1.000 |
| 1000-1500 | 7 | 7 | 1.000 |
| >=1500 | 2 | 2 | 1.000 |

**Buy rate by monopoly opportunity**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| fresh | 11 | 11 | 1.000 |
| partial_self | 8 | 8 | 1.000 |
| opponent_dominates | 8 | 8 | 1.000 |

## Surprising decisions (qualitative excerpts)

### PASS while flush (cash >= 1000)

_None._

### BUY at cash floor (cash < 300)

_None._
