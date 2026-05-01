# LLM-decision analysis: `ga_2p_winner`

- games: **1** (truncated: 0)
- mean rounds: **48.0**, mean transfer rate: **63.7** $/round
- winners: `LLM_p1`=1

## Decisions

- total decisions logged: **26**
- prefilter PASSes: `cant_afford`=8, `cash_floor`=2 (total 10)
- LLM calls: **16**, buy rate: **1.000**
- parse-path distribution: `last_token_fallback`=16

## Buy rate by colour group

| group | n | n_buy | buy_rate |
|-------|--:|------:|---------:|
| Railroads | 4 | 4 | 1.000 |
| Utilities | 2 | 2 | 1.000 |
| Red | 2 | 2 | 1.000 |
| Pink | 2 | 2 | 1.000 |
| Brown | 2 | 2 | 1.000 |
| Orange | 1 | 1 | 1.000 |
| Green | 1 | 1 | 1.000 |
| Indigo | 1 | 1 | 1.000 |
| Lightblue | 1 | 1 | 1.000 |

**Buy rate by cash bucket**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| 500-1000 | 8 | 8 | 1.000 |
| 1000-1500 | 5 | 5 | 1.000 |
| >=1500 | 3 | 3 | 1.000 |

**Buy rate by monopoly opportunity**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| fresh | 9 | 9 | 1.000 |
| partial_self | 4 | 4 | 1.000 |
| opponent_dominates | 3 | 3 | 1.000 |

## Surprising decisions (qualitative excerpts)

### PASS while flush (cash >= 1000)

_None._

### BUY at cash floor (cash < 300)

_None._
