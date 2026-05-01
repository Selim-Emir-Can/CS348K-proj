# Combined LLM-decision analysis

## default

# LLM-decision analysis: `default`

- games: **1** (truncated: 0)
- mean rounds: **131.0**, mean transfer rate: **54.6** $/round
- winners: `LLM_p0`=1

## Decisions

- total decisions logged: **60**
- prefilter PASSes: `cant_afford`=11, `cash_floor`=10 (total 21)
- LLM calls: **39**, buy rate: **0.667**
- LLM call latency: median 3276 ms, mean 3277 ms, max 3348 ms
- parse-path distribution: `first_answer_tag`=39

## Buy rate by colour group

| group | n | n_buy | buy_rate |
|-------|--:|------:|---------:|
| Pink | 7 | 3 | 0.429 |
| Railroads | 7 | 3 | 0.429 |
| Lightblue | 4 | 3 | 0.750 |
| Orange | 4 | 3 | 0.750 |
| Utilities | 4 | 2 | 0.500 |
| Red | 3 | 3 | 1.000 |
| Brown | 3 | 2 | 0.667 |
| Green | 3 | 3 | 1.000 |
| Indigo | 2 | 2 | 1.000 |
| Yellow | 2 | 2 | 1.000 |

**Buy rate by cash bucket**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| 200-500 | 2 | 2 | 1.000 |
| 500-1000 | 13 | 12 | 0.923 |
| 1000-1500 | 11 | 8 | 0.727 |
| >=1500 | 13 | 4 | 0.308 |

**Buy rate by monopoly opportunity**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| fresh | 21 | 10 | 0.476 |
| partial_self | 7 | 7 | 1.000 |
| opponent_dominates | 11 | 9 | 0.818 |

## Surprising decisions (qualitative excerpts)

### PASS while flush (cash >= 1000)

- cash=$1750, prop=`A1 Mediterranean Avenue` (Brown group, $60), self=0/2, opp=0 — reason: _Opponent already owns two-thirds of Brown so I cannot monopolise it; spending $60 here hurts my liquidity._
- cash=$1690, prop=`C1 St. Charles Place` (Pink group, $140), self=0/3, opp=0 — reason: _I have enough cash to buy this and I can afford to pass on the next move._
- cash=$1690, prop=`C3 Virginia Avenue` (Pink group, $160), self=0/3, opp=0 — reason: _I have enough cash to buy this and I can afford to pass on the next move._
- cash=$1550, prop=`U2 Waterworks` (Utilities group, $150), self=0/2, opp=0 — reason: _Buying this is a waste of cash and I have no path to monopoly._
- cash=$1500, prop=`B1 Oriental Avenue` (Lightblue group, $100), self=0/3, opp=0 — reason: _I have no cash to buy this and the opponent already owns all three Lightblue properties._
- cash=$1500, prop=`C3 Virginia Avenue` (Pink group, $160), self=0/3, opp=0 — reason: _I have no cash to buy this and the opponent already owns all three Pink properties._
- cash=$1500, prop=`R1 Reading Railroad` (Railroads group, $200), self=0/4, opp=0 — reason: _I have no cash to buy this and the opponent already owns all four Railroads._
- cash=$1500, prop=`D2 Tennessee Avenue` (Orange group, $180), self=0/3, opp=0 — reason: _I have no cash to buy this and the opponent already owns all three Orange properties._
- cash=$1500, prop=`C2 States Avenue` (Pink group, $140), self=0/3, opp=0 — reason: _I have no cash to buy this and the opponent already owns all three Pink properties._
- cash=$1404, prop=`R3 B&O Railroad` (Railroads group, $200), self=0/4, opp=1 — reason: _Buying this is a waste of cash and will not help me monopolise the Railroads._

### BUY at cash floor (cash < 300)

_None._

