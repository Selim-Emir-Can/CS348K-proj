# LLM-decision analysis: `default`

- games: **20** (truncated: 0)
- mean rounds: **76.0**, mean transfer rate: **100.5** $/round
- winners: `LLM_p1`=9, `LLM_p0`=7, `LLM_p2`=4

## Decisions

- total decisions logged: **1207**
- prefilter PASSes: `cash_floor`=149, `cant_afford`=141 (total 290)
- LLM calls: **917**, buy rate: **0.485**
- LLM call latency: median 8216 ms, mean 8234 ms, max 9434 ms
- parse-path distribution: `first_answer_tag`=917

## Hallucination detector

- LLM calls flagged by the validator: **0 / 917** (0.0%)
- of those, **real** model-side hallucinations (post 2026-04-29 reclassification): **0 / 917** (0.0%)
- and **spurious** validator-bug flags from float-drift on Player.money: **0**
- retries attempted (any): **0**, of which **0** (0%) cleared by the LAST attempt
- total retry calls across all decisions: **0** (avg 0.00 retries per LLM-call)
- decisions still flagged after MAX_RETRIES: **0** (of which **0** real, **0** spurious)

## Buy rate by colour group

| group | n | n_buy | buy_rate |
|-------|--:|------:|---------:|
| Green | 122 | 41 | 0.336 |
| Lightblue | 119 | 45 | 0.378 |
| Red | 113 | 42 | 0.372 |
| Orange | 110 | 52 | 0.473 |
| Railroads | 105 | 69 | 0.657 |
| Pink | 97 | 50 | 0.515 |
| Yellow | 89 | 46 | 0.517 |
| Indigo | 62 | 27 | 0.435 |
| Brown | 59 | 36 | 0.610 |
| Utilities | 41 | 37 | 0.902 |

**Buy rate by cash bucket**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| 200-500 | 41 | 22 | 0.537 |
| 500-1000 | 284 | 153 | 0.539 |
| 1000-1500 | 359 | 186 | 0.518 |
| >=1500 | 233 | 84 | 0.361 |

**Buy rate by monopoly opportunity**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| fresh | 227 | 208 | 0.916 |
| partial_self | 219 | 215 | 0.982 |
| opponent_dominates | 471 | 22 | 0.047 |

## Surprising decisions (qualitative excerpts)

### PASS while flush (cash >= 1000)

- cash=$6794.0, prop=`G1 Pacific Avenue` (Green group, $300), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Green; spending $300 hurts liquidity for no monopoly path._
- cash=$4898.0, prop=`G1 Pacific Avenue` (Green group, $300), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Green; spending $300 hurts liquidity for no monopoly path._
- cash=$4587.0, prop=`C3 Virginia Avenue` (Pink group, $160), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Pink; spending $160 hurts liquidity for no monopoly path._
- cash=$4213.0, prop=`C3 Virginia Avenue` (Pink group, $160), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Pink; spending $160 hurts liquidity for no monopoly path._
- cash=$4003.0, prop=`C3 Virginia Avenue` (Pink group, $160), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Pink; spending $160 hurts liquidity for no monopoly path._
- cash=$2940, prop=`R4 Short Line` (Railroads group, $200), self=1/4, opp=2 — reason: _opp_own_in_group is 2 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._
- cash=$3837, prop=`D3 New York Avenue` (Orange group, $200), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Orange; spending $200 hurts liquidity for no monopoly path._
- cash=$3412, prop=`B1 Oriental Avenue` (Lightblue group, $100), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Lightblue; spending $100 hurts liquidity for no monopoly path._
- cash=$3043.0, prop=`B2 Vermont Avenue` (Lightblue group, $100), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Lightblue; spending $100 hurts liquidity for no monopoly path._
- cash=$2997, prop=`U2 Waterworks` (Utilities group, $150), self=0/2, opp=1 — reason: _opp_own_in_group is 1 of 2 so opponent already controls most of Utilities; spending $150 hurts liquidity for no monopoly path._

### BUY at cash floor (cash < 300)

_None._
