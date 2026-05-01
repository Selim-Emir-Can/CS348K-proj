# LLM-decision analysis: `ga_3p_winner`

- games: **20** (truncated: 0)
- mean rounds: **43.2**, mean transfer rate: **127.4** $/round
- winners: `LLM_p2`=9, `LLM_p0`=6, `LLM_p1`=5

## Decisions

- total decisions logged: **750**
- prefilter PASSes: `cant_afford`=154, `cash_floor`=151 (total 305)
- LLM calls: **445**, buy rate: **0.701**
- LLM call latency: median 8204 ms, mean 8216 ms, max 9378 ms
- parse-path distribution: `first_answer_tag`=445

## Hallucination detector

- LLM calls flagged by the validator: **0 / 445** (0.0%)
- of those, **real** model-side hallucinations (post 2026-04-29 reclassification): **0 / 445** (0.0%)
- and **spurious** validator-bug flags from float-drift on Player.money: **0**
- retries attempted (any): **0**, of which **0** (0%) cleared by the LAST attempt
- total retry calls across all decisions: **0** (avg 0.00 retries per LLM-call)
- decisions still flagged after MAX_RETRIES: **0** (of which **0** real, **0** spurious)

## Buy rate by colour group

| group | n | n_buy | buy_rate |
|-------|--:|------:|---------:|
| Railroads | 82 | 62 | 0.756 |
| Orange | 75 | 42 | 0.560 |
| Pink | 68 | 39 | 0.574 |
| Brown | 59 | 36 | 0.610 |
| Utilities | 41 | 34 | 0.829 |
| Yellow | 36 | 23 | 0.639 |
| Red | 33 | 26 | 0.788 |
| Green | 19 | 18 | 0.947 |
| Indigo | 16 | 16 | 1.000 |
| Lightblue | 16 | 16 | 1.000 |

**Buy rate by cash bucket**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| 200-500 | 63 | 33 | 0.524 |
| 500-1000 | 135 | 91 | 0.674 |
| 1000-1500 | 144 | 112 | 0.778 |
| >=1500 | 103 | 76 | 0.738 |

**Buy rate by monopoly opportunity**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| fresh | 205 | 195 | 0.951 |
| partial_self | 92 | 92 | 1.000 |
| opponent_dominates | 148 | 25 | 0.169 |

## Surprising decisions (qualitative excerpts)

### PASS while flush (cash >= 1000)

- cash=$3092, prop=`D2 Tennessee Avenue` (Orange group, $328), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Orange; spending $328 hurts liquidity for no monopoly path._
- cash=$2958, prop=`F2 Ventnor Avenue` (Yellow group, $447), self=0/2, opp=1 — reason: _opp_own_in_group is 1 of 2 so opponent already controls most of Yellow; spending $447 hurts liquidity for no monopoly path._
- cash=$2781, prop=`D2 Tennessee Avenue` (Orange group, $328), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Orange; spending $328 hurts liquidity for no monopoly path._
- cash=$2319, prop=`D2 Tennessee Avenue` (Orange group, $328), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Orange; spending $328 hurts liquidity for no monopoly path._
- cash=$1846, prop=`C3 Virginia Avenue` (Pink group, $156), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Pink; spending $156 hurts liquidity for no monopoly path._
- cash=$1846, prop=`D3 New York Avenue` (Orange group, $344), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Orange; spending $344 hurts liquidity for no monopoly path._
- cash=$1804, prop=`F2 Ventnor Avenue` (Yellow group, $447), self=0/2, opp=1 — reason: _opp_own_in_group is 1 of 2 so opponent already controls most of Yellow; spending $447 hurts liquidity for no monopoly path._
- cash=$1701, prop=`C2 States Avenue` (Pink group, $198), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Pink; spending $198 hurts liquidity for no monopoly path._
- cash=$1635, prop=`A2 Baltic Avenue` (Brown group, $54), self=0/2, opp=1 — reason: _opp_own_in_group is 1 of 2 so opponent already controls most of Brown; spending $54 hurts liquidity for no monopoly path._
- cash=$1599, prop=`C1 St. Charles Place` (Pink group, $272), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Pink; spending $272 hurts liquidity for no monopoly path._

### BUY at cash floor (cash < 300)

_None._
