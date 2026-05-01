# Combined LLM-decision analysis

## default

# LLM-decision analysis: `default`

- games: **1** (truncated: 0)
- mean rounds: **74.0**, mean transfer rate: **60.9** $/round
- winners: `LLM_p1`=1

## Decisions

- total decisions logged: **58**
- prefilter PASSes: `cant_afford`=17, `cash_floor`=8 (total 25)
- LLM calls: **33**, buy rate: **0.636**
- LLM call latency: median 8235 ms, mean 8238 ms, max 8410 ms
- parse-path distribution: `first_answer_tag`=33

## Hallucination detector

- LLM calls flagged as contradicting STATE: **0 / 33** (0.0%)
- retries attempted (any): **0**, of which **0** (0%) cleared by the LAST attempt
- total retry calls across all decisions: **0** (avg 0.00 retries per LLM-call)
- decisions still flagged after MAX_RETRIES: **0**

## Buy rate by colour group

| group | n | n_buy | buy_rate |
|-------|--:|------:|---------:|
| Green | 5 | 1 | 0.200 |
| Railroads | 4 | 3 | 0.750 |
| Orange | 4 | 2 | 0.500 |
| Red | 4 | 3 | 0.750 |
| Lightblue | 3 | 3 | 1.000 |
| Pink | 3 | 3 | 1.000 |
| Indigo | 3 | 1 | 0.333 |
| Yellow | 3 | 1 | 0.333 |
| Utilities | 2 | 2 | 1.000 |
| Brown | 2 | 2 | 1.000 |

**Buy rate by cash bucket**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| 200-500 | 3 | 2 | 0.667 |
| 500-1000 | 10 | 7 | 0.700 |
| 1000-1500 | 12 | 8 | 0.667 |
| >=1500 | 8 | 4 | 0.500 |

**Buy rate by monopoly opportunity**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| fresh | 11 | 10 | 0.909 |
| partial_self | 8 | 8 | 1.000 |
| opponent_dominates | 14 | 3 | 0.214 |

## Surprising decisions (qualitative excerpts)

### PASS while flush (cash >= 1000)

- cash=$2362, prop=`D1 St. James Place` (Orange group, $180), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Orange; spending $180 hurts liquidity for no monopoly path._
- cash=$2340, prop=`G3 Pennsylvania Avenue` (Green group, $320), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Green; spending $320 hurts liquidity for no monopoly path._
- cash=$1742, prop=`G3 Pennsylvania Avenue` (Green group, $320), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Green; spending $320 hurts liquidity for no monopoly path._
- cash=$1642, prop=`F2 Ventnor Avenue` (Yellow group, $260), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Yellow; spending $260 hurts liquidity for no monopoly path._
- cash=$1348, prop=`G2 North Carolina Avenue` (Green group, $300), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Green; spending $300 hurts liquidity for no monopoly path._
- cash=$1210, prop=`F2 Ventnor Avenue` (Yellow group, $260), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Yellow; spending $260 hurts liquidity for no monopoly path._
- cash=$1186, prop=`H1 Park Place` (Indigo group, $350), self=0/2, opp=1 — reason: _opp_own_in_group is 1 of 2 so opponent already controls most of Indigo; spending $350 hurts liquidity for no monopoly path._
- cash=$1136, prop=`G3 Pennsylvania Avenue` (Green group, $320), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Green; spending $320 hurts liquidity for no monopoly path._

### BUY at cash floor (cash < 300)

_None._

