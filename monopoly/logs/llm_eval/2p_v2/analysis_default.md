# LLM-decision analysis: `default`

- games: **20** (truncated: 0)
- mean rounds: **60.5**, mean transfer rate: **50.1** $/round
- winners: `LLM_p1`=15, `LLM_p0`=5

## Decisions

- total decisions logged: **933**
- prefilter PASSes: `cash_floor`=187, `cant_afford`=162 (total 349)
- LLM calls: **584**, buy rate: **0.671**
- LLM call latency: median 8240 ms, mean 8248 ms, max 8763 ms
- parse-path distribution: `first_answer_tag`=584

## Hallucination detector

- LLM calls flagged by the validator: **0 / 584** (0.0%)
- of those, **real** model-side hallucinations (post 2026-04-29 reclassification): **0 / 584** (0.0%)
- and **spurious** validator-bug flags from float-drift on Player.money: **0**
- retries attempted (any): **0**, of which **0** (0%) cleared by the LAST attempt
- total retry calls across all decisions: **0** (avg 0.00 retries per LLM-call)
- decisions still flagged after MAX_RETRIES: **0** (of which **0** real, **0** spurious)

## Buy rate by colour group

| group | n | n_buy | buy_rate |
|-------|--:|------:|---------:|
| Lightblue | 83 | 46 | 0.554 |
| Railroads | 77 | 60 | 0.779 |
| Pink | 68 | 49 | 0.721 |
| Orange | 63 | 45 | 0.714 |
| Red | 60 | 38 | 0.633 |
| Yellow | 59 | 36 | 0.610 |
| Green | 54 | 31 | 0.574 |
| Utilities | 40 | 32 | 0.800 |
| Indigo | 40 | 23 | 0.575 |
| Brown | 40 | 32 | 0.800 |

**Buy rate by cash bucket**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| 200-500 | 73 | 44 | 0.603 |
| 500-1000 | 235 | 148 | 0.630 |
| 1000-1500 | 207 | 152 | 0.734 |
| >=1500 | 69 | 48 | 0.696 |

**Buy rate by monopoly opportunity**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| fresh | 219 | 203 | 0.927 |
| partial_self | 172 | 172 | 1.000 |
| opponent_dominates | 193 | 17 | 0.088 |

## Surprising decisions (qualitative excerpts)

### PASS while flush (cash >= 1000)

- cash=$4034, prop=`G1 Pacific Avenue` (Green group, $300), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Green; spending $300 hurts liquidity for no monopoly path._
- cash=$3659, prop=`F2 Ventnor Avenue` (Yellow group, $260), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Yellow; spending $260 hurts liquidity for no monopoly path._
- cash=$3024, prop=`H2 Boardwalk` (Indigo group, $400), self=0/2, opp=0 — reason: _opp_own_total is 11 of 12 so opponent already controls most of Indigo; spending $400 hurts liquidity for no monopoly path._
- cash=$2799, prop=`D2 Tennessee Avenue` (Orange group, $180), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Orange; spending $180 hurts liquidity for no monopoly path._
- cash=$2799, prop=`U2 Waterworks` (Utilities group, $150), self=0/2, opp=1 — reason: _opp_own_in_group is 1 of 2 so opponent already controls most of Utilities; spending $150 hurts liquidity for no monopoly path._
- cash=$2523, prop=`G3 Pennsylvania Avenue` (Green group, $320), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Green; spending $320 hurts liquidity for no monopoly path._
- cash=$2362, prop=`D1 St. James Place` (Orange group, $180), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Orange; spending $180 hurts liquidity for no monopoly path._
- cash=$2340, prop=`G3 Pennsylvania Avenue` (Green group, $320), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Green; spending $320 hurts liquidity for no monopoly path._
- cash=$1873, prop=`U2 Waterworks` (Utilities group, $150), self=0/2, opp=1 — reason: _opp_own_in_group is 1 of 2 so opponent already controls most of Utilities; spending $150 hurts liquidity for no monopoly path._
- cash=$1871, prop=`D1 St. James Place` (Orange group, $180), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Orange; spending $180 hurts liquidity for no monopoly path._

### BUY at cash floor (cash < 300)

- cash=$262, prop=`A2 Baltic Avenue` (Brown group, $60), self=1/2, opp=0 — reason: _you_own_in_group=1 and group_size=2 so I am one short; this purchase completes my Brown monopoly._
- cash=$272, prop=`A1 Mediterranean Avenue` (Brown group, $60), self=0/2, opp=1 — reason: _you_own_in_group=0 and group_size=2 so I am one short; this purchase completes my Brown monopoly._
- cash=$292, prop=`A2 Baltic Avenue` (Brown group, $60), self=0/2, opp=0 — reason: _you_own_in_group=0 and group_size=2 so I can monopolise this group._
