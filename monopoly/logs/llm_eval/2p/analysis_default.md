# LLM-decision analysis: `default`

- games: **20** (truncated: 0)
- mean rounds: **60.5**, mean transfer rate: **50.1** $/round
- winners: `LLM_p1`=15, `LLM_p0`=5

## Decisions

- total decisions logged: **944**
- prefilter PASSes: `cash_floor`=187, `cant_afford`=162 (total 349)
- LLM calls: **595**, buy rate: **0.677**
- LLM call latency: median 8226 ms, mean 8264 ms, max 10785 ms
- parse-path distribution: `first_answer_tag`=595

## Hallucination detector

- LLM calls flagged by the validator: **3 / 595** (0.5%)
- of those, **real** model-side hallucinations (post 2026-04-29 reclassification): **0 / 595** (0.0%)
- and **spurious** validator-bug flags from float-drift on Player.money: **3**
- retries attempted (any): **3**, of which **0** (0%) cleared by the LAST attempt
- total retry calls across all decisions: **12** (avg 0.02 retries per LLM-call)
- decisions still flagged after MAX_RETRIES: **3** (of which **2** real, **1** spurious)
- per-field first-attempt mismatch counts (spurious — validator bug):
  - `cash`: 3
- legacy issue labels (regex detector, kept for back-compat):
  - `echo unparseable for 'cash': '$416.0' (STATE.cash=416.0)`: 1
  - `echo unparseable for 'cash': '$372.0' (STATE.cash=372.0)`: 1
  - `echo unparseable for 'cash': '$623.0' (STATE.cash=623.0)`: 1

### Hallucinated reason examples

- cash=$416.0, cost=$180, self=1/3, opp=0 — **original** _you_own_in_group=1 and group_size=3 so I am one short; this purchase completes my Orange monopoly._
  - **final retry** (still flagged, 4 retries): _I have $1000, cost is only $100, and opp_own_in_group is 0 so I can still monopolise thisroperty._
- cash=$372.0, cost=$100, self=0/3, opp=2 — **original** _opp_own_in_group is 2 of 3 so opponent already controls most of Lightblue; spending $100 hurts liquidity for no monopoly path._
  - **final retry** (still flagged, 4 retries): _opp_own_in_group is 2 of 3 so opponent already controls most of Lightblue; spending $100 hurts liquidity for no monopoly path._
- cash=$623.0, cost=$280, self=2/3, opp=0 — **original** _opp_own_in_group is 0 so I can monopolise Yellow._
  - **final retry** (still flagged, 4 retries): _<no REASON>_

## Buy rate by colour group

| group | n | n_buy | buy_rate |
|-------|--:|------:|---------:|
| Lightblue | 85 | 48 | 0.565 |
| Railroads | 79 | 62 | 0.785 |
| Pink | 71 | 52 | 0.732 |
| Orange | 65 | 47 | 0.723 |
| Red | 61 | 39 | 0.639 |
| Yellow | 59 | 36 | 0.610 |
| Green | 54 | 31 | 0.574 |
| Utilities | 41 | 33 | 0.805 |
| Indigo | 40 | 23 | 0.575 |
| Brown | 40 | 32 | 0.800 |

**Buy rate by cash bucket**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| 200-500 | 73 | 44 | 0.603 |
| 500-1000 | 235 | 148 | 0.630 |
| 1000-1500 | 214 | 159 | 0.743 |
| >=1500 | 73 | 52 | 0.712 |

**Buy rate by monopoly opportunity**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| fresh | 229 | 213 | 0.930 |
| partial_self | 172 | 172 | 1.000 |
| opponent_dominates | 194 | 18 | 0.093 |

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
