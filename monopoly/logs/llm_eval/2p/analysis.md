# Combined LLM-decision analysis

## default

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


## ga_2p_winner

# LLM-decision analysis: `ga_2p_winner`

- games: **20** (truncated: 0)
- mean rounds: **46.8**, mean transfer rate: **78.6** $/round
- winners: `LLM_p1`=14, `LLM_p0`=6

## Decisions

- total decisions logged: **614**
- prefilter PASSes: `cant_afford`=166, `cash_floor`=107 (total 273)
- LLM calls: **341**, buy rate: **0.745**
- LLM call latency: median 8176 ms, mean 8186 ms, max 8670 ms
- parse-path distribution: `first_answer_tag`=341

## Hallucination detector

- LLM calls flagged by the validator: **15 / 341** (4.4%)
- of those, **real** model-side hallucinations (post 2026-04-29 reclassification): **0 / 341** (0.0%)
- and **spurious** validator-bug flags from float-drift on Player.money: **15**
- retries attempted (any): **15**, of which **0** (0%) cleared by the LAST attempt
- total retry calls across all decisions: **60** (avg 0.18 retries per LLM-call)
- decisions still flagged after MAX_RETRIES: **15** (of which **6** real, **9** spurious)
- per-field first-attempt mismatch counts (spurious — validator bug):
  - `cash`: 15
- legacy issue labels (regex detector, kept for back-compat):
  - `echo unparseable for 'cash': '$411.79999999999995' (STATE.cash=411.79999999999995)`: 2
  - `echo unparseable for 'cash': '$1093.4' (STATE.cash=1093.4)`: 2
  - `echo unparseable for 'cash': '$1274.4' (STATE.cash=1274.4)`: 2
  - `echo unparseable for 'cash': '$534.8' (STATE.cash=534.8)`: 1
  - `echo unparseable for 'cash': '$460.19999999999993' (STATE.cash=460.19999999999993)`: 1
  - `echo unparseable for 'cash': '$511.79999999999995' (STATE.cash=511.79999999999995)`: 1
  - `echo unparseable for 'cash': '$361.79999999999995' (STATE.cash=361.79999999999995)`: 1
  - `echo unparseable for 'cash': '$900.8' (STATE.cash=900.8)`: 1
  - `echo unparseable for 'cash': '$319.79999999999995' (STATE.cash=319.79999999999995)`: 1
  - `echo unparseable for 'cash': '$339.79999999999995' (STATE.cash=339.79999999999995)`: 1
  - `echo unparseable for 'cash': '$672.8' (STATE.cash=672.8)`: 1
  - `echo unparseable for 'cash': '$1018.8' (STATE.cash=1018.8)`: 1

### Hallucinated reason examples

- cash=$534.8, cost=$262, self=0/3, opp=2 — **original** _opp_own_in_group is 2 of 3 so opponent already controls most of Red; spending $262 hurts liquidity for no monopoly path._
  - **final retry** (still flagged, 4 retries): _opp_own_in_group is 2 of 3 so opponent already controls most of Red; spending $262 hurts liquidity for no monopoly path._
- cash=$460.19999999999993, cost=$41, self=1/2, opp=0 — **original** _opp_own_in_group is 0 so I can monopolise this group._
  - **final retry** (still flagged, 4 retries): _<no REASON>_
- cash=$511.79999999999995, cost=$200, self=1/4, opp=1 — **original** _cash is $511.79999999999995 and cost is $200 so I can afford this._
  - **final retry** (still flagged, 4 retries): _<no REASON>_
- cash=$361.79999999999995, cost=$150, self=0/2, opp=0 — **original** _cash is $361.79999999999995 and cost is $150 so I can afford this._
  - **final retry** (still flagged, 4 retries): _<no REASON>_
- cash=$411.79999999999995, cost=$41, self=0/2, opp=1 — **original** _opp_own_in_group is 1 of 2 so opponent already controls most of Brown; spending $41 hurts liquidity for no monopoly path._
  - **final retry** (still flagged, 4 retries): _opp_own_in_group is 1 of 2 so opponent already controls most of Brown; spending $41 hurts liquidity for no monopoly path._
- cash=$411.79999999999995, cost=$160, self=0/1, opp=0 — **original** _I have $411.79999999999995 and cost is $160 so I can afford this._
  - **final retry** (still flagged, 4 retries): _<no REASON>_
- cash=$1093.4, cost=$296, self=0/3, opp=2 — **original** _opp_own_in_group is 2 of 3 so opponent already controls most of Red; spending $296 hurts liquidity for no monopoly path._
  - **final retry** (still flagged, 4 retries): _opp_own_in_group is 2 of 3 so opponent already controls most of Red; spending $296 hurts liquidity for no monopoly path._
- cash=$1093.4, cost=$200, self=1/4, opp=2 — **original** _opp_own_in_group is 2 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._
  - **final retry** (still flagged, 4 retries): _opp_own_in_group is 2 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._
- cash=$1274.4, cost=$200, self=1/4, opp=2 — **original** _opp_own_in_group is 2 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._
  - **final retry** (still flagged, 4 retries): _opp_own_in_group is 2 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._
- cash=$1274.4, cost=$600, self=0/2, opp=0 — **original** _opp_own_in_group is 0 so I can afford this and monopolise Indigo._
  - **final retry** (still flagged, 4 retries): _opp_own_in_group is 0 so I can afford this and monopolise Indigo._

## Buy rate by colour group

| group | n | n_buy | buy_rate |
|-------|--:|------:|---------:|
| Railroads | 90 | 62 | 0.689 |
| Red | 61 | 37 | 0.607 |
| Brown | 47 | 32 | 0.681 |
| Pink | 40 | 33 | 0.825 |
| Utilities | 32 | 26 | 0.812 |
| Indigo | 25 | 18 | 0.720 |
| Lightblue | 18 | 18 | 1.000 |
| Orange | 16 | 16 | 1.000 |
| Green | 12 | 12 | 1.000 |

**Buy rate by cash bucket**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| 200-500 | 37 | 22 | 0.595 |
| 500-1000 | 134 | 101 | 0.754 |
| 1000-1500 | 108 | 77 | 0.713 |
| >=1500 | 62 | 54 | 0.871 |

**Buy rate by monopoly opportunity**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| fresh | 172 | 161 | 0.936 |
| partial_self | 84 | 84 | 1.000 |
| opponent_dominates | 85 | 9 | 0.106 |

## Surprising decisions (qualitative excerpts)

### PASS while flush (cash >= 1000)

- cash=$2463, prop=`R3 B&O Railroad` (Railroads group, $200), self=0/4, opp=3 — reason: _opp_own_in_group is 3 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._
- cash=$2313, prop=`R3 B&O Railroad` (Railroads group, $200), self=0/4, opp=3 — reason: _opp_own_in_group is 3 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._
- cash=$1274.4, prop=`R4 Short Line` (Railroads group, $200), self=1/4, opp=2 — reason: _opp_own_in_group is 2 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._
- cash=$1093.4, prop=`R4 Short Line` (Railroads group, $200), self=1/4, opp=2 — reason: _opp_own_in_group is 2 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._
- cash=$2056, prop=`C1 St. Charles Place` (Pink group, $247), self=0/2, opp=1 — reason: _opp_own_in_group is 1 of 2 so opponent already controls most of Pink; spending $247 hurts liquidity for no monopoly path._
- cash=$2056, prop=`E3 Illinois Avenue` (Red group, $444), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Red; spending $444 hurts liquidity for no monopoly path._
- cash=$1725, prop=`C1 St. Charles Place` (Pink group, $247), self=0/2, opp=1 — reason: _opp_own_in_group is 1 of 2 so opponent already controls most of Pink; spending $247 hurts liquidity for no monopoly path._
- cash=$1525, prop=`R3 B&O Railroad` (Railroads group, $200), self=0/4, opp=1 — reason: _opp_own_in_group is 1 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._
- cash=$1525, prop=`R4 Short Line` (Railroads group, $200), self=0/4, opp=1 — reason: _opp_own_in_group is 1 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._
- cash=$1525, prop=`E3 Illinois Avenue` (Red group, $444), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Red; spending $444 hurts liquidity for no monopoly path._

### BUY at cash floor (cash < 300)

- cash=$295, prop=`A2 Baltic Avenue` (Brown group, $41), self=1/2, opp=0 — reason: _you_own_in_group=1 and group_size=2 so I am one short; this purchase completes my Brown monopoly._

