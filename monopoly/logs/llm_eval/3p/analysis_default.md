# LLM-decision analysis: `default`

- games: **20** (truncated: 0)
- mean rounds: **76.0**, mean transfer rate: **100.6** $/round
- winners: `LLM_p1`=9, `LLM_p0`=7, `LLM_p2`=4

## Decisions

- total decisions logged: **1210**
- prefilter PASSes: `cash_floor`=148, `cant_afford`=142 (total 290)
- LLM calls: **920**, buy rate: **0.483**
- LLM call latency: median 8175 ms, mean 8185 ms, max 8506 ms
- parse-path distribution: `first_answer_tag`=920

## Hallucination detector

- LLM calls flagged by the validator: **52 / 920** (5.7%)
- of those, **real** model-side hallucinations (post 2026-04-29 reclassification): **0 / 920** (0.0%)
- and **spurious** validator-bug flags from float-drift on Player.money: **52**
- retries attempted (any): **52**, of which **0** (0%) cleared by the LAST attempt
- total retry calls across all decisions: **208** (avg 0.23 retries per LLM-call)
- decisions still flagged after MAX_RETRIES: **52** (of which **19** real, **33** spurious)
- per-field first-attempt mismatch counts (spurious — validator bug):
  - `cash`: 52
- legacy issue labels (regex detector, kept for back-compat):
  - `echo unparseable for 'cash': '$1109.0' (STATE.cash=1109.0)`: 2
  - `echo unparseable for 'cash': '$1232.0' (STATE.cash=1232.0)`: 2
  - `echo unparseable for 'cash': '$2639.0' (STATE.cash=2639.0)`: 1
  - `echo unparseable for 'cash': '$2893.0' (STATE.cash=2893.0)`: 1
  - `echo unparseable for 'cash': '$3043.0' (STATE.cash=3043.0)`: 1
  - `echo unparseable for 'cash': '$4165.0' (STATE.cash=4165.0)`: 1
  - `echo unparseable for 'cash': '$520.0' (STATE.cash=520.0)`: 1
  - `echo unparseable for 'cash': '$726.0' (STATE.cash=726.0)`: 1
  - `echo unparseable for 'cash': '$2044.0' (STATE.cash=2044.0)`: 1
  - `echo unparseable for 'cash': '$2168.0' (STATE.cash=2168.0)`: 1
  - `echo unparseable for 'cash': '$2798.0' (STATE.cash=2798.0)`: 1
  - `echo unparseable for 'cash': '$800.0' (STATE.cash=800.0)`: 1
  - `echo unparseable for 'cash': '$486.0' (STATE.cash=486.0)`: 1
  - `echo unparseable for 'cash': '$948.0' (STATE.cash=948.0)`: 1
  - `echo unparseable for 'cash': '$634.0' (STATE.cash=634.0)`: 1
  - `echo unparseable for 'cash': '$534.0' (STATE.cash=534.0)`: 1
  - `echo unparseable for 'cash': '$1082.0' (STATE.cash=1082.0)`: 1
  - `echo unparseable for 'cash': '$882.0' (STATE.cash=882.0)`: 1
  - `echo unparseable for 'cash': '$577.0' (STATE.cash=577.0)`: 1
  - `echo unparseable for 'cash': '$815.0' (STATE.cash=815.0)`: 1
  - `echo unparseable for 'cash': '$377.0' (STATE.cash=377.0)`: 1
  - `echo unparseable for 'cash': '$422.0' (STATE.cash=422.0)`: 1
  - `echo unparseable for 'cash': '$2081.0' (STATE.cash=2081.0)`: 1
  - `echo unparseable for 'cash': '$481.0' (STATE.cash=481.0)`: 1
  - `echo unparseable for 'cash': '$1039.0' (STATE.cash=1039.0)`: 1
  - `echo unparseable for 'cash': '$788.0' (STATE.cash=788.0)`: 1
  - `echo unparseable for 'cash': '$738.0' (STATE.cash=738.0)`: 1
  - `echo unparseable for 'cash': '$1924.0' (STATE.cash=1924.0)`: 1
  - `echo unparseable for 'cash': '$2063.0' (STATE.cash=2063.0)`: 1
  - `echo unparseable for 'cash': '$2850.0' (STATE.cash=2850.0)`: 1
  - `echo unparseable for 'cash': '$2750.0' (STATE.cash=2750.0)`: 1
  - `echo unparseable for 'cash': '$2966.0' (STATE.cash=2966.0)`: 1
  - `echo unparseable for 'cash': '$2672.0' (STATE.cash=2672.0)`: 1
  - `echo unparseable for 'cash': '$3935.0' (STATE.cash=3935.0)`: 1
  - `echo unparseable for 'cash': '$3940.0' (STATE.cash=3940.0)`: 1
  - `echo unparseable for 'cash': '$3789.0' (STATE.cash=3789.0)`: 1
  - `echo unparseable for 'cash': '$4039.0' (STATE.cash=4039.0)`: 1
  - `echo unparseable for 'cash': '$4300.0' (STATE.cash=4300.0)`: 1
  - `echo unparseable for 'cash': '$4450.0' (STATE.cash=4450.0)`: 1
  - `echo unparseable for 'cash': '$6170.0' (STATE.cash=6170.0)`: 1
  - `echo unparseable for 'cash': '$404.0' (STATE.cash=404.0)`: 1
  - `echo unparseable for 'cash': '$605.0' (STATE.cash=605.0)`: 1
  - `echo unparseable for 'cash': '$583.0' (STATE.cash=583.0)`: 1
  - `echo unparseable for 'cash': '$871.0' (STATE.cash=871.0)`: 1
  - `echo unparseable for 'cash': '$1111.0' (STATE.cash=1111.0)`: 1
  - `echo unparseable for 'cash': '$810.0' (STATE.cash=810.0)`: 1
  - `echo unparseable for 'cash': '$710.0' (STATE.cash=710.0)`: 1
  - `echo unparseable for 'cash': '$1855.0' (STATE.cash=1855.0)`: 1
  - `echo unparseable for 'cash': '$3874.0' (STATE.cash=3874.0)`: 1
  - `echo unparseable for 'cash': '$1550.0' (STATE.cash=1550.0)`: 1

### Hallucinated reason examples

- cash=$2639.0, cost=$200, self=3/4, opp=0 — **original** _you_own_in_group=3 and group_size=4 so I am one short; this purchase completes my Railroads monopoly._
  - **final retry** (still flagged, 4 retries): _<no REASON>_
- cash=$2893.0, cost=$300, self=0/3, opp=1 — **original** _opp_own_in_group is 1 of 3 so opponent already controls most of Green; spending $300 hurts liquidity for no monopoly path._
  - **final retry** (still flagged, 4 retries): _opp_own_in_group is 1 of 3 so opponent already controls most of Green; spending $300 hurts liquidity for no monopoly path._
- cash=$3043.0, cost=$100, self=0/3, opp=2 — **original** _opp_own_in_group is 2 of 3 so opponent already controls most of Lightblue; spending $100 hurts liquidity for no monopoly path._
  - **final retry** (still flagged, 4 retries): _opp_own_in_group is 2 of 3 so opponent already controls most of Lightblue; spending $100 hurts liquidity for no monopoly path._
- cash=$4165.0, cost=$280, self=2/3, opp=0 — **original** _you_own_in_group=2 and group_size=3 so I am one short; this purchase completes my Yellow monopoly._
  - **final retry** (still flagged, 4 retries): _<no REASON>_
- cash=$520.0, cost=$220, self=1/3, opp=0 — **original** _you_own_in_group=1 and group_size=3 so I am one short; this purchase completes my Red monopoly._
  - **final retry** (still flagged, 4 retries): _I have $1000, cost is only $100, and opp_own_in_group is 0 so I can still monopolise thisroperty._
- cash=$726.0, cost=$140, self=0/3, opp=2 — **original** _you_own_total=15 and group_size=3 so I have a monopoly._
  - **final retry** (still flagged, 4 retries): _you_own_total=15 and group_size=3 so I have a monopoly._
- cash=$2044.0, cost=$100, self=2/3, opp=0 — **original** _you_own_in_group=2 and group_size=3 so I am one short; this purchase completes my Lightblue monopoly._
  - **final retry** (still flagged, 4 retries): _you_own_in_group=2 and group_size=3 so I am one short; this purchase completes my Green monopoly._
- cash=$2168.0, cost=$320, self=1/3, opp=0 — **original** _you_own_in_group=1 and group_size=3 so I am one short; this purchase completes my Green monopoly._
  - **final retry** (still flagged, 4 retries): _I have $1000, cost is only $100, and opp_own_in_group is 0 so I can still monopolise thisroperty._
- cash=$2798.0, cost=$400, self=1/2, opp=0 — **original** _you_own_in_group=1 and group_size=2 so I am one short; this purchase completes my Indigo monopoly._
  - **final retry** (still flagged, 4 retries): _<no REASON>_
- cash=$800.0, cost=$300, self=2/3, opp=0 — **original** _you_own_in_group=2 and group_size=3 so I am one short; this purchase completes my Green monopoly._
  - **final retry** (still flagged, 4 retries): _I have $1000, cost is only $100, and opp_own_in_group is 0 so I can still monopolise thisroperty._

## Buy rate by colour group

| group | n | n_buy | buy_rate |
|-------|--:|------:|---------:|
| Green | 122 | 41 | 0.336 |
| Lightblue | 119 | 45 | 0.378 |
| Red | 113 | 42 | 0.372 |
| Orange | 110 | 52 | 0.473 |
| Railroads | 108 | 68 | 0.630 |
| Pink | 97 | 49 | 0.505 |
| Yellow | 89 | 46 | 0.517 |
| Indigo | 62 | 28 | 0.452 |
| Brown | 59 | 36 | 0.610 |
| Utilities | 41 | 37 | 0.902 |

**Buy rate by cash bucket**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| 200-500 | 40 | 21 | 0.525 |
| 500-1000 | 285 | 154 | 0.540 |
| 1000-1500 | 359 | 185 | 0.515 |
| >=1500 | 236 | 84 | 0.356 |

**Buy rate by monopoly opportunity**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| fresh | 227 | 208 | 0.916 |
| partial_self | 219 | 215 | 0.982 |
| opponent_dominates | 474 | 21 | 0.044 |

## Surprising decisions (qualitative excerpts)

### PASS while flush (cash >= 1000)

- cash=$6170.0, prop=`G1 Pacific Avenue` (Green group, $300), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Green; spending $300 hurts liquidity for no monopoly path._
- cash=$4450.0, prop=`R4 Short Line` (Railroads group, $200), self=1/4, opp=2 — reason: _opp_own_in_group is 2 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._
- cash=$3789.0, prop=`R4 Short Line` (Railroads group, $200), self=1/4, opp=2 — reason: _opp_own_in_group is 2 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._
- cash=$4300.0, prop=`G1 Pacific Avenue` (Green group, $300), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Green; spending $300 hurts liquidity for no monopoly path._
- cash=$4039.0, prop=`C3 Virginia Avenue` (Pink group, $160), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Pink; spending $160 hurts liquidity for no monopoly path._
- cash=$2940, prop=`R4 Short Line` (Railroads group, $200), self=1/4, opp=2 — reason: _opp_own_in_group is 2 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._
- cash=$3940.0, prop=`C3 Virginia Avenue` (Pink group, $160), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Pink; spending $160 hurts liquidity for no monopoly path._
- cash=$3935.0, prop=`C3 Virginia Avenue` (Pink group, $160), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Pink; spending $160 hurts liquidity for no monopoly path._
- cash=$3837, prop=`D3 New York Avenue` (Orange group, $200), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Orange; spending $200 hurts liquidity for no monopoly path._
- cash=$2672.0, prop=`R4 Short Line` (Railroads group, $200), self=1/4, opp=2 — reason: _opp_own_in_group is 2 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._

### BUY at cash floor (cash < 300)

_None._
