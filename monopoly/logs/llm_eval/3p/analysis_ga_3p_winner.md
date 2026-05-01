# LLM-decision analysis: `ga_3p_winner`

- games: **20** (truncated: 0)
- mean rounds: **44.6**, mean transfer rate: **127.6** $/round
- winners: `LLM_p2`=9, `LLM_p0`=6, `LLM_p1`=5

## Decisions

- total decisions logged: **758**
- prefilter PASSes: `cant_afford`=157, `cash_floor`=153 (total 310)
- LLM calls: **448**, buy rate: **0.694**
- LLM call latency: median 8190 ms, mean 8200 ms, max 8482 ms
- parse-path distribution: `first_answer_tag`=448

## Hallucination detector

- LLM calls flagged by the validator: **30 / 448** (6.7%)
- of those, **real** model-side hallucinations (post 2026-04-29 reclassification): **0 / 448** (0.0%)
- and **spurious** validator-bug flags from float-drift on Player.money: **30**
- retries attempted (any): **30**, of which **0** (0%) cleared by the LAST attempt
- total retry calls across all decisions: **120** (avg 0.27 retries per LLM-call)
- decisions still flagged after MAX_RETRIES: **30** (of which **20** real, **10** spurious)
- per-field first-attempt mismatch counts (spurious — validator bug):
  - `cash`: 30
- legacy issue labels (regex detector, kept for back-compat):
  - `echo unparseable for 'cash': '$388.0' (STATE.cash=388.0)`: 1
  - `echo unparseable for 'cash': '$861.0' (STATE.cash=861.0)`: 1
  - `echo unparseable for 'cash': '$1885.0' (STATE.cash=1885.0)`: 1
  - `echo unparseable for 'cash': '$944.0' (STATE.cash=944.0)`: 1
  - `echo unparseable for 'cash': '$665.0' (STATE.cash=665.0)`: 1
  - `echo unparseable for 'cash': '$718.0' (STATE.cash=718.0)`: 1
  - `echo unparseable for 'cash': '$429.4' (STATE.cash=429.4)`: 1
  - `echo unparseable for 'cash': '$429.2' (STATE.cash=429.2)`: 1
  - `echo unparseable for 'cash': '$1419.6' (STATE.cash=1419.6)`: 1
  - `echo unparseable for 'cash': '$446.5999999999999' (STATE.cash=446.5999999999999)`: 1
  - `echo unparseable for 'cash': '$756.5999999999999' (STATE.cash=756.5999999999999)`: 1
  - `echo unparseable for 'cash': '$562.0' (STATE.cash=562.0)`: 1
  - `echo unparseable for 'cash': '$508.0' (STATE.cash=508.0)`: 1
  - `echo unparseable for 'cash': '$493.0' (STATE.cash=493.0)`: 1
  - `echo unparseable for 'cash': '$435.40000000000003' (STATE.cash=435.40000000000003)`: 1
  - `echo unparseable for 'cash': '$448.40000000000003' (STATE.cash=448.40000000000003)`: 1
  - `echo unparseable for 'cash': '$555.4000000000001' (STATE.cash=555.4000000000001)`: 1
  - `echo unparseable for 'cash': '$575.4000000000001' (STATE.cash=575.4000000000001)`: 1
  - `echo unparseable for 'cash': '$430.2' (STATE.cash=430.2)`: 1
  - `echo unparseable for 'cash': '$484.4' (STATE.cash=484.4)`: 1
  - `echo unparseable for 'cash': '$542.0' (STATE.cash=542.0)`: 1
  - `echo unparseable for 'cash': '$400.2' (STATE.cash=400.2)`: 1
  - `echo unparseable for 'cash': '$1327.3999999999999' (STATE.cash=1327.3999999999999)`: 1
  - `echo unparseable for 'cash': '$1063.6' (STATE.cash=1063.6)`: 1
  - `echo unparseable for 'cash': '$935.5999999999999' (STATE.cash=935.5999999999999)`: 1
  - `echo unparseable for 'cash': '$781.5999999999999' (STATE.cash=781.5999999999999)`: 1
  - `echo unparseable for 'cash': '$365.2' (STATE.cash=365.2)`: 1
  - `echo unparseable for 'cash': '$479.0' (STATE.cash=479.0)`: 1
  - `echo unparseable for 'cash': '$351.2' (STATE.cash=351.2)`: 1
  - `echo unparseable for 'cash': '$447.0' (STATE.cash=447.0)`: 1

### Hallucinated reason examples

- cash=$388.0, cost=$156, self=1/3, opp=0 — **original** _you_own_in_group=1 and group_size=3 so I am one short; this purchase completes my Pink monopoly._
  - **final retry** (still flagged, 4 retries): _I have $1000, cost is only $100, and opp_own_in_group is 0 so I can still monopolise thisroperty._
- cash=$861.0, cost=$344, self=0/3, opp=1 — **original** _you_own_in_group=0 and group_size=3 so I can monopolise this group._
  - **final retry** (still flagged, 4 retries): _<no REASON>_
- cash=$1885.0, cost=$58, self=1/2, opp=0 — **original** _you_own_in_group=1 and group_size=2 so I am one short; this purchase completes my Brown monopoly._
  - **final retry** (still flagged, 4 retries): _opp_own_in_group is 0 of 2 so opponent already controls most of Brown; spending $58 hurts liquidity for no monopoly path._
- cash=$944.0, cost=$328, self=1/3, opp=1 — **original** _you_own_in_group=1 and group_size=3 so I am one short; this purchase completes my Orange monopoly._
  - **final retry** (still flagged, 4 retries): _you_own_in_group=1 and group_size=3 so I am one short; this purchase completes my Purple monopoly._
- cash=$665.0, cost=$200, self=3/4, opp=0 — **original** _you_own_in_group=3 and group_size=4 so I am one short; this purchase completes my Railroads monopoly._
  - **final retry** (still flagged, 4 retries): _I have $1000, cost is only $100, and opp_own_in_group is 0 so I can still monopolise this group._
- cash=$718.0, cost=$480, self=1/2, opp=0 — **original** _you_own_in_group=1 and group_size=2 so I am one short; this purchase completes my Red monopoly._
  - **final retry** (still flagged, 4 retries): _I have $1000, cost is only $100, and opp_own_in_group is 0 so I can still monopolise thisroperty._
- cash=$429.4, cost=$58, self=0/2, opp=1 — **original** _opp_own_in_group is 1 of 2 so opponent already controls most of Brown; spending $58 hurts liquidity for no monopoly path._
  - **final retry** (still flagged, 4 retries): _opp_own_in_group is 1 of 2 so opponent already controls most of Brown; spending $58 hurts liquidity for no monopoly path._
- cash=$429.2, cost=$200, self=0/4, opp=2 — **original** _opp_own_in_group is 2 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._
  - **final retry** (still flagged, 4 retries): _opp_own_in_group is 2 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._
- cash=$1419.6, cost=$333, self=0/1, opp=0 — **original** _you_own_in_group=0 and group_size=1 so I can monopolise this group._
  - **final retry** (still flagged, 4 retries): _I have $1419.6, cost is $333, and opp_own_in_group is 0 so I can monopolise this group._
- cash=$446.5999999999999, cost=$200, self=2/4, opp=0 — **original** _you_own_in_group=2 and group_size=4 so I am one short; this purchase completes my Railroads monopoly._
  - **final retry** (still flagged, 4 retries): _<no REASON>_

## Buy rate by colour group

| group | n | n_buy | buy_rate |
|-------|--:|------:|---------:|
| Railroads | 82 | 61 | 0.744 |
| Orange | 75 | 42 | 0.560 |
| Pink | 68 | 39 | 0.574 |
| Brown | 61 | 34 | 0.557 |
| Utilities | 41 | 35 | 0.854 |
| Yellow | 36 | 23 | 0.639 |
| Red | 34 | 27 | 0.794 |
| Green | 19 | 18 | 0.947 |
| Indigo | 16 | 16 | 1.000 |
| Lightblue | 16 | 16 | 1.000 |

**Buy rate by cash bucket**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| 200-500 | 64 | 31 | 0.484 |
| 500-1000 | 138 | 95 | 0.688 |
| 1000-1500 | 145 | 112 | 0.772 |
| >=1500 | 101 | 73 | 0.723 |

**Buy rate by monopoly opportunity**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| fresh | 205 | 195 | 0.951 |
| partial_self | 93 | 91 | 0.978 |
| opponent_dominates | 150 | 25 | 0.167 |

## Surprising decisions (qualitative excerpts)

### PASS while flush (cash >= 1000)

- cash=$3092, prop=`D2 Tennessee Avenue` (Orange group, $328), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Orange; spending $328 hurts liquidity for no monopoly path._
- cash=$2958, prop=`F2 Ventnor Avenue` (Yellow group, $447), self=0/2, opp=1 — reason: _opp_own_in_group is 1 of 2 so opponent already controls most of Yellow; spending $447 hurts liquidity for no monopoly path._
- cash=$1885.0, prop=`A1 Mediterranean Avenue` (Brown group, $58), self=1/2, opp=0 — reason: _you_own_in_group=1 and group_size=2 so I am one short; this purchase completes my Brown monopoly._
- cash=$2781, prop=`D2 Tennessee Avenue` (Orange group, $328), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Orange; spending $328 hurts liquidity for no monopoly path._
- cash=$2319, prop=`D2 Tennessee Avenue` (Orange group, $328), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Orange; spending $328 hurts liquidity for no monopoly path._
- cash=$1846, prop=`C3 Virginia Avenue` (Pink group, $156), self=0/3, opp=2 — reason: _opp_own_in_group is 2 of 3 so opponent already controls most of Pink; spending $156 hurts liquidity for no monopoly path._
- cash=$1846, prop=`D3 New York Avenue` (Orange group, $344), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Orange; spending $344 hurts liquidity for no monopoly path._
- cash=$1804, prop=`F2 Ventnor Avenue` (Yellow group, $447), self=0/2, opp=1 — reason: _opp_own_in_group is 1 of 2 so opponent already controls most of Yellow; spending $447 hurts liquidity for no monopoly path._
- cash=$1701, prop=`C2 States Avenue` (Pink group, $198), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Pink; spending $198 hurts liquidity for no monopoly path._
- cash=$1635, prop=`A2 Baltic Avenue` (Brown group, $54), self=0/2, opp=1 — reason: _opp_own_in_group is 1 of 2 so opponent already controls most of Brown; spending $54 hurts liquidity for no monopoly path._

### BUY at cash floor (cash < 300)

_None._
