# LLM-decision analysis: `llm_ga_2p_winner`

- games: **5** (truncated: 0)
- mean rounds: **43.6**, mean transfer rate: **0.0** $/round
- winners: `LLM_p1`=3, `LLM_p0`=2

## Decisions

- total decisions logged: **148**
- prefilter PASSes: `cant_afford`=43, `cash_floor`=31 (total 74)
- LLM calls: **74**, buy rate: **0.824**
- LLM call latency: median 8171 ms, mean 8173 ms, max 8446 ms
- parse-path distribution: `first_answer_tag`=74

## Hallucination detector

- LLM calls flagged by the validator: **0 / 74** (0.0%)
- of those, **real** model-side hallucinations (post 2026-04-29 reclassification): **0 / 74** (0.0%)
- and **spurious** validator-bug flags from float-drift on Player.money: **0**
- retries attempted (any): **0**, of which **0** (0%) cleared by the LAST attempt
- total retry calls across all decisions: **0** (avg 0.00 retries per LLM-call)
- decisions still flagged after MAX_RETRIES: **0** (of which **0** real, **0** spurious)

## Buy rate by colour group

| group | n | n_buy | buy_rate |
|-------|--:|------:|---------:|
| Railroads | 23 | 15 | 0.652 |
| Red | 11 | 8 | 0.727 |
| Utilities | 9 | 9 | 1.000 |
| Brown | 9 | 7 | 0.778 |
| Pink | 5 | 5 | 1.000 |
| Lightblue | 5 | 5 | 1.000 |
| Orange | 4 | 4 | 1.000 |
| Green | 4 | 4 | 1.000 |
| Indigo | 4 | 4 | 1.000 |

**Buy rate by cash bucket**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| 200-500 | 11 | 10 | 0.909 |
| 500-1000 | 34 | 25 | 0.735 |
| 1000-1500 | 17 | 14 | 0.824 |
| >=1500 | 12 | 12 | 1.000 |

**Buy rate by monopoly opportunity**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| fresh | 42 | 42 | 1.000 |
| partial_self | 17 | 17 | 1.000 |
| opponent_dominates | 15 | 2 | 0.133 |

## Surprising decisions (qualitative excerpts)

### PASS while flush (cash >= 1000)

- cash=$1442, prop=`E1 Kentucky Avenue` (Red group, $345), self=0/3, opp=1 — reason: _opp_own_in_group is 1 of 3 so opponent already controls most of Red; spending $345 hurts liquidity for no monopoly path._
- cash=$1385, prop=`R3 B&O Railroad` (Railroads group, $200), self=0/4, opp=2 — reason: _opp_own_in_group is 2 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._
- cash=$1215.6, prop=`R4 Short Line` (Railroads group, $200), self=0/4, opp=2 — reason: _opp_own_in_group is 2 of 4 so opponent already controls most of Railroads; spending $200 hurts liquidity for no monopoly path._

### BUY at cash floor (cash < 300)

- cash=$291, prop=`A2 Baltic Avenue` (Brown group, $50), self=0/2, opp=0 — reason: _cash is $291 and cost is $50 so I can afford this._
