# LLM-decision analysis: `default`

- games: **1** (truncated: 0)
- mean rounds: **61.0**, mean transfer rate: **49.8** $/round
- winners: `LLM_p1`=1

## Decisions

- total decisions logged: **42**
- prefilter PASSes: `cant_afford`=6, `cash_floor`=4 (total 10)
- LLM calls: **32**, buy rate: **0.812**
- LLM call latency: median 2216 ms, mean 2206 ms, max 2290 ms
- parse-path distribution: `first_answer_tag`=31, `last_token_fallback`=1

## Hallucination detector

- LLM calls flagged as contradicting STATE: **6 / 32** (18.8%)
- retries attempted: **6**, of which **0** (0%) cleared on the second try
- issue labels:
  - `reason claims monopoly completion but you are not one short`: 6

### Hallucinated reason examples

- cash=$992, cost=$200, self=1/3, opp=0 — **original** _you_own_in_group is 1 of 3 so this completes my Orange monopoly — highest-value buy available._
  - **retry** (still flagged): _you_own_in_group is 1 of 3 so this completes my Orange monopoly — highest-value buy available._
- cash=$814, cost=$120, self=1/3, opp=0 — **original** _you_own_in_group is 1 of 3 so this completes my Lightblue monopoly — highest-value buy available._
  - **retry** (still flagged): _you_own_in_group is 1 of 3 so this completes my Lightblue monopoly — highest-value buy available._
- cash=$633, cost=$300, self=1/3, opp=0 — **original** _you_own_in_group is 1 of 3 so this completes my Green monopoly — highest-value buy available._
  - **retry** (still flagged): _you_own_in_group is 1 of 3 so this completes my Green monopoly — highest-value buy available._
- cash=$998, cost=$220, self=1/3, opp=0 — **original** _you_own_in_group is 1 of 3 so this completes my Red monopoly — highest-value buy available._
  - **retry** (still flagged): _you_own_in_group is 1 of 3 so this completes my Red monopoly — highest-value buy available._
- cash=$1485, cost=$140, self=1/3, opp=1 — **original** _you_own_in_group is 1 of 3 so this completes my Pink monopoly — highest-value buy available._
  - **retry** (still flagged): _you_own_in_group is 1 of 3 so this completes my Pink monopoly — highest-value buy available._
- cash=$1272, cost=$280, self=1/3, opp=1 — **original** _you_own_in_group is 1 of 3 so this completes my Yellow monopoly — highest-value buy available._
  - **retry** (still flagged): _you_own_in_group is 1 of 3 so this completes my Yellow monopoly — highest-value buy available._

## Buy rate by colour group

| group | n | n_buy | buy_rate |
|-------|--:|------:|---------:|
| Green | 6 | 2 | 0.333 |
| Railroads | 4 | 4 | 1.000 |
| Red | 4 | 3 | 0.750 |
| Lightblue | 3 | 3 | 1.000 |
| Pink | 3 | 3 | 1.000 |
| Orange | 3 | 3 | 1.000 |
| Yellow | 3 | 3 | 1.000 |
| Utilities | 2 | 2 | 1.000 |
| Indigo | 2 | 1 | 0.500 |
| Brown | 2 | 2 | 1.000 |

**Buy rate by cash bucket**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| 200-500 | 1 | 1 | 1.000 |
| 500-1000 | 17 | 12 | 0.706 |
| 1000-1500 | 12 | 11 | 0.917 |
| >=1500 | 2 | 2 | 1.000 |

**Buy rate by monopoly opportunity**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| fresh | 13 | 11 | 0.846 |
| partial_self | 9 | 9 | 1.000 |
| opponent_dominates | 10 | 6 | 0.600 |

## Surprising decisions (qualitative excerpts)

### PASS while flush (cash >= 1000)

- cash=$1085, prop=`G3 Pennsylvania Avenue` (Green group, $320), self=0/3, opp=2 — reason: _you_own_in_group is 2 of 3 so opponent already controls most of Green; spending $320 hurts liquidity for no monopoly path._

### BUY at cash floor (cash < 300)

_None._
