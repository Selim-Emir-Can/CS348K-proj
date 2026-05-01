# Combined LLM-decision analysis

## default

# LLM-decision analysis: `default`

- games: **1** (truncated: 0)
- mean rounds: **46.0**, mean transfer rate: **74.5** $/round
- winners: `LLM_p1`=1

## Decisions

- total decisions logged: **38**
- prefilter PASSes: `cash_floor`=6, `cant_afford`=4 (total 10)
- LLM calls: **28**, buy rate: **0.893**
- LLM call latency: median 2203 ms, mean 2211 ms, max 2285 ms
- parse-path distribution: `first_answer_tag`=28

## Hallucination detector

- LLM calls flagged as contradicting STATE: **4 / 28** (14.3%)
- retries attempted: **4**, of which **4** (100%) cleared on the second try
- issue labels:
  - `reason claims monopoly completion but you are not one short`: 4

### Hallucinated reason examples

- cash=$592, cost=$200, self=1/3, opp=0 — **original** _you_own_in_group=1 and group_size=3 so I am one short; this purchase completes my Orange monopoly._
  - **retry** (resolved): _I have $592, cost is $200, and you_own_in_group is 1 of 3 so I can still monopolise this group._
- cash=$674, cost=$120, self=1/3, opp=0 — **original** _you_own_in_group=1 and group_size=3 so I am one short; this purchase completes my Lightblue monopoly._
  - **retry** (resolved): _I have $674, cost is $120, and you_own_in_group is 1 so I can still monopolise this group._
- cash=$602, cost=$240, self=1/3, opp=0 — **original** _you_own_in_group=1 and group_size=3 so I am one short; this purchase completes my Red monopoly._
  - **retry** (resolved): _I have $602, cost is $240, and you_own_in_group is 1 of 3 so I can still monopolise this group._
- cash=$1613, cost=$140, self=1/3, opp=1 — **original** _you_own_in_group=1 and group_size=3 so I am one short; this purchase completes my Pink monopoly._
  - **retry** (resolved): _I have $1613, cost is only $140, and you_own_in_group is 1 so I can still monopolise this group._

## Buy rate by colour group

| group | n | n_buy | buy_rate |
|-------|--:|------:|---------:|
| Railroads | 4 | 4 | 1.000 |
| Red | 4 | 3 | 0.750 |
| Lightblue | 3 | 3 | 1.000 |
| Pink | 3 | 3 | 1.000 |
| Orange | 3 | 3 | 1.000 |
| Indigo | 3 | 2 | 0.667 |
| Green | 3 | 2 | 0.667 |
| Utilities | 2 | 2 | 1.000 |
| Brown | 2 | 2 | 1.000 |
| Yellow | 1 | 1 | 1.000 |

**Buy rate by cash bucket**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| 200-500 | 3 | 2 | 0.667 |
| 500-1000 | 14 | 12 | 0.857 |
| 1000-1500 | 6 | 6 | 1.000 |
| >=1500 | 5 | 5 | 1.000 |

**Buy rate by monopoly opportunity**

| bucket | n_calls | n_buy | buy_rate |
|--------|--------:|------:|---------:|
| fresh | 11 | 11 | 1.000 |
| partial_self | 9 | 9 | 1.000 |
| opponent_dominates | 8 | 5 | 0.625 |

## Surprising decisions (qualitative excerpts)

### PASS while flush (cash >= 1000)

_None._

### BUY at cash floor (cash < 300)

_None._

