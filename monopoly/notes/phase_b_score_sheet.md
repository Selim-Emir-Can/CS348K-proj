# Phase B: human playtest — printable score sheet

Print one of these per game. The same statistics appear on the agent-side
JSONL output, so post-game cross-comparison is direct.

---

## Game header

```
Date: ____________________      Game number (this session): ___

Board:    [ ] default     [ ] salary x2     [ ] drop Brown     [ ] orange rent x2

Player A name: ______________________  archetype self-classification (circle one):
  AggressiveBuilder   CashHoarder   Trader   RailroadKing   LowCostOnly
  HighCostOnly        Balanced      Passive  Bully          RiskAverse

Player B name: ______________________  archetype self-classification (circle one):
  AggressiveBuilder   CashHoarder   Trader   RailroadKing   LowCostOnly
  HighCostOnly        Balanced      Passive  Bully          RiskAverse

Wall-clock start: ______:______       Wall-clock end: ______:______
```

## Per-game outcome

```
Outcome (circle one):
   [ ] Player A won by bankrupting Player B   at turn ______
   [ ] Player B won by bankrupting Player A   at turn ______
   [ ] Game truncated at 90 minutes / 200 turns (DRAW)

Final cash:        Player A $______      Player B $______
Final net worth:   Player A $______      Player B $______
First monopoly completed at turn ______ by player [ ] A  [ ] B  [ ] none

Total turns played: ______
```

## Money-transfer log (rough)

Tally a tick for every rent payment between players ($1 per tick of effort):

```
Rent paid by A to B:   |||| |||| |||| ...    total $______
Rent paid by B to A:   |||| |||| |||| ...    total $______
                                              total transfer $______
Mean transfer per turn = total transfer / total turns = $______
```

## Likert ratings (post-game)

```
Player A rates this game:
  Fairness    1 (very unfair)  2  3  4  5 (perfectly fair)
  Length      1 (way too long) 2  3  4  5 (just right)  6  7 (way too short)
  Engagement  1 (boring)       2  3  4  5 (very engaging)

Player B rates this game:
  Fairness    1  2  3  4  5
  Length      1  2  3  4  5  6  7
  Engagement  1  2  3  4  5
```

## Free-form comments

```
What stood out about this game?
__________________________________________________________________________
__________________________________________________________________________

If you had to change one thing about this board, what would it be?
__________________________________________________________________________
__________________________________________________________________________
```

---

## Tester instructions (read aloud once at session start)

1. You will play 3-5 games of Monopoly on different board variants. Each
   game lasts ~10 minutes (90-minute hard cap if it drags).
2. We are studying the *board itself*, not your play style. Play however
   feels natural. Self-classify your style up front; do not change it
   between games.
3. After each game, fill out the score sheet honestly. The numeric
   tallies (turns, rent, final cash) are more important than the Likert
   ratings, so do those first.
4. Do not look at the agent-side predictions while you play. Those are
   collected at session end.
