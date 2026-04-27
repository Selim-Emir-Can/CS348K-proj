# LLM rule-mutation loop: v1 vs v2 invocation

The CEO plan splits #5 into two stages so v1 is a falsification-safe
deliverable before v2 takes any wall-clock budget. Both stages run from
the same script (`scripts/llm_rule_loop.py`); the only operational
difference is the `--rule-cap` flag and the output `--variant` subdir.

## v1 (capped at 2 patches per iteration)

```
set ANTHROPIC_API_KEY=...
set PYTHONPATH=. && python scripts/llm_rule_loop.py \
    --backend anthropic \
    --model claude-sonnet-4-6 \
    --rule-cap 2 \
    --variant v1 \
    --n-seeds 3 --K 6 --n-games 100 \
    --max-wall-seconds 21600 \
    --out-dir report/figures/llm_rule_loop
```

Outputs:
- `.../v1/llm/<board>__seed<S>.jsonl`     -- LLM trajectories
- `.../v1/random/<board>__seed<S>.jsonl`  -- random rule-menu baseline
- `.../v1/house/<board>.json`             -- house-rule static ceiling
- `.../v1/goodhart_audit.md`              -- top-3 review surface
- `report/figures/llm_rules/rejected_corpus.jsonl` -- cumulative rejection log (CEO #8)

## v2 (uncapped, run only after v1 lands)

```
set PYTHONPATH=. && python scripts/llm_rule_loop.py \
    --backend anthropic \
    --model claude-sonnet-4-6 \
    --rule-cap 0 \
    --variant v2 \
    --n-seeds 3 --K 6 --n-games 100 \
    --max-wall-seconds 21600 \
    --out-dir report/figures/llm_rule_loop
```

`--rule-cap 0` is interpreted as "no cap" (the validator's
`if rule_cap and len(patches) > rule_cap` short-circuits). The system
prompt is updated per iteration to say `RULE CAP THIS ITERATION: UNCAPPED`.

Per the CEO plan, **v2 must only start after v1 has produced a complete
deliverable trajectory.** The 6h wall-clock cap (`--max-wall-seconds`) is
honored independently for each stage so a stuck v2 can't eat v1's slot.

## Cost estimate

Per the CEO compute estimate, both v1 and v2 together fit inside ~1-2 hr
wall-clock plus ~$5-30 of Sonnet API time. The dev burn was ~5-10x that
during build (estimated $50-300 total per CEO C10).
