"""Post-run timing summary for the round-1 overnight artifact.

Reads `timing.jsonl` written by `optimizer.timing.WallClockLogger`, emits
`timing_report.md`. Designed to be a paper artifact: the table of measured
per-phase totals replaces the Apr-27 estimates in the Compute section.

Usage:
    python scripts/summarize_round1_timing.py \\
        --in report/figures/round1/timing.jsonl \\
        --out report/figures/round1/timing_report.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path) -> List[dict]:
    out: List[dict] = []
    if not path.exists():
        return out
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _per_phase_totals(records: List[dict]) -> Dict[str, Dict[str, float]]:
    """Per-phase aggregate from phase_start/phase_end pairs."""
    starts: Dict[str, float] = {}
    out: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {'wall_seconds': 0.0, 'iters': 0, 'llm_calls': 0,
                 'eval_seconds': 0.0, 'bootstrap_seconds': 0.0,
                 'gen_seconds': 0.0, 'gen_tokens': 0})
    for r in records:
        p = r.get('phase')
        if not p:
            continue
        if r.get('event') == 'phase_start':
            starts[p] = r.get('elapsed_total_s', 0.0)
        elif r.get('event') == 'phase_end':
            wall = r.get('elapsed_total_s', 0.0) - starts.get(p, 0.0)
            out[p]['wall_seconds'] += float(wall)
        elif r.get('event') in ('iter_end', 'iter__end'):
            out[p]['iters'] += 1
        elif r.get('event') in ('llm_call__end', 'llm_call'):
            out[p]['llm_calls'] += 1
            out[p]['gen_seconds'] += float(r.get('seconds', r.get('gen_seconds', 0.0)) or 0.0)
            out[p]['gen_tokens'] += int(r.get('gen_tokens', 0) or 0)
        elif r.get('event') in ('eval__end', 'eval'):
            out[p]['eval_seconds'] += float(r.get('seconds', r.get('eval_seconds', 0.0)) or 0.0)
        elif r.get('event') in ('bootstrap__end', 'bootstrap'):
            out[p]['bootstrap_seconds'] += float(
                r.get('seconds', r.get('bootstrap_seconds', 0.0)) or 0.0)
    return dict(out)


def _llm_call_latencies(records: List[dict], instrument: str) -> List[float]:
    """All llm_call durations for a given instrument label."""
    out: List[float] = []
    for r in records:
        if r.get('event') in ('llm_call__end', 'llm_call'):
            if r.get('instrument') == instrument:
                v = r.get('seconds') or r.get('gen_seconds')
                if v is not None:
                    out.append(float(v))
    return out


def _hist_lines(samples: List[float], bins: int = 8) -> List[str]:
    if not samples:
        return ['(no samples)']
    s = sorted(samples)
    lo, hi = s[0], s[-1]
    if hi <= lo:
        return [f'{lo:.3f} s × {len(samples)}']
    step = (hi - lo) / bins
    edges = [lo + step * i for i in range(bins + 1)]
    counts = [0] * bins
    for x in samples:
        idx = min(int((x - lo) / step), bins - 1) if step > 0 else 0
        counts[idx] += 1
    out = []
    width_total = 36
    cmax = max(counts) or 1
    for i in range(bins):
        bar = '#' * int(counts[i] / cmax * width_total)
        out.append(f'  {edges[i]:6.2f}–{edges[i+1]:6.2f} s | {bar:<{width_total}} {counts[i]}')
    return out


def _longest_iters(records: List[dict], k: int = 10) -> List[dict]:
    iters = [r for r in records
             if r.get('event') in ('iter__end', 'iter_end')]
    iters.sort(key=lambda r: r.get('seconds') or 0.0, reverse=True)
    return iters[:k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in',  dest='in_path',
                    default='report/figures/round1/timing.jsonl')
    ap.add_argument('--out', dest='out_path',
                    default='report/figures/round1/timing_report.md')
    args = ap.parse_args()

    records = _load(Path(args.in_path))
    lines: List[str] = ['# Round 1 — Timing Report',
                        '',
                        f'_source: `{args.in_path}`  '
                        f'(records read: {len(records)})_',
                        '']

    totals = _per_phase_totals(records)
    lines.append('## Per-phase totals')
    lines.append('')
    lines.append('| phase | wall(s) | iters | llm_calls | gen_s | eval_s | boot_s | tokens |')
    lines.append('|-------|--------:|------:|----------:|------:|-------:|-------:|-------:|')
    for p in sorted(totals.keys()):
        t = totals[p]
        lines.append(f'| {p} | {t["wall_seconds"]:.1f} | {t["iters"]} | '
                     f'{t["llm_calls"]} | {t["gen_seconds"]:.1f} | '
                     f'{t["eval_seconds"]:.1f} | {t["bootstrap_seconds"]:.1f} | '
                     f'{t["gen_tokens"]} |')
    lines.append('')

    for instrument, label in (('NEUTRAL-PLAYER', 'NEUTRAL-PLAYER'),
                              ('DIRECTIVE-DESIGNER', 'DIRECTIVE-DESIGNER')):
        lat = _llm_call_latencies(records, instrument)
        lines.append(f'## {label} llm_call latency (n={len(lat)})')
        lines.append('')
        lines.extend(_hist_lines(lat))
        if lat:
            mean = sum(lat) / len(lat); med = sorted(lat)[len(lat) // 2]
            lines.append('')
            lines.append(f'  mean={mean:.3f}s  median={med:.3f}s  '
                         f'min={min(lat):.3f}s  max={max(lat):.3f}s')
        lines.append('')

    longest = _longest_iters(records, k=10)
    if longest:
        lines.append('## Longest 10 iterations')
        lines.append('')
        lines.append('| seconds | phase | board | seed | iter |')
        lines.append('|--------:|-------|-------|-----:|-----:|')
        for r in longest:
            lines.append(f'| {(r.get("seconds") or 0.0):.2f} | '
                         f'{r.get("phase","?")} | '
                         f'{r.get("board","?")} | {r.get("seed","?")} | '
                         f'{r.get("iter","?")} |')
        lines.append('')

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
