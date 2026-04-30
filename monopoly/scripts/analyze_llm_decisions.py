"""Post-hoc analysis of an eval_llm_on_boards.py run.

Reads ``logs/llm_eval/<run>/decisions/<board_tag>/seed_*.jsonl`` and the
top-level ``summary.csv`` produced by the driver, and emits one
``analysis.md`` per board tag plus a top-level ``analysis_combined.md``.

Float-drift reclassification (added 2026-04-29):
The pre-2026-04-29T11 echo validator parsed numeric STATE fields with
``int()`` and treated values like ``$411.79999999999995`` (legitimate
float drift in ``Player.money``) as ``echo unparseable``. The model
itself echoed those values *correctly* — it just copied STATE
verbatim. The Task 1 80-game logs contain a few hundred of these
spurious "hallucination" flags. This analyser detects them by
re-parsing the original mismatch string with ``float()`` + a $0.50
tolerance against STATE, and reclassifies them as
``validator_bug_float_drift`` so the report's hallucination headline
reflects the actual model behaviour rather than the validator's
limitations.

Metrics reported per board:
  - per-game aggregates (mean rounds, draw rate, mean transfer rate)
  - decision counts split into prefilter PASSes vs LLM calls
  - LLM-call buy rate, sliced by:
        - colour group
        - cash bucket [<200, 200-500, 500-1000, 1000-1500, >1500]
        - monopoly opportunity {fresh, partial_self, opponent_dominates}
  - parse-path distribution (first_answer_tag / last_token_fallback / default_buy)
  - top-K "surprising" decisions (PASS while flush + monopoly-completing,
    or BUY at cash floor) with the REASON text — these are the qualitative
    excerpts that go into the report appendix.

Usage (run from monopoly/):
    python scripts/analyze_llm_decisions.py --in logs/llm_eval/run1
"""
import argparse
import csv
import json
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean


# --------------------------------------------------------------------------- #
# Bucketing helpers                                                             #
# --------------------------------------------------------------------------- #

_CASH_BUCKETS = [
    ('<200',       lambda c: c < 200),
    ('200-500',    lambda c: 200 <= c < 500),
    ('500-1000',   lambda c: 500 <= c < 1000),
    ('1000-1500',  lambda c: 1000 <= c < 1500),
    ('>=1500',     lambda c: c >= 1500),
]


def _cash_bucket(c: int) -> str:
    for name, pred in _CASH_BUCKETS:
        if pred(c):
            return name
    return 'unknown'


def _monopoly_opportunity(rec: dict) -> str:
    """Categorise the monopoly-completion situation at decision time.

    'opponent_dominates' takes precedence over 'partial_self' to reflect the
    strategic reality that a property is rarely worth buying when the opponent
    already controls more of the group than you do.
    """
    g = rec.get('group_size') or 0
    own = rec.get('same_group_self') or 0
    opp = rec.get('opp_in_group') or 0
    if g == 0:
        return 'unknown'
    # Define opponent-dominates as: opponent owns at least half the group
    # AND at least as many as you.
    if opp >= max(1, g // 2) and opp >= own:
        return 'opponent_dominates'
    if own > 0:
        return 'partial_self'
    return 'fresh'


_UNPARSEABLE_RE = __import__('re').compile(
    r"echo unparseable for '([a-z_]+)': '([^']*)' \(STATE\.\1=([^)]+)\)"
)
_MISMATCH_RE = __import__('re').compile(
    r"echo mismatch on '([a-z_]+)': model echoed (.+?), STATE\.\1=(.+)"
)


def _classify_issue(issue: str) -> str:
    """Reclassify a pre-2026-04-29T11 issue string.

    Returns one of:
      'real'                       — actual mismatch the model got wrong
      'validator_bug_float_drift'  — model echoed STATE correctly but the
                                     old validator tripped on float
                                     parsing (STATE itself contained a
                                     float like 411.79999999999995)
      'unparseable_other'          — value really doesn't parse as a number;
                                     keep as a real flag

    The reclassification is purely string-based — it inspects the
    original issue text the validator wrote into the JSONL, not the
    raw response. This means it works on any historical run without
    needing to re-tokenise.
    """
    m = _UNPARSEABLE_RE.match(issue)
    if m:
        _field, model_str, state_str = m.group(1), m.group(2), m.group(3)
        try:
            mv = float(model_str.lstrip('$').replace('$', '').strip())
            sv = float(state_str.strip())
            if abs(mv - sv) <= 0.5:
                return 'validator_bug_float_drift'
            return 'real'
        except ValueError:
            return 'unparseable_other'
    m = _MISMATCH_RE.match(issue)
    if m:
        _field, model_str, state_str = m.group(1), m.group(2), m.group(3)
        # Even fully-formed mismatches can be float-drift if both parse
        # numerically and round-match.
        try:
            mv = float(model_str.lstrip("'\"").rstrip("'\"").lstrip('$').replace('$', '').strip())
            sv = float(state_str.lstrip("'\"").rstrip("'\"").strip())
            if abs(mv - sv) <= 0.5:
                return 'validator_bug_float_drift'
        except ValueError:
            pass
        return 'real'
    return 'real'


def _bucket_buy_rate(records, key_fn):
    """Return {bucket -> (n, n_buy, buy_rate)} for the LLM-decided records."""
    by_bucket = defaultdict(lambda: [0, 0])
    for r in records:
        if r.get('prefilter') != 'sent_to_llm':
            continue
        k = key_fn(r)
        by_bucket[k][0] += 1
        if r.get('parsed') == 'BUY':
            by_bucket[k][1] += 1
    return {k: (n, b, (b / n) if n else 0.0) for k, (n, b) in by_bucket.items()}


# --------------------------------------------------------------------------- #
# IO                                                                            #
# --------------------------------------------------------------------------- #

def _read_decisions(decisions_dir: Path):
    """Yield every decision record under decisions_dir/seed_*.jsonl."""
    if not decisions_dir.exists():
        return
    for log_path in sorted(decisions_dir.glob('seed_*.jsonl')):
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _read_summary(summary_path: Path):
    rows = []
    if not summary_path.exists():
        return rows
    with open(summary_path, 'r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Reporting                                                                     #
# --------------------------------------------------------------------------- #

def _format_bucket_table(title: str, by_bucket: dict, order=None):
    keys = order if order else sorted(by_bucket.keys())
    lines = [f'**{title}**', '', '| bucket | n_calls | n_buy | buy_rate |',
             '|--------|--------:|------:|---------:|']
    for k in keys:
        if k not in by_bucket:
            continue
        n, b, r = by_bucket[k]
        lines.append(f'| {k} | {n} | {b} | {r:.3f} |')
    return '\n'.join(lines) + '\n'


def _surprising(records, k=10):
    """Pick decisions that are interesting to read by hand."""
    flush_passes = []
    poor_buys = []
    for r in records:
        if r.get('prefilter') != 'sent_to_llm':
            continue
        cash = r.get('cash', 0)
        parsed = r.get('parsed')
        if parsed == 'PASS' and cash >= 1000:
            # Especially suspicious if we own some of the group already.
            score = (1 if (r.get('same_group_self') or 0) > 0 else 0) + (cash / 1000.0)
            flush_passes.append((score, r))
        if parsed == 'BUY' and cash < 300:
            poor_buys.append((300 - cash, r))
    flush_passes.sort(key=lambda x: -x[0])
    poor_buys.sort(key=lambda x: -x[0])
    return flush_passes[:k], poor_buys[:k]


def _analyse_board(board_tag: str, records, summary_rows, out_md_path: Path):
    """Write one analysis.md for a single board tag."""
    summary_rows = [r for r in summary_rows if r['board_tag'] == board_tag]
    n_games = len(summary_rows)
    if n_games == 0:
        out_md_path.write_text(f'# Analysis: {board_tag}\n\nNo games found.\n',
                               encoding='utf-8')
        return

    # Per-game aggregates
    rounds = [int(r['rounds']) for r in summary_rows]
    transfer_rates = [float(r['transfer_rate']) for r in summary_rows]
    n_truncated = sum(1 for r in summary_rows if int(r['truncated']))
    winners = Counter(r['winner'] for r in summary_rows)

    # Per-decision aggregates
    n_dec = len(records)
    n_prefilter = Counter()
    n_llm_calls = 0
    n_llm_buy = 0
    parse_path_counts = Counter()
    buy_by_group = defaultdict(lambda: [0, 0])
    reason_lengths = []
    ms_elapsed_list = []
    # Hallucination tallies (only populated for runs whose LLMPlayer wrote
    # the post-2026-04-28 detector fields; older runs leave these at 0).
    n_hallucinated = 0
    n_retried = 0
    n_retry_resolved = 0
    issue_label_counts = Counter()
    hallucinated_examples = []
    # Echo-validation tallies (populated by the post-2026-04-28T18 logger
    # when the response includes an ECHO block). Counts per-field
    # mismatches across all *first attempts*.
    n_retries_total = 0
    n_with_attempts = 0
    n_unresolved_after_retries = 0
    field_mismatch_counts = Counter()
    # Post-2026-04-29T11 reclassification. The original validator counted
    # any "echo unparseable" issue as a hallucination, including legitimate
    # float drift in Player.money (e.g. STATE.cash=411.79999999999995 with
    # the model echoing $411.79999...). Reclassifying those lets us report
    # the true model-side hallucination rate.
    n_hallucinated_real     = 0    # at least one real issue on first attempt
    n_hallucinated_spurious = 0    # all issues on first attempt are validator bugs
    n_unresolved_real       = 0    # post-retry, still has a real issue
    n_unresolved_spurious   = 0    # post-retry, only validator-bug issues remain
    field_real_counts       = Counter()
    field_spurious_counts   = Counter()

    for r in records:
        pre = r.get('prefilter')
        if pre and pre != 'sent_to_llm':
            n_prefilter[pre] += 1
            continue
        n_llm_calls += 1
        if r.get('parsed') == 'BUY':
            n_llm_buy += 1
        parse_path_counts[r.get('parse_path') or 'unknown'] += 1
        g = r.get('prop_group') or 'unknown'
        buy_by_group[g][0] += 1
        if r.get('parsed') == 'BUY':
            buy_by_group[g][1] += 1
        rt = r.get('reason_text')
        if rt:
            reason_lengths.append(len(rt))
        ms = r.get('ms_elapsed')
        if isinstance(ms, (int, float)) and ms > 0:
            ms_elapsed_list.append(ms)
        if r.get('hallucination_detected'):
            n_hallucinated += 1
            for label in (r.get('hallucination_issues') or []):
                issue_label_counts[label] += 1
            if len(hallucinated_examples) < 10:
                hallucinated_examples.append(r)
        if r.get('retry_attempted'):
            n_retried += 1
            if r.get('retry_resolved'):
                n_retry_resolved += 1
        # Echo-validation per-field tallies.
        attempts = r.get('echo_attempts')
        if attempts:
            n_with_attempts += 1
            n_retries_total += int(r.get('n_retries') or 0)
            if not r.get('final_resolved'):
                n_unresolved_after_retries += 1
            # Count first-attempt mismatches by field, AND reclassify each
            # issue as real-vs-validator-bug.
            first = attempts[0] if attempts else {}
            first_issues = first.get('echo_mismatches') or []
            first_classifications = [_classify_issue(issue) for issue in first_issues]
            any_real_first = any(c != 'validator_bug_float_drift'
                                  for c in first_classifications)
            if first_issues:
                if any_real_first:
                    n_hallucinated_real += 1
                else:
                    n_hallucinated_spurious += 1
            for issue, kind in zip(first_issues, first_classifications):
                # Pull the field name with a small regex.
                import re as _re
                m = _re.search(r"on '([a-z_]+)'|field '([a-z_]+)'|for '([a-z_]+)'",
                                issue)
                field = (m.group(1) or m.group(2) or m.group(3)) if m else '<unparsed>'
                field_mismatch_counts[field] += 1
                if kind == 'validator_bug_float_drift':
                    field_spurious_counts[field] += 1
                else:
                    field_real_counts[field] += 1
            # Same reclassification for the FINAL attempt to compute the
            # post-retries "still flagged" stat correctly.
            if not r.get('final_resolved'):
                last_issues = (attempts[-1].get('echo_mismatches') or [])
                last_kinds = [_classify_issue(i) for i in last_issues]
                if any(k != 'validator_bug_float_drift' for k in last_kinds):
                    n_unresolved_real += 1
                else:
                    n_unresolved_spurious += 1

    by_cash = _bucket_buy_rate(records, lambda r: _cash_bucket(r.get('cash', 0)))
    by_op   = _bucket_buy_rate(records, _monopoly_opportunity)
    flush_passes, poor_buys = _surprising(records, k=10)

    md = []
    md.append(f'# LLM-decision analysis: `{board_tag}`')
    md.append('')
    md.append(f'- games: **{n_games}** (truncated: {n_truncated})')
    md.append(f'- mean rounds: **{mean(rounds):.1f}**, mean transfer rate: '
              f'**{mean(transfer_rates):.1f}** $/round')
    if winners:
        md.append('- winners: ' + ', '.join(f'`{w}`={c}' for w, c in
                                              winners.most_common()))
    md.append('')
    md.append(f'## Decisions')
    md.append('')
    md.append(f'- total decisions logged: **{n_dec}**')
    md.append(f'- prefilter PASSes: ' + ', '.join(
        f'`{k}`={v}' for k, v in n_prefilter.most_common()) +
        f' (total {sum(n_prefilter.values())})')
    md.append(f'- LLM calls: **{n_llm_calls}**, buy rate: '
              f'**{(n_llm_buy / n_llm_calls if n_llm_calls else 0.0):.3f}**')
    if ms_elapsed_list:
        md.append(f'- LLM call latency: median '
                  f'{sorted(ms_elapsed_list)[len(ms_elapsed_list)//2]:.0f} ms, '
                  f'mean {mean(ms_elapsed_list):.0f} ms, max '
                  f'{max(ms_elapsed_list):.0f} ms')
    md.append(f'- parse-path distribution: ' + ', '.join(
        f'`{k}`={v}' for k, v in parse_path_counts.most_common()))
    md.append('')

    md.append('## Hallucination detector')
    md.append('')
    if n_llm_calls == 0:
        md.append('_No LLM calls in this run._')
    else:
        rate_raw  = n_hallucinated / n_llm_calls
        rate_real = n_hallucinated_real / n_llm_calls
        md.append(f'- LLM calls flagged by the validator: '
                  f'**{n_hallucinated} / {n_llm_calls}** ({rate_raw*100:.1f}%)')
        md.append(f'- of those, **real** model-side hallucinations '
                  f'(post 2026-04-29 reclassification): '
                  f'**{n_hallucinated_real} / {n_llm_calls}** '
                  f'({rate_real*100:.1f}%)')
        md.append(f'- and **spurious** validator-bug flags from float-drift '
                  f'on Player.money: **{n_hallucinated_spurious}**')
        md.append(f'- retries attempted (any): **{n_retried}**, of which '
                  f'**{n_retry_resolved}** ({(n_retry_resolved/n_retried*100) if n_retried else 0.0:.0f}%) '
                  f'cleared by the LAST attempt')
        if n_with_attempts:
            md.append(f'- total retry calls across all decisions: '
                      f'**{n_retries_total}** '
                      f'(avg {n_retries_total/n_with_attempts:.2f} retries per LLM-call)')
            md.append(f'- decisions still flagged after MAX_RETRIES: '
                      f'**{n_unresolved_after_retries}** '
                      f'(of which **{n_unresolved_real}** real, '
                      f'**{n_unresolved_spurious}** spurious)')
        if field_real_counts:
            md.append('- per-field first-attempt mismatch counts (real):')
            for k, v in field_real_counts.most_common():
                md.append(f'  - `{k}`: {v}')
        if field_spurious_counts:
            md.append('- per-field first-attempt mismatch counts (spurious — validator bug):')
            for k, v in field_spurious_counts.most_common():
                md.append(f'  - `{k}`: {v}')
        if issue_label_counts:
            md.append('- legacy issue labels (regex detector, kept for back-compat):')
            for k, v in issue_label_counts.most_common():
                md.append(f'  - `{k}`: {v}')
        if hallucinated_examples:
            md.append('')
            md.append('### Hallucinated reason examples')
            md.append('')
            for r in hallucinated_examples:
                md.append(
                    f'- cash=${r.get("cash")}, '
                    f'cost=${r.get("prop_cost")}, '
                    f'self={r.get("same_group_self")}/{r.get("group_size")}, '
                    f'opp={r.get("opp_in_group")} — '
                    f'**original** _{r.get("reason_text") or "<no REASON>"}_'
                )
                if r.get('retry_attempted'):
                    md.append(
                        f'  - **final retry** ({"resolved" if r.get("retry_resolved") else "still flagged"}, '
                        f'{r.get("n_retries")} retries): '
                        f'_{r.get("retry_reason_text") or "<no REASON>"}_'
                    )
    md.append('')

    md.append('## Buy rate by colour group')
    md.append('')
    md.append('| group | n | n_buy | buy_rate |')
    md.append('|-------|--:|------:|---------:|')
    for g, (n, b) in sorted(buy_by_group.items(), key=lambda kv: -kv[1][0]):
        md.append(f'| {g} | {n} | {b} | {(b/n if n else 0.0):.3f} |')
    md.append('')

    md.append(_format_bucket_table('Buy rate by cash bucket', by_cash,
                                    order=[k for k, _ in _CASH_BUCKETS]))
    md.append(_format_bucket_table('Buy rate by monopoly opportunity', by_op,
                                    order=['fresh', 'partial_self',
                                           'opponent_dominates', 'unknown']))

    md.append('## Surprising decisions (qualitative excerpts)')
    md.append('')
    md.append('### PASS while flush (cash >= 1000)')
    md.append('')
    if not flush_passes:
        md.append('_None._')
    for score, r in flush_passes:
        md.append(f'- cash=${r.get("cash")}, prop=`{r.get("prop_name")}` '
                  f'({r.get("prop_group")} group, ${r.get("prop_cost")}), '
                  f'self={r.get("same_group_self")}/{r.get("group_size")}, '
                  f'opp={r.get("opp_in_group")} — '
                  f'reason: _{r.get("reason_text") or "<no REASON>"}_')
    md.append('')
    md.append('### BUY at cash floor (cash < 300)')
    md.append('')
    if not poor_buys:
        md.append('_None._')
    for score, r in poor_buys:
        md.append(f'- cash=${r.get("cash")}, prop=`{r.get("prop_name")}` '
                  f'({r.get("prop_group")} group, ${r.get("prop_cost")}), '
                  f'self={r.get("same_group_self")}/{r.get("group_size")}, '
                  f'opp={r.get("opp_in_group")} — '
                  f'reason: _{r.get("reason_text") or "<no REASON>"}_')
    md.append('')

    out_md_path.write_text('\n'.join(md), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_dir', required=True,
                    help='eval_llm_on_boards.py output directory '
                         '(e.g. logs/llm_eval/run1).')
    ap.add_argument('--out-name', default='analysis.md')
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    summary_rows = _read_summary(in_dir / 'summary.csv')
    if not summary_rows:
        print(f'No summary.csv rows found under {in_dir}.')
        return

    # Group by board_tag
    boards = sorted({r['board_tag'] for r in summary_rows})
    print(f'Boards found: {boards}')

    combined_md = ['# Combined LLM-decision analysis', '']
    for tag in boards:
        decisions_dir = in_dir / 'decisions' / tag
        records = list(_read_decisions(decisions_dir))
        out_path = in_dir / f'analysis_{tag}.md'
        _analyse_board(tag, records, summary_rows, out_path)
        print(f'  wrote {out_path}')
        combined_md.append(f'## {tag}')
        combined_md.append('')
        combined_md.append(out_path.read_text(encoding='utf-8'))
        combined_md.append('')

    (in_dir / args.out_name).write_text('\n'.join(combined_md), encoding='utf-8')
    print(f'Wrote {in_dir / args.out_name}')


if __name__ == '__main__':
    main()
