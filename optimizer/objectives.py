"""Scoring functions over per-game simulation results.

Pure functions: take a list of per-game dicts (as returned by
simulate.run_matchup) and return scalars or dicts. No side-effects.

The combined score is:
    score = w_fair  * mean_fairness
          + w_fmax  * max_fairness
          + w_len   * |mean_rounds − target_rounds| / target_rounds
          + w_draw  * mean_draw_rate
          + w_money * |mean_transfer_rate − target_transfer| / target_transfer

All terms are non-negative; minimising drives the design toward a fair, paced,
decisive, money-circulating game.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence


# --------------------------------------------------------------------------- #
# Primitive aggregates                                                          #
# --------------------------------------------------------------------------- #

def per_strategy_win_rates(results: List[dict]) -> Dict[str, float]:
    """Return {strategy_name: wins / games_played}.

    Respects seat-permutation: a strategy gets credit for games where *it*
    (not its seat) won, using result['winner'] against result['strategy_names'].
    """
    plays: Dict[str, int] = {}
    wins:  Dict[str, int] = {}
    for r in results:
        for name in r['strategy_names']:
            plays[name] = plays.get(name, 0) + 1
        if r['winner'] is not None:
            wins[r['winner']] = wins.get(r['winner'], 0) + 1
    return {k: wins.get(k, 0) / plays[k] for k in plays}


def fairness_within_matchup(results: List[dict]) -> float:
    """Within a single matchup's games, return max_wr − min_wr across strategies.

    For 2p this equals |win_rate_p0 − win_rate_p1|. For 3p it's the spread
    between the best and worst player's win rates, which is directly
    comparable to the 2p metric.
    """
    wrs = per_strategy_win_rates(results)
    if not wrs:
        return 0.0
    return max(wrs.values()) - min(wrs.values())


def draw_rate(results: List[dict]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r['truncated']) / len(results)


def mean_rounds(results: List[dict]) -> float:
    if not results:
        return 0.0
    return sum(r['rounds'] for r in results) / len(results)


def mean_transfer_rate(results: List[dict]) -> float:
    """Player-to-player cash flow per round, averaged over games."""
    if not results:
        return 0.0
    rates = []
    for r in results:
        rounds = max(r['rounds'], 1)
        rates.append(r['transfer_total'] / rounds)
    return sum(rates) / len(rates)


# --------------------------------------------------------------------------- #
# Objective bundle                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class Weights:
    w_fair:  float = 1.0
    w_fmax:  float = 0.5
    w_len:   float = 0.5
    w_draw:  float = 0.3
    w_money: float = 0.3


@dataclass
class Targets:
    target_rounds:   float = 60.0
    target_transfer: float = 100.0   # $ / round


def evaluate(results_by_matchup: Sequence[List[dict]],
             weights: Weights = None, targets: Targets = None) -> Dict:
    """Given per-matchup result lists, return the combined score and metric dict.

    results_by_matchup: one list per matchup (10 matchups per candidate).
    Returns: {
      'score':    float,
      'metrics':  {mean_fairness, max_fairness, mean_rounds, draw_rate, mean_transfer_rate},
      'per_matchup': [{fairness, draw_rate, mean_rounds, transfer_rate}, ...],
    }
    """
    weights = weights or Weights()
    targets = targets or Targets()

    per_mu = []
    for rs in results_by_matchup:
        per_mu.append({
            'fairness':      fairness_within_matchup(rs),
            'draw_rate':     draw_rate(rs),
            'mean_rounds':   mean_rounds(rs),
            'transfer_rate': mean_transfer_rate(rs),
            'n_games':       len(rs),
        })

    if not per_mu:
        return {'score': 0.0, 'metrics': {}, 'per_matchup': []}

    all_results = [r for rs in results_by_matchup for r in rs]
    mean_fair = sum(m['fairness']  for m in per_mu) / len(per_mu)
    max_fair  = max((m['fairness'] for m in per_mu), default=0.0)
    mean_r    = mean_rounds(all_results)
    d_rate    = draw_rate(all_results)
    t_rate    = mean_transfer_rate(all_results)

    length_pen   = abs(mean_r - targets.target_rounds) / max(targets.target_rounds, 1.0)
    transfer_pen = abs(t_rate - targets.target_transfer) / max(targets.target_transfer, 1.0)

    score = (
        weights.w_fair  * mean_fair
      + weights.w_fmax  * max_fair
      + weights.w_len   * length_pen
      + weights.w_draw  * d_rate
      + weights.w_money * transfer_pen
    )

    return {
        'score': score,
        'metrics': {
            'mean_fairness':      mean_fair,
            'max_fairness':       max_fair,
            'mean_rounds':        mean_r,
            'mean_draw_rate':     d_rate,
            'mean_transfer_rate': t_rate,
            'length_penalty':     length_pen,
            'transfer_penalty':   transfer_pen,
        },
        'per_matchup': per_mu,
    }
