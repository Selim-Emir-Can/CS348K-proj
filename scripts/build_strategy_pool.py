"""Generate and save the 30-strategy pool + print a human-readable summary.

Usage (from monopoly/):
    python scripts/build_strategy_pool.py                    # default seed, saves to optimizer/strategy_pool.json
    python scripts/build_strategy_pool.py --seed 42 --out optimizer/strategy_pool_v2.json
"""
import argparse
from pathlib import Path

from optimizer.strategy_pool import build_pool, save_pool, load_eval_matchups


def _summarise(pool):
    print(f'{"name":<22}{"cash":>6}{"bld_fl":>8}{"trade":>6}{"aggr":>6}'
          f'{"util":>6}{"rail":>6}{"jail":>6}  ignored')
    print('-' * 90)
    for name, s in pool:
        ignored = ','.join(sorted(s.ignore_property_groups)) or '-'
        print(f'{name:<22}{s.unspendable_cash:>6}{s.build_cash_floor:>8}'
              f'{str(s.is_willing_to_make_trades)[0]:>6}'
              f'{str(s.aggressive_build)[0]:>6}'
              f'{str(s.buy_utilities)[0]:>6}'
              f'{str(s.buy_railroads)[0]:>6}'
              f'{s.jail_pay_threshold:>6}  {ignored}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-sampled', type=int, default=20)
    ap.add_argument('--out', default='optimizer/strategy_pool.json')
    ap.add_argument('--matchups-seed', type=int, default=1234)
    args = ap.parse_args()

    pool = build_pool(seed=args.seed, n_sampled=args.n_sampled)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_pool(args.out, pool)
    print(f'Saved {len(pool)} strategies to {args.out}\n')
    _summarise(pool)

    # Preview the eval matchups the optimiser will use.
    for n_players in (2, 3):
        m = load_eval_matchups(n_players, pool_size=len(pool),
                               seed=args.matchups_seed)
        print(f'\n--- {n_players}-player eval matchups ({len(m)} total) ---')
        for i, idxs in enumerate(m):
            names = ' vs '.join(pool[j][0] for j in idxs)
            print(f'  [{i:2d}]  {names}')


if __name__ == '__main__':
    main()
