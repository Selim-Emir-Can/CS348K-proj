"""Diagnostic smoke test for the LLMPlayer's buy-decision behaviour.

Runs the same three probes against whatever LLM backend / model is
currently configured in agents.LLMPlayer (default: Qwen2.5-1.5B-Instruct
via local huggingface transformers; override with LLM_MODEL env var).

  Test 1: vary cash level only, no opponent context. Sanity-check that
          the LLM differentiates BUY/PASS based on liquidity.
  Test 2: opponent already dominates the colour group. Expect: PASS
          (or at least more PASS than Test 1 at the same cash).
  Test 3: a buy that COMPLETES the player's own monopoly. Expect: BUY
          at any reasonable cash level.

Output: one line per probe with cash level, decision, and (where useful)
the raw model response so we can see what the LLM actually said.

Usage (from monopoly/):
    set PYTHONPATH=. && python scripts/smoke_llm_player.py
    set PYTHONPATH=. && python scripts/smoke_llm_player.py --model Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse
import os
import time

from dataclasses import replace

from agents import LLMPlayer
from monopoly.core.cell import Property


def _make_stub_board(cells, owner_self, owner_opp, opp_props_in_group=()):
    """Construct a minimal board / players stub so _build_buy_prompt can
    populate the opponent-context fields."""
    class _Stub:
        pass
    opp = _Stub()
    opp.owned = list(opp_props_in_group)
    for p in opp.owned:
        p.owner = opp
    class _BoardStub:
        pass
    b = _BoardStub()
    b.cells = list(cells)
    return b, opp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None,
                    help='Override LLM_MODEL env var (e.g. Qwen/Qwen2.5-0.5B-Instruct).')
    ap.add_argument('--backend', default='local', choices=('local', 'openai', 'heuristic'))
    ap.add_argument('--cache-dir', default=None,
                    help='HF cache directory for model weights. '
                         'Default: models/hf_cache relative to CWD.')
    args = ap.parse_args()

    if args.model:
        os.environ['LLM_MODEL'] = args.model
    if args.cache_dir:
        os.environ['LLM_CACHE_DIR'] = args.cache_dir

    p = LLMPlayer('LLM', backend=args.backend)
    p.settings = replace(p.settings, unspendable_cash=0)   # disable affordability shortcut
    boardwalk = Property('H2 Boardwalk', 400, 50, 200, (200, 600, 1400, 1700, 2000), 'Indigo')

    # ----- Test 1: cash variation only ----------------------------------- #
    print(f'=== Test 1: cash variation, no opponent context (model={os.environ.get("LLM_MODEL", "default")}) ===')
    for cash in (450, 600, 1000, 2000, 5000):
        p.money = cash
        t0 = time.time()
        ans = p._should_buy(boardwalk)
        print(f'  cash=${cash:>5}  ->  {"BUY" if ans else "PASS"}    [{(time.time()-t0)*1000:>5.0f} ms]')

    # ----- Test 2: opponent dominates Indigo ----------------------------- #
    print()
    print('=== Test 2: opponent owns 1/2 of Indigo (PASS expected at high cash) ===')
    own_brown = Property('A1 Mediterranean Avenue', 60, 2, 50, (10, 30, 90, 160, 250), 'Brown')
    own_brown.owner = p
    p.owned = [own_brown]
    opp_indigo = Property('H1 Park Place', 350, 35, 200, (175, 500, 1100, 1300, 1500), 'Indigo')
    b, opp = _make_stub_board([own_brown, boardwalk, opp_indigo], p, None,
                              opp_props_in_group=[opp_indigo])
    p._board_ref = b
    p._players_ref = [p, opp]
    for cash in (450, 600, 1000, 2000, 5000):
        p.money = cash
        ans = p._should_buy(boardwalk)
        print(f'  cash=${cash:>5}  ->  {"BUY" if ans else "PASS"}')

    # ----- Test 3: buying COMPLETES my own monopoly ---------------------- #
    print()
    print('=== Test 3: completing my OWN Indigo monopoly (BUY expected) ===')
    own_indigo = Property('H1 Park Place', 350, 35, 200, (175, 500, 1100, 1300, 1500), 'Indigo')
    own_indigo.owner = p
    p.owned = [own_brown, own_indigo]
    b, opp = _make_stub_board([own_brown, boardwalk, own_indigo], p, None, opp_props_in_group=[])
    p._board_ref = b
    p._players_ref = [p, opp]
    for cash in (450, 1000, 2000):
        p.money = cash
        ans = p._should_buy(boardwalk)
        print(f'  cash=${cash:>5}  ->  {"BUY" if ans else "PASS"}')

    print()
    print(f'Total queries: {p._n_buy_decisions}   BUY rate: {p._n_buy_yes / max(p._n_buy_decisions, 1):.0%}')

    # ----- Raw response sample on the most diagnostic case --------------- #
    if args.backend != 'heuristic':
        print()
        print('=== Raw response on monopoly-completion @ cash=$2000 ===')
        p.money = 2000
        prompt = p._build_buy_prompt(boardwalk)
        if args.backend == 'openai':
            raw = p._query_openai(prompt)
        else:
            raw = p._query_local(prompt)
        print('Prompt:')
        print(' ', prompt)
        print('Response:')
        print(' ', raw)


if __name__ == '__main__':
    main()
