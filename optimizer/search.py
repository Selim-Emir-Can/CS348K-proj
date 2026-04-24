"""Random search + Genetic Algorithm over the board design space.

Both algorithms share the same callback signature: `eval_fn(vec) -> (score, metrics_dict)`.
They emit a history of {iter, vec, score, metrics} dicts. The outer driver
(scripts/optimize_board.py) writes one JSONL line per iteration; post-hoc
reporting uses that file.
"""
from typing import Callable, Dict, List, Tuple

import numpy as np


EvalFn = Callable[[np.ndarray], Tuple[float, Dict]]


def _log_iter(history: List[Dict], it: int, vec: np.ndarray,
              score: float, metrics: Dict, extra: Dict = None,
              on_iter: Callable[[Dict], None] = None):
    entry = {
        'iter':    it,
        'vec':     vec.tolist(),
        'score':   float(score),
        'metrics': metrics,
    }
    if extra:
        entry.update(extra)
    history.append(entry)
    if on_iter is not None:
        on_iter(entry)


# --------------------------------------------------------------------------- #
# Random search                                                                  #
# --------------------------------------------------------------------------- #

def random_search(space, eval_fn: EvalFn, n_iters: int,
                  seed: int = 0, on_iter=None) -> List[Dict]:
    rng = np.random.default_rng(seed)
    history: List[Dict] = []
    for it in range(n_iters):
        vec = space.sample(rng)
        score, metrics = eval_fn(vec)
        _log_iter(history, it, vec, score, metrics, on_iter=on_iter)
    return history


# --------------------------------------------------------------------------- #
# Genetic algorithm                                                             #
# --------------------------------------------------------------------------- #

def _tournament_pick(pop_scores: List[float], rng: np.random.Generator,
                     k: int = 3) -> int:
    """Return the index of the winner (lowest score) of a k-tournament."""
    candidates = rng.choice(len(pop_scores), size=k, replace=False)
    winner = int(min(candidates, key=lambda i: pop_scores[int(i)]))
    return winner


def _uniform_crossover(a: np.ndarray, b: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(len(a)) < 0.5
    child = np.where(mask, a, b)
    return child


def _mutate(vec: np.ndarray, space, rng: np.random.Generator,
            sigma: float = 0.1, p_mut_cont: float = 0.1,
            p_mut_binary: float = 0.05) -> np.ndarray:
    """Gaussian-mutate the continuous head; bit-flip the binary tail."""
    child = vec.copy()
    n_cont = getattr(space, 'N_CONT', len(child) - 1)
    # Continuous mutation
    cmask = rng.random(n_cont) < p_mut_cont
    child[:n_cont] = np.where(
        cmask, child[:n_cont] + rng.normal(0, sigma, n_cont), child[:n_cont])
    # Binary bit-flip mutation (skip for legacy spaces whose tail is discrete N_props)
    n_bin = len(child) - n_cont
    if n_bin >= 2:
        bmask = rng.random(n_bin) < p_mut_binary
        child[n_cont:] = np.where(bmask, 1.0 - child[n_cont:], child[n_cont:])
    return space.clip(child)


def genetic_algorithm(space, eval_fn: EvalFn, pop_size: int = 20,
                      generations: int = 10, seed: int = 0,
                      elitism: int = 2, tournament_k: int = 3,
                      on_iter=None) -> List[Dict]:
    """Minimise `eval_fn`. Returns per-evaluation history (length pop*generations)."""
    rng = np.random.default_rng(seed)
    history: List[Dict] = []
    global_iter = [0]   # mutable counter across nested calls

    def _eval_and_log(vec, gen, slot):
        score, metrics = eval_fn(vec)
        _log_iter(history, global_iter[0], vec, score, metrics,
                  extra={'gen': gen, 'slot': slot}, on_iter=on_iter)
        global_iter[0] += 1
        return score, metrics

    # --- init population: gen 0 --------------------------------------- #
    population = [space.sample(rng) for _ in range(pop_size)]
    scores = []
    for i, v in enumerate(population):
        s, _ = _eval_and_log(v, gen=0, slot=i)
        scores.append(s)

    for gen in range(1, generations):
        # Elitism: carry the top `elitism` unchanged
        order = sorted(range(pop_size), key=lambda i: scores[i])
        elites_idx = order[:elitism]
        new_pop = [population[i].copy() for i in elites_idx]
        new_scores = [scores[i] for i in elites_idx]

        # Fill the rest via tournament selection → crossover → mutation
        while len(new_pop) < pop_size:
            i = _tournament_pick(scores, rng, k=tournament_k)
            j = _tournament_pick(scores, rng, k=tournament_k)
            child = _uniform_crossover(population[i], population[j], rng)
            child = _mutate(child, space, rng)
            new_pop.append(child)
            new_scores.append(None)   # placeholder; evaluated below

        # Evaluate new individuals (elites already have scores)
        for slot in range(elitism, pop_size):
            s, _ = _eval_and_log(new_pop[slot], gen=gen, slot=slot)
            new_scores[slot] = s

        population, scores = new_pop, new_scores

    return history
