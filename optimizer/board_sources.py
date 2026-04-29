"""Canonical "five board" set used by the game-space exploration scripts.

Centralised so `scripts/hazard_curves.py` and `scripts/llm_character.py` (and
any future probe) resolve the same five boards from the same primitives:

  default     -- canonical Hasbro-shaped config
  GA-2p       -- best-score entry from a 2p GA optimiser run log
  GA-3p       -- best-score entry from a 3p GA optimiser run log
  salary x2   -- mini board with the per-pass salary doubled
  drop Brown  -- mini board with the entire Brown colour group removed

Prior to this module these helpers lived duplicated in each script. Drift
between the two copies was a real risk (they had already started to differ
in `removal_direction` plumbing).
"""
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config import GameConfig
from monopoly.core.cell import Cell, Property
from optimizer.design_space import DesignSpace


def modify_salary(cfg: GameConfig, multiplier: float) -> GameConfig:
    """Return a deep-copied config with mechanics.salary scaled by `multiplier`."""
    out = deepcopy(cfg)
    out.settings = replace(
        out.settings,
        mechanics=replace(out.settings.mechanics,
                          salary=int(out.settings.mechanics.salary * multiplier)),
    )
    return out


def remove_group(cfg: GameConfig, group_name: str) -> GameConfig:
    """Return a deep-copied config with all Property cells in `group_name`
    replaced by inert Cell stubs."""
    out = deepcopy(cfg)
    out.cells = [
        Cell(c.name) if isinstance(c, Property) and c.group == group_name else c
        for c in out.cells
    ]
    return out


def best_vec_from_run(run_path: str) -> Optional[List[float]]:
    """Return the lowest-score entry's vec from a JSONL optimiser log, or None
    if the file does not exist or is empty."""
    p = Path(run_path)
    if not p.exists():
        return None
    best = None
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if best is None or e.get('score', float('inf')) < best.get('score', float('inf')):
                best = e
    return list(best['vec']) if best is not None else None


def build_five_boards(canonical_config: str,
                      mini_config: str,
                      ga_2p: Optional[str] = None,
                      ga_3p: Optional[str] = None,
                      removal_direction: str = 'cheapest',
                      ) -> List[Tuple[str, GameConfig]]:
    """Resolve the five (label, GameConfig) tuples used by hazard / character
    probes. GA boards are skipped (with a printed warning) when their JSONL
    run logs are absent, so the caller can run on a fresh checkout."""
    canonical = GameConfig.from_yaml(canonical_config)
    mini = GameConfig.from_yaml(mini_config)
    out: List[Tuple[str, GameConfig]] = [('default', canonical)]

    space = DesignSpace(canonical, removal_direction=removal_direction)
    for label, run_path in (('GA-2p', ga_2p), ('GA-3p', ga_3p)):
        if run_path is None:
            print(f'  [skip] {label}: no run path provided')
            continue
        vec = best_vec_from_run(run_path)
        if vec is None:
            print(f'  [skip] {label}: run log not found at {run_path}')
            continue
        out.append((label, space.decode(np.asarray(vec))))

    out.append(('salary x2', modify_salary(mini, 2.0)))
    out.append(('drop Brown', remove_group(mini, 'Brown')))
    return out
