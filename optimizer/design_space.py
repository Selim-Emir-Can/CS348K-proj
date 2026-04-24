"""Board design space: 66-dim vector ↔ GameConfig encoding (with full mask).

Dimensions (in order):
  [0 .. 21]    cost multipliers   c_i  ∈ [0.5, 2.0]  for the 22 colour-group properties
  [22 .. 43]   rent multipliers   r_i  ∈ [0.5, 2.0]  for the same 22 properties
  [44 .. 65]   keep mask          m_i  ∈ {0, 1}      — 1 = keep property, 0 = remove

A removed property (mask bit = 0) is replaced with a FreeParking cell at the
same board index. This keeps the 40-cell board length intact so hardcoded
Chance-card positions (Advance to Boardwalk=39, nearest-Railroad=5/15/25/35,
etc.) keep working.

Constraint: the mask must keep at least MIN_KEPT=8 properties so games can
still form monopolies. If a sampled or mutated mask goes below that, we flip
random off-bits on until the floor is reached.

Railroads and utilities are fixed in v1 (not in the design space).

Legacy support: 45-dim vecs from the earlier ordered-removal design space
(continuous × 44 + integer N_props at index 44) are still decodable via the
legacy codepath so older run logs and rendered boards remain reproducible.
"""
from copy import deepcopy
from dataclasses import replace
from typing import List

import numpy as np

from monopoly.core.cell import FreeParking, Property


# --- Constants --------------------------------------------------------------- #

_MIN_MULT, _MAX_MULT = 0.5, 2.0
_MIN_NPROPS, _MAX_NPROPS = 8, 22
_N_CG_PROPS = 22   # number of colour-group properties on the default board


# --- Helpers ----------------------------------------------------------------- #

def _colour_group_indices(cells) -> List[int]:
    """Board indices of colour-group properties (NOT Railroads/Utilities).

    Ordered by board position so results are stable across runs.
    """
    from monopoly.core.constants import RAILROADS, UTILITIES
    return [i for i, c in enumerate(cells)
            if isinstance(c, Property) and c.group not in (RAILROADS, UTILITIES)]


def _cost_rank_order(cells, cg_indices, direction: str = 'cheapest') -> List[int]:
    """Positions in cg_indices ordered by default cost_base.

    direction:
      'cheapest'  -- ascending; shrinkage removes cheap properties first
                     (original behaviour).
      'expensive' -- descending; shrinkage removes expensive properties
                     first (e.g. Green/Indigo, which are the dominant source
                     of late-game blowouts).
      'middle'    -- removes from the middle of the cost distribution
                     (pink/orange/red). Kept for experimentation.
    """
    with_costs = [(pos, cells[bi].cost_base) for pos, bi in enumerate(cg_indices)]
    if direction == 'cheapest':
        with_costs.sort(key=lambda x: x[1])
    elif direction == 'expensive':
        with_costs.sort(key=lambda x: -x[1])
    elif direction == 'middle':
        median_cost = sorted(x[1] for x in with_costs)[len(with_costs) // 2]
        with_costs.sort(key=lambda x: abs(x[1] - median_cost))
    else:
        raise ValueError(f'unknown removal direction: {direction}')
    return [pos for pos, _ in with_costs]


# --- Public class ------------------------------------------------------------ #

class DesignSpace:
    """Encoder/decoder between a 66-dim vector and a `GameConfig`.

    Genotype = 22 cost multipliers + 22 rent multipliers + 22 keep-bits.
    `decode(vec)` returns a new `GameConfig` with multipliers applied and
    mask-based shrinkage performed; `sample(rng)` draws a uniform random
    vector; `bounds` returns per-dim (lo, hi) tuples.
    """

    # Public structural constants used by search.py to split continuous /
    # discrete portions of the genotype.
    N_CONT = 2 * _N_CG_PROPS   # 44: cost multipliers + rent multipliers
    N_BIN  = _N_CG_PROPS       # 22: keep/remove mask
    MIN_KEPT = 8               # soft floor so boards stay playable

    def __init__(self, base_cfg, removal_direction: str = 'cheapest'):
        self.base_cfg = base_cfg
        # removal_direction only matters for legacy 45-dim vec decoding.
        self.removal_direction = removal_direction
        self._cg_indices = _colour_group_indices(base_cfg.cells)
        assert len(self._cg_indices) == _N_CG_PROPS, (
            f'Expected {_N_CG_PROPS} colour-group properties on the base board, '
            f'found {len(self._cg_indices)}'
        )
        # Legacy rank only used when decoding 45-dim vecs from old runs.
        self._cost_rank = _cost_rank_order(base_cfg.cells, self._cg_indices,
                                           direction=removal_direction)
        # Pre-cache default params so decode() can regenerate without reading them repeatedly.
        self._defaults = [
            {
                'name':         base_cfg.cells[bi].name,
                'cost_base':    base_cfg.cells[bi].cost_base,
                'rent_base':    base_cfg.cells[bi].rent_base,
                'cost_house':   base_cfg.cells[bi].cost_house,
                'rent_house':   tuple(base_cfg.cells[bi].rent_house),
                'group':        base_cfg.cells[bi].group,
            } for bi in self._cg_indices
        ]

    # ---- bounds / dims -------------------------------------------------- #

    @property
    def n_dims(self) -> int:
        return self.N_CONT + self.N_BIN   # 66

    def bounds(self):
        """Return a list[(lo, hi)] of length n_dims."""
        return (
            [(_MIN_MULT, _MAX_MULT)] * self.N_CONT +
            [(0.0, 1.0)] * self.N_BIN
        )

    # ---- sampling ------------------------------------------------------ #

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        v = np.empty(self.n_dims, dtype=np.float64)
        v[:self.N_CONT] = rng.uniform(_MIN_MULT, _MAX_MULT, self.N_CONT)
        mask = rng.integers(0, 2, self.N_BIN).astype(np.float64)
        v[self.N_CONT:] = self._enforce_min_kept(mask, rng)
        return v

    def _enforce_min_kept(self, mask: np.ndarray, rng) -> np.ndarray:
        """Flip random off-bits on until popcount >= MIN_KEPT."""
        kept = int(mask.sum())
        if kept >= self.MIN_KEPT:
            return mask
        off_idx = np.where(mask == 0)[0]
        n_flip = self.MIN_KEPT - kept
        chosen = rng.choice(off_idx, size=n_flip, replace=False)
        mask = mask.copy()
        mask[chosen] = 1.0
        return mask

    def clip(self, v: np.ndarray) -> np.ndarray:
        out = np.asarray(v, dtype=np.float64).copy()
        out[:self.N_CONT] = np.clip(out[:self.N_CONT], _MIN_MULT, _MAX_MULT)
        # Round binary dims to {0, 1}
        out[self.N_CONT:] = np.round(np.clip(out[self.N_CONT:], 0.0, 1.0))
        # Enforce popcount deterministically. Use a content-stable hash
        # (SHA256) rather than Python's built-in hash() — the latter is
        # randomised per-process via PYTHONHASHSEED, which makes clip() (and
        # therefore the whole GA) non-reproducible across runs.
        if out[self.N_CONT:].sum() < self.MIN_KEPT:
            import hashlib
            digest = hashlib.sha256(out.tobytes()).digest()
            seed = int.from_bytes(digest[:4], 'big')
            rng = np.random.default_rng(seed)
            out[self.N_CONT:] = self._enforce_min_kept(out[self.N_CONT:], rng)
        return out

    # ---- encoding of the default (identity vector) --------------------- #

    def identity_vec(self) -> np.ndarray:
        """Default board = all multipliers at 1.0, all properties kept."""
        return np.concatenate([
            np.ones(self.N_CONT, dtype=np.float64),   # cost + rent mults
            np.ones(self.N_BIN,  dtype=np.float64),   # full keep mask
        ])

    # ---- decoding ------------------------------------------------------ #

    def decode(self, vec):
        """Return a new GameConfig with per-property multipliers and shrinkage applied.

        Supports both the new 66-dim (mask) encoding and legacy 45-dim
        (ordered-removal) vecs for backward compatibility with old run logs.
        """
        vec = np.asarray(vec, dtype=np.float64)
        if len(vec) == 2 * _N_CG_PROPS + 1:
            return self._decode_legacy(vec)
        if len(vec) == self.n_dims:
            return self._decode_mask(self.clip(vec))
        raise ValueError(f'bad vec length {len(vec)}; expected 45 (legacy) or '
                         f'{self.n_dims} (mask)')

    def decode_as_substituted(self, vec):
        """Alternative decode for rendering only: apply the mask by replacing
        removed properties with FreeParking cells at the SAME board index,
        keeping the board at 40 cells.

        Not used by the optimiser (which uses the shrunk decode via decode()).
        Useful only for side-by-side visualisation comparisons with the old
        board layout. The returned GameConfig is NOT suitable for simulation.
        """
        vec = np.asarray(vec, dtype=np.float64)
        if len(vec) == 2 * _N_CG_PROPS + 1:
            # Legacy 45-dim vecs already produce a 40-cell substituted board
            # via _decode_legacy, so just delegate.
            return self._decode_legacy(vec)
        if len(vec) != self.n_dims:
            raise ValueError(f'bad vec length {len(vec)}; expected 45 (legacy) or '
                             f'{self.n_dims} (mask)')
        vec = self.clip(vec)
        cost_mults = vec[:_N_CG_PROPS]
        rent_mults = vec[_N_CG_PROPS:2*_N_CG_PROPS]
        mask       = vec[self.N_CONT:]
        cfg = deepcopy(self.base_cfg)
        new_cells = list(cfg.cells)
        for pos, bi in enumerate(self._cg_indices):
            if mask[pos] < 0.5:
                new_cells[bi] = FreeParking(self._defaults[pos]['name'])
                continue
            d = self._defaults[pos]
            c_mult = float(cost_mults[pos])
            r_mult = float(rent_mults[pos])
            new_cells[bi] = Property(
                d['name'],
                max(1, int(round(d['cost_base']  * c_mult))),
                max(1, int(round(d['rent_base']  * r_mult))),
                max(1, int(round(d['cost_house'] * c_mult))),
                tuple(max(1, int(round(r * r_mult))) for r in d['rent_house']),
                d['group'],
            )
        cfg.cells = new_cells
        return cfg

    def _decode_mask(self, vec: np.ndarray):
        """Build a shrunk cell list by dropping masked-off properties entirely.

        Unlike the legacy decode, we do not FreeParking-substitute removed
        cells; we remove them. The resulting board is shorter than 40 cells,
        which genuinely changes game dynamics (shorter laps → more salary
        per game). All engine hardcoded positions (jail, go-to-jail, card
        targets) now resolve via board.cell_index_by_name /
        board.next_cell_of_group so the shorter board plays correctly.
        """
        cost_mults = vec[:_N_CG_PROPS]
        rent_mults = vec[_N_CG_PROPS:2*_N_CG_PROPS]
        mask       = vec[self.N_CONT:]

        # Build a set of board indices to drop.
        drop_indices = {bi for pos, bi in enumerate(self._cg_indices)
                        if mask[pos] < 0.5}

        # Map colour-group board-index → (cost_mult, rent_mult) for those we keep.
        kept_multipliers = {
            bi: (float(cost_mults[pos]), float(rent_mults[pos]))
            for pos, bi in enumerate(self._cg_indices) if bi not in drop_indices
        }

        cfg = deepcopy(self.base_cfg)
        new_cells = []
        for bi, cell in enumerate(cfg.cells):
            if bi in drop_indices:
                continue   # truly skip — the board gets shorter by one cell
            # Non-colour-group cells pass through untouched.
            if bi not in kept_multipliers:
                new_cells.append(cell)
                continue
            # Apply multipliers to kept colour-group properties.
            c_mult, r_mult = kept_multipliers[bi]
            # Find this property's defaults by its board index via the cg_indices lookup.
            pos = self._cg_indices.index(bi)
            d = self._defaults[pos]
            new_cells.append(Property(
                d['name'],
                max(1, int(round(d['cost_base']  * c_mult))),
                max(1, int(round(d['rent_base']  * r_mult))),
                max(1, int(round(d['cost_house'] * c_mult))),
                tuple(max(1, int(round(r * r_mult))) for r in d['rent_house']),
                d['group'],
            ))
        cfg.cells = new_cells
        return cfg

    def _decode_legacy(self, vec: np.ndarray):
        """Old 45-dim encoding: N_props integer + cost-ranked removal order."""
        out = vec.copy()
        out[:2*_N_CG_PROPS] = np.clip(out[:2*_N_CG_PROPS], _MIN_MULT, _MAX_MULT)
        out[-1] = int(np.clip(round(float(out[-1])), _MIN_NPROPS, _MAX_NPROPS))

        cost_mults = out[:_N_CG_PROPS]
        rent_mults = out[_N_CG_PROPS:2*_N_CG_PROPS]
        n_props = int(out[-1])
        n_remove = _N_CG_PROPS - n_props
        removed_positions = set(self._cost_rank[:n_remove])

        cfg = deepcopy(self.base_cfg)
        new_cells = list(cfg.cells)
        for pos, bi in enumerate(self._cg_indices):
            if pos in removed_positions:
                new_cells[bi] = FreeParking(self._defaults[pos]['name'])
                continue
            d = self._defaults[pos]
            c_mult = float(cost_mults[pos])
            r_mult = float(rent_mults[pos])
            new_cells[bi] = Property(
                d['name'],
                max(1, int(round(d['cost_base']  * c_mult))),
                max(1, int(round(d['rent_base']  * r_mult))),
                max(1, int(round(d['cost_house'] * c_mult))),
                tuple(max(1, int(round(r * r_mult))) for r in d['rent_house']),
                d['group'],
            )
        cfg.cells = new_cells
        return cfg
