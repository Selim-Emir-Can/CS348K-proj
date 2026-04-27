"""Group-level design vocabulary shared by the transfer audit (Step 0),
the parametric closed-loop designer (#4), and the rule-mutation loop's
benchmark designs (#5).

A "group design" is a small, JSON-serialisable record that can be applied to
EITHER the mini config OR the canonical config. The set of supported
primitives is the exact set of operations the LLM-as-parametric-designer is
allowed to emit; centralising the schema here means the audit really does
test the same vocabulary the loop iterates over.

Primitives
----------
- salary_mult : float                multiplier on `mechanics.salary`
- drop_groups : list[str]            colour groups whose properties become inert Cells
- group_cost_mult : dict[str, float] per-group multiplier on Property.cost_base
- group_rent_mult : dict[str, float] per-group multiplier on Property.rent_base
                                      AND every entry of rent_house
- prop_overrides : dict[str, dict]   ESCAPE HATCH: per-property cost/rent overrides
                                      keyed by property name. Used sparingly; the
                                      group-level primitives are the recommended path.

Group names accepted are exactly those present on the target config; passing
a group not on the board is an error so the LLM cannot silently no-op a
"raise rent on Indigo" instruction issued to the mini board (which has no
Indigo).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from config import GameConfig
from monopoly.core.cell import Cell, Property
from settings import GameMechanics
from dataclasses import replace as _replace


@dataclass
class GroupDesign:
    """JSON-serialisable description of a parametric intervention."""
    salary_mult: float = 1.0
    drop_groups: List[str] = field(default_factory=list)
    group_cost_mult: Dict[str, float] = field(default_factory=dict)
    group_rent_mult: Dict[str, float] = field(default_factory=dict)
    prop_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Optional human-readable label for figures and JSONL.
    label: str = ''
    # Optional rationale text (LLM emits this; audit designs leave blank).
    rationale: str = ''

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'GroupDesign':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _board_groups(cfg: GameConfig) -> List[str]:
    """Distinct colour-group names present on cfg's board, in board order."""
    seen = []
    for c in cfg.cells:
        if isinstance(c, Property) and c.group not in seen:
            seen.append(c.group)
    return seen


def apply_design(cfg: GameConfig, design: GroupDesign,
                 strict_groups: bool = True) -> GameConfig:
    """Return a deep-copied cfg with the design applied.

    strict_groups=True (default) raises ValueError if the design references a
    colour group not present on cfg. Set False to allow silent no-ops on
    cross-board designs whose groups are a superset of the target board.
    """
    out = deepcopy(cfg)
    present = set(_board_groups(out))

    # Salary
    if design.salary_mult != 1.0:
        new_salary = int(round(out.settings.mechanics.salary * design.salary_mult))
        out.settings = _replace(
            out.settings,
            mechanics=_replace(out.settings.mechanics, salary=new_salary),
        )

    # Validate group references up front so a typo is loud, not silent.
    referenced = (set(design.drop_groups)
                  | set(design.group_cost_mult.keys())
                  | set(design.group_rent_mult.keys()))
    missing = referenced - present
    if missing and strict_groups:
        raise ValueError(
            f'design references groups not on board: {sorted(missing)}; '
            f'present: {sorted(present)}')

    # Drop groups (replace each Property in those groups with an inert Cell stub).
    drop_set = set(design.drop_groups)
    if drop_set:
        out.cells = [
            Cell(c.name) if (isinstance(c, Property) and c.group in drop_set) else c
            for c in out.cells
        ]

    # Group-level cost/rent multipliers.
    if design.group_cost_mult or design.group_rent_mult:
        new_cells = []
        for c in out.cells:
            if not isinstance(c, Property):
                new_cells.append(c); continue
            cost_m = design.group_cost_mult.get(c.group, 1.0)
            rent_m = design.group_rent_mult.get(c.group, 1.0)
            if cost_m == 1.0 and rent_m == 1.0:
                new_cells.append(c); continue
            new_cells.append(Property(
                name=c.name,
                cost_base=int(round(c.cost_base * cost_m)),
                rent_base=int(round(c.rent_base * rent_m)),
                cost_house=c.cost_house,
                rent_house=tuple(int(round(r * rent_m)) for r in c.rent_house),
                group=c.group,
            ))
        out.cells = new_cells

    # Per-property escape hatch (applied last, wins over group rules).
    for prop_name, overrides in design.prop_overrides.items():
        for i, c in enumerate(out.cells):
            if not isinstance(c, Property) or c.name != prop_name:
                continue
            new_c = Property(
                name=c.name,
                cost_base=int(overrides.get('cost_base', c.cost_base)),
                rent_base=int(overrides.get('rent_base', c.rent_base)),
                cost_house=int(overrides.get('cost_house', c.cost_house)),
                rent_house=tuple(overrides.get('rent_house', c.rent_house)),
                group=c.group,
            )
            out.cells[i] = new_c
            break

    return out


# --------------------------------------------------------------------------- #
# Audit design set (5 designs that apply to BOTH mini and canonical)           #
# --------------------------------------------------------------------------- #
#
# Built from groups present on mini ({Brown, Lightblue, Pink, Orange}) so the
# same record works on canonical (which is a superset). The set is intentionally
# diverse: a no-op control, a salary perturbation, a structural drop, a
# rent-only knob, and a mixed multi-knob design. Picked before any audit ran
# (no cherry-picking).
AUDIT_DESIGNS: List[GroupDesign] = [
    GroupDesign(label='baseline',
                 rationale='no-op control'),
    GroupDesign(label='salary x1.5',
                 salary_mult=1.5,
                 rationale='probe pacing knob'),
    GroupDesign(label='drop Brown',
                 drop_groups=['Brown'],
                 rationale='probe structural removal'),
    GroupDesign(label='Orange rent x1.5',
                 group_rent_mult={'Orange': 1.5},
                 rationale='probe single-group rent inflation'),
    GroupDesign(label='salary x0.75 + Lightblue cost x1.5',
                 salary_mult=0.75,
                 group_cost_mult={'Lightblue': 1.5},
                 rationale='probe interaction between two knobs'),
]
