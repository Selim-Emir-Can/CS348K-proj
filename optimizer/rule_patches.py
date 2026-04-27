"""Structured rule patches for the LLM rule-mutation closed loop (#5).

The CEO plan specifies an "AST whitelist" restricting LLM-emitted patches
to {salary, tax, jail rules, card effects, payout calculations}. We encode
the whitelist at the SEMANTIC level (one `kind` per category) rather than
attempting to validate arbitrary unified diffs — the latter is brittle
(every ast.NodeVisitor change-kind would need an "in scope" classifier)
while the former gives type-checked enum dispatch with a clean
rejected-corpus story.

Two surfaces:
  - validate_patch / apply_patch     -> used by the loop driver and sandbox
  - render_patch_as_diff             -> used by the writeup; cosmetic only

The LLM emits one of these patches per iteration (or a list, capped at the
v1 max). validate_patch returns either an accepted PatchResult or a
RejectedPatch; rejected patches go to the persistent corpus at
report/figures/llm_rules/rejected_corpus.jsonl.

Patch schema
------------
All patches are JSON dicts with at minimum:
  {"kind": <str>, ...kind-specific fields...}

Whitelisted kinds (one per CEO-plan semantic category):

  salary_change         field=int               salary in [0, 2000]
  tax_change            tax_kind={"luxury","income"}
                        value=int               in [0, 2000]
  income_tax_pct        value=float             in [0.0, 0.5]
  jail_fine_change      value=int               in [0, 1000]
  jail_pay_threshold_global   value=int         in [0, 5000]
  free_parking_jackpot  enabled=bool
  card_effect_change    deck={"chance","chest"}
                        index=int               in [0, len(deck))
                        new_text=str            <=200 chars, must match a known card-action grammar
  property_payout_mult  group=str               must exist on board
                        rent_multiplier=float   in [0.5, 2.0]

Anything outside this enum is rejected.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import GameConfig
from monopoly.core.cell import Property
from settings import GameMechanics
from dataclasses import replace as _replace


# --------------------------------------------------------------------------- #
# Bounds (load-bearing for safety + sandbox stability)                          #
# --------------------------------------------------------------------------- #

_BOUNDS: Dict[str, Tuple[Any, Any]] = {
    'salary':                  (0, 2000),
    'luxury_tax':              (0, 2000),
    'income_tax':              (0, 2000),
    'income_tax_percentage':   (0.0, 0.5),
    'exit_jail_fine':          (0, 1000),
    'rent_multiplier':         (0.5, 2.0),
}

WHITELISTED_KINDS = {
    'salary_change',
    'tax_change',
    'income_tax_pct',
    'jail_fine_change',
    'free_parking_jackpot',
    'card_effect_change',
    'property_payout_mult',
}

# Card-action grammar: a deliberately restricted set of token templates the
# LLM is allowed to use for new card text. Anything else is rejected. The
# tokens map to existing card-handling code paths so we are never "executing"
# free-form text — the card text is interpreted by the engine's card parser.
_ALLOWED_CARD_TOKEN_PATTERNS = [
    r'^Pay (?:the )?bank \$\d{1,4}\.?$',
    r'^Collect \$\d{1,4} from (?:the )?bank\.?$',
    r'^Advance to (?:Go|Boardwalk|Illinois Avenue|St Charles Place|nearest Railroad|nearest Utility)\.?$',
    r'^Go to Jail\.?$',
    r'^Get out of Jail Free\.?$',
    r'^Pay each player \$\d{1,3}\.?$',
    r'^Collect \$\d{1,3} from each player\.?$',
]


# --------------------------------------------------------------------------- #
# Patch dataclasses                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class RejectedPatch:
    """Persistent record for the rejected_corpus.jsonl artifact (CEO #8)."""
    raw_response:   str
    parsed_patch:   Optional[dict]    # None if JSON parse failed
    reason:         str
    iteration:      Optional[int] = None
    seed:           Optional[int] = None
    board_label:    Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AcceptedPatch:
    patch:        dict
    summary:      str             # one-line human-readable (for the trajectory log)


# --------------------------------------------------------------------------- #
# Validation                                                                    #
# --------------------------------------------------------------------------- #

def _board_groups(cfg: GameConfig) -> List[str]:
    seen = []
    for c in cfg.cells:
        if isinstance(c, Property) and c.group not in seen:
            seen.append(c.group)
    return seen


def _bounded(value, lo, hi, name) -> Optional[str]:
    if not (lo <= value <= hi):
        return f'{name}={value!r} out of bounds [{lo}, {hi}]'
    return None


def validate_patch(patch: dict, cfg: GameConfig) -> Optional[str]:
    """Return None if the patch is valid against cfg; else a rejection reason."""
    if not isinstance(patch, dict):
        return f'patch must be dict, got {type(patch).__name__}'
    kind = patch.get('kind')
    if kind not in WHITELISTED_KINDS:
        return f'kind={kind!r} not in whitelist {sorted(WHITELISTED_KINDS)}'

    if kind == 'salary_change':
        v = patch.get('value')
        if not isinstance(v, int):
            return f'salary_change.value must be int, got {type(v).__name__}'
        return _bounded(v, *_BOUNDS['salary'], 'value')

    if kind == 'tax_change':
        tk = patch.get('tax_kind')
        if tk not in ('luxury', 'income'):
            return f'tax_change.tax_kind={tk!r} not in {{"luxury","income"}}'
        v = patch.get('value')
        if not isinstance(v, int):
            return f'tax_change.value must be int'
        bnd_key = 'luxury_tax' if tk == 'luxury' else 'income_tax'
        return _bounded(v, *_BOUNDS[bnd_key], 'value')

    if kind == 'income_tax_pct':
        v = patch.get('value')
        if not isinstance(v, (int, float)):
            return 'income_tax_pct.value must be numeric'
        return _bounded(float(v), *_BOUNDS['income_tax_percentage'], 'value')

    if kind == 'jail_fine_change':
        v = patch.get('value')
        if not isinstance(v, int):
            return 'jail_fine_change.value must be int'
        return _bounded(v, *_BOUNDS['exit_jail_fine'], 'value')

    if kind == 'free_parking_jackpot':
        en = patch.get('enabled')
        if not isinstance(en, bool):
            return 'free_parking_jackpot.enabled must be bool'
        return None

    if kind == 'card_effect_change':
        d = patch.get('deck')
        if d not in ('chance', 'chest'):
            return f'card_effect_change.deck={d!r} not in {{"chance","chest"}}'
        idx = patch.get('index')
        deck = (cfg.chance if d == 'chance' else cfg.chest).cards
        if not isinstance(idx, int) or not (0 <= idx < len(deck)):
            return f'card_effect_change.index={idx!r} out of range [0,{len(deck)})'
        new_text = patch.get('new_text', '')
        if not isinstance(new_text, str):
            return 'card_effect_change.new_text must be str'
        if len(new_text) > 200:
            return f'card_effect_change.new_text length {len(new_text)} > 200'
        if not any(re.match(p, new_text) for p in _ALLOWED_CARD_TOKEN_PATTERNS):
            return ('card_effect_change.new_text does not match any allowed '
                    f'card-action token grammar (got {new_text!r})')
        return None

    if kind == 'property_payout_mult':
        g = patch.get('group')
        groups = set(_board_groups(cfg))
        if g not in groups:
            return f'property_payout_mult.group={g!r} not on board {sorted(groups)}'
        rm = patch.get('rent_multiplier')
        if not isinstance(rm, (int, float)):
            return 'property_payout_mult.rent_multiplier must be numeric'
        return _bounded(float(rm), *_BOUNDS['rent_multiplier'], 'rent_multiplier')

    return f'unhandled kind {kind!r}'   # defensive; should be caught above


# --------------------------------------------------------------------------- #
# Application                                                                   #
# --------------------------------------------------------------------------- #

def apply_patch(cfg: GameConfig, patch: dict) -> GameConfig:
    """Return a deep-copied cfg with `patch` applied. Caller MUST validate
    first; passing an invalid patch raises ValueError."""
    err = validate_patch(patch, cfg)
    if err is not None:
        raise ValueError(f'cannot apply invalid patch: {err}')
    out = deepcopy(cfg)
    kind = patch['kind']
    m = out.settings.mechanics

    if kind == 'salary_change':
        out.settings = _replace(
            out.settings,
            mechanics=_replace(m, salary=patch['value']))
    elif kind == 'tax_change':
        if patch['tax_kind'] == 'luxury':
            out.settings = _replace(
                out.settings,
                mechanics=_replace(m, luxury_tax=patch['value']))
        else:
            out.settings = _replace(
                out.settings,
                mechanics=_replace(m, income_tax=patch['value']))
    elif kind == 'income_tax_pct':
        out.settings = _replace(
            out.settings,
            mechanics=_replace(m, income_tax_percentage=float(patch['value'])))
    elif kind == 'jail_fine_change':
        out.settings = _replace(
            out.settings,
            mechanics=_replace(m, exit_jail_fine=patch['value']))
    elif kind == 'free_parking_jackpot':
        out.settings = _replace(
            out.settings,
            mechanics=_replace(m, free_parking_money=patch['enabled']))
    elif kind == 'card_effect_change':
        deck = out.chance if patch['deck'] == 'chance' else out.chest
        new_cards = list(deck.cards)
        new_cards[patch['index']] = patch['new_text']
        if patch['deck'] == 'chance':
            from monopoly.core.deck import Deck
            out.chance = Deck(new_cards)
        else:
            from monopoly.core.deck import Deck
            out.chest = Deck(new_cards)
    elif kind == 'property_payout_mult':
        rm = float(patch['rent_multiplier'])
        new_cells = []
        for c in out.cells:
            if isinstance(c, Property) and c.group == patch['group']:
                new_cells.append(Property(
                    name=c.name,
                    cost_base=c.cost_base,
                    rent_base=int(round(c.rent_base * rm)),
                    cost_house=c.cost_house,
                    rent_house=tuple(int(round(r * rm)) for r in c.rent_house),
                    group=c.group,
                ))
            else:
                new_cells.append(c)
        out.cells = new_cells
    return out


def apply_patches(cfg: GameConfig, patches: List[dict]) -> GameConfig:
    """Apply patches in order. Each must validate against the cfg-after-
    previous-patches; a failure halts and surfaces the offending patch index.
    """
    out = cfg
    for i, p in enumerate(patches):
        err = validate_patch(p, out)
        if err is not None:
            raise ValueError(f'patch {i}: {err}')
        out = apply_patch(out, p)
    return out


# --------------------------------------------------------------------------- #
# Diff rendering (cosmetic only — for the §5f report table)                    #
# --------------------------------------------------------------------------- #

def render_patch_as_diff(patch: dict) -> str:
    """Render a structured patch as a unified-diff-flavoured string.

    NOT a real patch a human could `patch -p1` apply: the goal is presentation
    in the report, where a reader scanning §5f's "top-3 rule diffs per loop"
    table sees Python-shaped before/after text rather than a raw JSON record.
    """
    kind = patch.get('kind', '?')
    if kind == 'salary_change':
        return ('--- a/settings.py\n+++ b/settings.py\n'
                f'-    salary: int = <prev>\n+    salary: int = {patch["value"]}\n')
    if kind == 'tax_change':
        f = 'luxury_tax' if patch['tax_kind'] == 'luxury' else 'income_tax'
        return (f'--- a/settings.py\n+++ b/settings.py\n'
                f'-    {f}: int = <prev>\n+    {f}: int = {patch["value"]}\n')
    if kind == 'income_tax_pct':
        return ('--- a/settings.py\n+++ b/settings.py\n'
                f'-    income_tax_percentage: float = <prev>\n'
                f'+    income_tax_percentage: float = {float(patch["value"]):.3f}\n')
    if kind == 'jail_fine_change':
        return ('--- a/settings.py\n+++ b/settings.py\n'
                f'-    exit_jail_fine: int = <prev>\n+    exit_jail_fine: int = {patch["value"]}\n')
    if kind == 'free_parking_jackpot':
        return ('--- a/settings.py\n+++ b/settings.py\n'
                f'-    free_parking_money: bool = <prev>\n'
                f'+    free_parking_money: bool = {patch["enabled"]}\n')
    if kind == 'card_effect_change':
        return (f'--- a/configs/.../{patch["deck"]}.yaml\n'
                f'+++ b/configs/.../{patch["deck"]}.yaml\n'
                f'-  - <prev card text at index {patch["index"]}>\n'
                f'+  - {patch["new_text"]}\n')
    if kind == 'property_payout_mult':
        return (f'--- a/configs/.../properties.yaml\n'
                f'+++ b/configs/.../properties.yaml\n'
                f'  # group {patch["group"]}: rent x {float(patch["rent_multiplier"]):.2f}\n')
    return f'# unrecognised patch kind: {kind!r}\n'


def summarise_patch(patch: dict) -> str:
    """One-line summary for the loop trajectory log."""
    k = patch.get('kind')
    if k == 'salary_change':       return f'salary -> {patch["value"]}'
    if k == 'tax_change':          return f'{patch["tax_kind"]} tax -> {patch["value"]}'
    if k == 'income_tax_pct':      return f'income_tax_pct -> {float(patch["value"]):.3f}'
    if k == 'jail_fine_change':    return f'jail_fine -> {patch["value"]}'
    if k == 'free_parking_jackpot': return f'free_parking_jackpot={patch["enabled"]}'
    if k == 'card_effect_change':  return f'{patch["deck"]} card[{patch["index"]}] := {patch["new_text"][:50]}'
    if k == 'property_payout_mult': return f'rent x{float(patch["rent_multiplier"]):.2f} on {patch["group"]}'
    return f'<{k}>'


# --------------------------------------------------------------------------- #
# Persistent rejected corpus (CEO #8)                                          #
# --------------------------------------------------------------------------- #

REJECTED_CORPUS_PATH = Path('report/figures/llm_rules/rejected_corpus.jsonl')


def append_rejection(rejection: RejectedPatch,
                     path: Path = REJECTED_CORPUS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as fh:
        fh.write(json.dumps(rejection.to_dict()) + '\n')


# --------------------------------------------------------------------------- #
# LLM response parsing                                                         #
# --------------------------------------------------------------------------- #

_PATCH_BLOCK = re.compile(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```',
                          re.DOTALL | re.IGNORECASE)


def parse_llm_response(text: str) -> Tuple[Optional[List[dict]], Optional[str]]:
    """Pull a list of patches out of an LLM response.

    Accepts:
      - a fenced ```json``` block containing either a single object or a list
      - a bare JSON object/list as the entire response (no fence)

    Returns (patches, error). On success patches is a list[dict] (length >= 1).
    On failure error is a one-line reason describing the parse problem.
    """
    text = text.strip()
    candidates: List[str] = []
    for m in _PATCH_BLOCK.finditer(text):
        candidates.append(m.group(1))
    if not candidates:
        candidates.append(text)

    last_err = 'no parseable JSON in response'
    for c in candidates:
        c = c.strip()
        try:
            obj = json.loads(c)
        except json.JSONDecodeError as ex:
            last_err = f'JSONDecodeError: {ex.msg} at line {ex.lineno} col {ex.colno}'
            continue
        if isinstance(obj, dict):
            return [obj], None
        if isinstance(obj, list):
            if not all(isinstance(x, dict) for x in obj):
                return None, 'list contains non-dict element'
            if not obj:
                return None, 'empty patch list'
            return obj, None
        return None, f'top-level JSON must be dict or list, got {type(obj).__name__}'
    return None, last_err
