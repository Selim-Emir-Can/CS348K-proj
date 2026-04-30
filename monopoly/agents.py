"""Agent population for the Monopoly optimizer.

Three agents in increasing order of skill:
  RandomPlayer    -- buys each unowned property with fixed probability
  RuleBasedPlayer -- always buy, always build (via RuleBasedPlayerSettings in settings.py)
  (DQN agent in Week 3)

Settings classes live in settings.py alongside StandardPlayerSettings.
Player subclasses (requiring custom logic beyond settings) live here.
"""
import random as _random

from monopoly.core.constants import RAILROADS, UTILITIES
from monopoly.core.player import Player
from player_settings import ParametricPlayerSettings, RandomPlayerSettings


_BUILD_THRESHOLDS = [500, 200, 0]  # indexed by action // 2


class DQNPlayer(Player):
    """Player whose buy and build decisions are injected by an external RL agent.

    Action encoding (Discrete 6):
      action % 2  -> buy decision (0=pass, 1=buy)
      action // 2 -> build level  (0=hold $500, 1=moderate $200, 2=aggressive $0)

    Set player._buy_action and player._build_threshold before calling make_a_move().
    """

    def __init__(self, name: str, settings=None):
        super().__init__(name, settings or RandomPlayerSettings())
        self._buy_action: int = 1       # overwritten each step by the env
        self._build_threshold: int = 200  # overwritten each step by the env

    @property
    def _improve_cash_floor(self) -> int:
        return self._build_threshold

    def _should_buy(self, property_to_buy) -> bool:
        if property_to_buy.cost_base > self.money:
            return False
        if self.money - property_to_buy.cost_base < self.settings.unspendable_cash:
            return False
        return bool(self._buy_action)


class ParametricPlayer(Player):
    """Rule-based player with the full parametric knob-set for the optimiser.

    Every decision point the optimiser might want to vary is gated by a field on
    ``ParametricPlayerSettings``. Behaviours not yet expressible in the base
    Player (aggressive-vs-one-at-a-time building, skip-utilities/skip-railroads,
    early jail exit) are implemented here; everything else (trading thresholds,
    unspendable_cash, ignore_property_groups) is inherited from
    StandardPlayerSettings semantics via base-class logic.
    """

    def __init__(self, name: str, settings=None):
        super().__init__(name, settings or ParametricPlayerSettings())

    # ------------------------------------------------------------------ #
    # Buy decision                                                          #
    # ------------------------------------------------------------------ #
    def _should_buy(self, property_to_buy) -> bool:
        # Base affordability + ignored-groups check from parent
        if property_to_buy.cost_base > self.money:
            return False
        if self.money - property_to_buy.cost_base < self.settings.unspendable_cash:
            return False
        if property_to_buy.group in self.settings.ignore_property_groups:
            return False
        # Parametric knobs
        if property_to_buy.group == UTILITIES and not getattr(
                self.settings, 'buy_utilities', True):
            return False
        if property_to_buy.group == RAILROADS and not getattr(
                self.settings, 'buy_railroads', True):
            return False
        return True

    # ------------------------------------------------------------------ #
    # Build decision                                                        #
    # ------------------------------------------------------------------ #
    @property
    def _improve_cash_floor(self) -> int:
        return getattr(self.settings, 'build_cash_floor',
                       self.settings.unspendable_cash)

    def improve_properties(self, board, log):
        """If aggressive_build is False, build at most 1 house per turn.

        This is a targeted override of ``Player.improve_properties`` (core/player.py).
        When aggressive, we just delegate to super(). Otherwise we inline the
        one-iteration version, kept in sync with core/player.py:528-602.
        """
        if getattr(self.settings, 'aggressive_build', True):
            return super().improve_properties(board, log)

        # Non-aggressive: exactly one build attempt per turn.
        from monopoly.core.constants import RAILROADS as _RR, UTILITIES as _UT
        can_be_improved = []
        for cell in self.owned:
            if (cell.has_hotel == 0
                    and not cell.is_mortgaged
                    and cell.monopoly_multiplier == 2
                    and cell.group not in (_RR, _UT)):
                eligible = True
                for other_cell in board.groups[cell.group]:
                    if ((other_cell.has_houses < cell.has_houses
                         and not other_cell.has_hotel)
                            or other_cell.is_mortgaged):
                        eligible = False
                        break
                if eligible:
                    if cell.has_houses != 4 and board.available_houses > 0 or \
                            cell.has_houses == 4 and board.available_hotels > 0:
                        can_be_improved.append(cell)
        if not can_be_improved:
            return
        can_be_improved.sort(key=lambda x: x.cost_house)
        cell_to_improve = can_be_improved[0]
        if self.money - cell_to_improve.cost_house < self._improve_cash_floor:
            return
        ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
        if cell_to_improve.has_houses != 4:
            cell_to_improve.has_houses += 1
            board.available_houses -= 1
            self.money -= cell_to_improve.cost_house
            log.add(f"{self} built {ordinal[cell_to_improve.has_houses]} "
                    f"house on {cell_to_improve} for ${cell_to_improve.cost_house}")
        else:
            cell_to_improve.has_houses = 0
            cell_to_improve.has_hotel = 1
            board.available_houses += 4
            board.available_hotels -= 1
            self.money -= cell_to_improve.cost_house
            log.add(f"{self} built a hotel on {cell_to_improve}")

    # ------------------------------------------------------------------ #
    # Jail decision                                                         #
    # ------------------------------------------------------------------ #
    def handle_jail(self, dice_roll_is_double, board, log):
        """Extend base jail handling with an early-pay option.

        Parent (`core/player.py:handle_jail`) only pays the fine after the
        mandatory 2-turn wait. If ``jail_pay_threshold`` is set and the player
        has comfortably more cash than that threshold, pay on day 0 or 1 too
        — it's strategically useful when opponents have few developed cells.
        """
        from settings import GameMechanics
        threshold = getattr(self.settings, 'jail_pay_threshold', None)
        if (threshold is not None
                and self.in_jail
                and not self.get_out_of_jail_chance
                and not self.get_out_of_jail_comm_chest
                and not dice_roll_is_double
                and self.days_in_jail < 2
                and self.money > threshold + GameMechanics.exit_jail_fine):
            log.add(f"{self} chooses to pay ${GameMechanics.exit_jail_fine} "
                    f"and exit jail early (cash > threshold={threshold})")
            self.pay_money(GameMechanics.exit_jail_fine, "bank", board, log)
            self.in_jail = False
            self.days_in_jail = 0
            return False
        return super().handle_jail(dice_roll_is_double, board, log)


class LLMPlayer(Player):
    """Monopoly player whose buy decisions are queried from an LLM.

    Default backend: a local Qwen2.5-1.5B-Instruct model via huggingface
    transformers. The model is loaded lazily on the first decision so
    constructing an LLMPlayer is cheap, and is shared across instances
    via a module-level cache so multi-game runs don't re-load weights.

    For build / trade / jail decisions we fall back to base-Player logic:
    LLM latency on every micro-decision would make games take minutes
    instead of milliseconds, and the buy decision is the strategically
    decisive one anyway.

    Backend selection (decided at construction):
      - backend='local'     (default): use transformers + Qwen locally.
      - backend='openai'    : POST to an OpenAI-compatible endpoint.
      - backend='heuristic' : skip the LLM entirely; same as RuleBasedPlayer
        with extra logging hooks. Useful for unit tests.

    Environment variables (only read in the matching backend):
      LLM_MODEL            (default: 'Qwen/Qwen2.5-1.5B-Instruct')
      LLM_CACHE_DIR        Folder for HF model weights. Default: 'models/hf_cache'
                           relative to CWD, so weights land inside the project
                           rather than the user's global ~/.cache/huggingface.
      LLM_OPENAI_BASE_URL  (e.g. 'http://localhost:11434/v1' for ollama)
      LLM_OPENAI_API_KEY
      LLM_OPENAI_MODEL     (e.g. 'qwen2.5:1.5b' for ollama)
    """

    _MODEL_CACHE: dict = {}

    def __init__(self, name: str, settings=None, backend: str = 'local',
                 model_name: str = None, max_new_tokens: int = 160,
                 decision_log_path: str = None, decision_log_meta: dict = None):
        from player_settings import StandardPlayerSettings
        super().__init__(name, settings or StandardPlayerSettings())
        self._backend = backend
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        # Per-instance counter for diagnostic logging.
        self._n_buy_decisions = 0
        self._n_buy_yes = 0
        # Refs cached at the start of each turn by make_a_move so the buy
        # prompt can reference board state and opponents.
        self._board_ref = None
        self._players_ref = None
        # Optional per-decision JSONL log — append one record per _should_buy
        # call (prefilter PASSes included), so post-hoc analysis can examine
        # the prompt, raw response, parse path, and timing of every LLM call.
        self._decision_log_path = decision_log_path
        self._decision_log_meta = dict(decision_log_meta or {})
        self._decision_idx = 0
        self._round_n = 0

    def make_a_move(self, board, players, dice, log):
        # Cache refs so _should_buy / _build_buy_prompt can include
        # board state in the LLM query.
        self._board_ref = board
        self._players_ref = players
        return super().make_a_move(board, players, dice, log)

    # ------------------------------------------------------------------ #
    # Backend                                                              #
    # ------------------------------------------------------------------ #

    def _get_local_model(self):
        import os
        from pathlib import Path
        model_name = self._model_name or os.environ.get(
            'LLM_MODEL', 'Qwen/Qwen2.5-1.5B-Instruct')
        if model_name in self._MODEL_CACHE:
            return self._MODEL_CACHE[model_name]
        # If model_name looks like a path on disk, load from there directly
        # (no HF hub interaction). Otherwise treat it as a hub repo id and
        # use a project-local HF cache so weights don't pollute the global
        # ~/.cache/huggingface and are easy to wipe / .gitignore.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        is_local = Path(model_name).is_dir()
        if is_local:
            from_pretrained_kw = {}
        else:
            cache_dir = os.environ.get('LLM_CACHE_DIR', 'models/hf_cache')
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            from_pretrained_kw = {'cache_dir': cache_dir}
        tok = AutoTokenizer.from_pretrained(model_name, **from_pretrained_kw)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load in float16 on GPU for speed; default precision on CPU.
        kw = {'torch_dtype': torch.float16} if device == 'cuda' else {}
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **from_pretrained_kw, **kw).to(device)
        model.eval()
        self._MODEL_CACHE[model_name] = (tok, model, device)
        return self._MODEL_CACHE[model_name]

    # ------------------------------------------------------------------ #
    # Prompting                                                            #
    # ------------------------------------------------------------------ #

    _SYSTEM_PROMPT = (
        "You are a strategic Monopoly player who decides whether to buy "
        "properties. Your goal is to win by bankrupting opponents through "
        "rent. You must NOT mindlessly buy every property: passing is "
        "correct when (a) cash is dangerously low, (b) the property is "
        "in a group an opponent already dominates and you have no path "
        "to monopoly, or (c) the rent return is poor relative to the "
        "cost. Equally, buying is correct when the property advances a "
        "group you already own most of, or when it denies the group to "
        "an opponent who would otherwise complete it.\n\n"
        "You receive a STATE block with 10 fields. Before reasoning, you "
        "MUST emit an ECHO block that copies all 10 STATE fields verbatim "
        "(numeric values must equal STATE; string values must equal "
        "STATE). The ECHO block is checked deterministically — if it "
        "differs from STATE in any field you will be asked to redo the "
        "answer. Only after the ECHO block do you write REASON and "
        "ANSWER. Do not invent any number or name not present in STATE. "
        "A purchase ONLY 'completes a monopoly' when "
        "you_own_in_group + 1 == group_size; do not claim completion "
        "otherwise.\n\n"
        "Reply in EXACTLY this format and order:\n"
        "  ECHO:\n"
        "    cash: $<value>\n"
        "    property: <name>\n"
        "    group: <name>\n"
        "    cost: $<value>\n"
        "    base_rent: $<value>\n"
        "    you_own_total: <value>\n"
        "    you_own_in_group: <value>\n"
        "    group_size: <value>\n"
        "    opp_own_in_group: <value>\n"
        "    opp_own_total: <value>\n"
        "  REASON: <one short sentence citing only STATE numbers>\n"
        "  ANSWER: BUY\n"
        "or\n"
        "  ... (same ECHO block) ...\n"
        "  REASON: <one short sentence>\n"
        "  ANSWER: PASS\n"
        "Do not output anything else."
    )

    _FEW_SHOT = [
        # Example 1 — clear BUY: cheap, plenty of cash, no opponent ownership.
        {"role": "user", "content":
            "STATE:\n"
            "  cash: $1200\n"
            "  property: B1 Oriental Avenue\n"
            "  group: Lightblue\n"
            "  cost: $100\n"
            "  base_rent: $6\n"
            "  you_own_total: 0\n"
            "  you_own_in_group: 0\n"
            "  group_size: 3\n"
            "  opp_own_in_group: 0\n"
            "  opp_own_total: 2"},
        {"role": "assistant", "content":
            "ECHO:\n"
            "  cash: $1200\n"
            "  property: B1 Oriental Avenue\n"
            "  group: Lightblue\n"
            "  cost: $100\n"
            "  base_rent: $6\n"
            "  you_own_total: 0\n"
            "  you_own_in_group: 0\n"
            "  group_size: 3\n"
            "  opp_own_in_group: 0\n"
            "  opp_own_total: 2\n"
            "REASON: I have $1200, cost is only $100, and opp_own_in_group is 0 "
            "so I can still monopolise this group.\nANSWER: BUY"},
        # Example 2 — PASS due to opponent dominance (cash is fine).
        {"role": "user", "content":
            "STATE:\n"
            "  cash: $500\n"
            "  property: G1 Pacific Avenue\n"
            "  group: Green\n"
            "  cost: $300\n"
            "  base_rent: $26\n"
            "  you_own_total: 8\n"
            "  you_own_in_group: 0\n"
            "  group_size: 3\n"
            "  opp_own_in_group: 2\n"
            "  opp_own_total: 5"},
        {"role": "assistant", "content":
            "ECHO:\n"
            "  cash: $500\n"
            "  property: G1 Pacific Avenue\n"
            "  group: Green\n"
            "  cost: $300\n"
            "  base_rent: $26\n"
            "  you_own_total: 8\n"
            "  you_own_in_group: 0\n"
            "  group_size: 3\n"
            "  opp_own_in_group: 2\n"
            "  opp_own_total: 5\n"
            "REASON: opp_own_in_group is 2 of 3 so opponent already controls "
            "most of Green; spending $300 hurts liquidity for no monopoly path."
            "\nANSWER: PASS"},
        # Example 3 — BUY to complete one's own monopoly.
        {"role": "user", "content":
            "STATE:\n"
            "  cash: $700\n"
            "  property: C2 States Avenue\n"
            "  group: Pink\n"
            "  cost: $140\n"
            "  base_rent: $10\n"
            "  you_own_total: 6\n"
            "  you_own_in_group: 2\n"
            "  group_size: 3\n"
            "  opp_own_in_group: 0\n"
            "  opp_own_total: 4"},
        {"role": "assistant", "content":
            "ECHO:\n"
            "  cash: $700\n"
            "  property: C2 States Avenue\n"
            "  group: Pink\n"
            "  cost: $140\n"
            "  base_rent: $10\n"
            "  you_own_total: 6\n"
            "  you_own_in_group: 2\n"
            "  group_size: 3\n"
            "  opp_own_in_group: 0\n"
            "  opp_own_total: 4\n"
            "REASON: you_own_in_group=2 and group_size=3 so I am one short; "
            "this purchase completes my Pink monopoly.\nANSWER: BUY"},
        # Example 4 — PASS due to genuinely low cash AND opponent has nothing
        # in this group. Designed so the model has a "low-cash PASS" template
        # that does NOT require citing opponent ownership.
        {"role": "user", "content":
            "STATE:\n"
            "  cash: $80\n"
            "  property: H2 Boardwalk\n"
            "  group: Indigo\n"
            "  cost: $400\n"
            "  base_rent: $50\n"
            "  you_own_total: 4\n"
            "  you_own_in_group: 0\n"
            "  group_size: 2\n"
            "  opp_own_in_group: 0\n"
            "  opp_own_total: 3"},
        {"role": "assistant", "content":
            "ECHO:\n"
            "  cash: $80\n"
            "  property: H2 Boardwalk\n"
            "  group: Indigo\n"
            "  cost: $400\n"
            "  base_rent: $50\n"
            "  you_own_total: 4\n"
            "  you_own_in_group: 0\n"
            "  group_size: 2\n"
            "  opp_own_in_group: 0\n"
            "  opp_own_total: 3\n"
            "REASON: cash is $80 and cost is $400 so I cannot afford this "
            "without going broke, regardless of the opponent.\nANSWER: PASS"},
    ]

    def _query_local(self, prompt: str) -> str:
        import torch
        tok, model, device = self._get_local_model()
        msgs = [{'role': 'system', 'content': self._SYSTEM_PROMPT}]
        msgs.extend(self._FEW_SHOT)
        msgs.append({'role': 'user', 'content': prompt})
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors='pt').to(device)
        # Stop as soon as the model emits the chat-template's user-turn
        # marker (so it can't hallucinate a fake new user message after
        # its answer). Falling back to eos for older templates.
        eos_ids = [tok.eos_token_id]
        for marker in ('<|im_end|>', '<|endoftext|>'):
            tid = tok.convert_tokens_to_ids(marker)
            if tid is not None and tid != tok.unk_token_id:
                eos_ids.append(tid)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=eos_ids,
            )
        gen = tok.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return gen.strip()

    def _query_local_traced(self, prompt: str, prefix_msgs=None,
                             tail_msgs=None):
        """Same as _query_local but additionally returns timing + the rendered
        chat messages + a generation-meta dict, for the decision-log path.

        Kept as a sibling of _query_local (rather than changing _query_local's
        return shape) so existing callers — including callers from other
        scripts that import this method directly — continue to receive a
        plain ``str``. Only the logging branch in ``_should_buy`` calls this.

        ``prefix_msgs``: optional list of ``{role, content}`` dicts spliced
        BEFORE the runtime user prompt (after system + few-shot). Rarely
        used; kept for API symmetry.
        ``tail_msgs``: optional list of dicts appended AFTER the runtime
        user prompt. Used by the retry-on-hallucination path to inject the
        model's first attempt as an assistant turn followed by a corrective
        user turn — that ordering is required for chat templates to treat
        the two as a real follow-up exchange (see
        notes/ if interested in why prefix_msgs alone broke retries).
        """
        import time
        import torch
        tok, model, device = self._get_local_model()
        msgs = [{'role': 'system', 'content': self._SYSTEM_PROMPT}]
        msgs.extend(self._FEW_SHOT)
        if prefix_msgs:
            msgs.extend(prefix_msgs)
        msgs.append({'role': 'user', 'content': prompt})
        if tail_msgs:
            msgs.extend(tail_msgs)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors='pt').to(device)
        eos_ids = [tok.eos_token_id]
        for marker in ('<|im_end|>', '<|endoftext|>'):
            tid = tok.convert_tokens_to_ids(marker)
            if tid is not None and tid != tok.unk_token_id:
                eos_ids.append(tid)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=eos_ids,
            )
        ms_elapsed = (time.perf_counter() - t0) * 1000.0
        gen = tok.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        gen_meta = {
            'model_id': self._model_name or 'Qwen/Qwen2.5-1.5B-Instruct',
            'backend': 'local',
            'max_new_tokens': self._max_new_tokens,
            'do_sample': False,
            'temperature': None,
            'eos_token_ids': eos_ids,
            'device': device,
            'prompt_token_count': int(inputs['input_ids'].shape[1]),
            'completion_token_count': int(out.shape[1] - inputs['input_ids'].shape[1]),
        }
        return gen.strip(), ms_elapsed, msgs, gen_meta

    def _query_openai_traced(self, prompt: str):
        """Sibling of _query_openai that also returns timing + msgs + meta."""
        import time
        import os, json, urllib.request
        base = os.environ.get('LLM_OPENAI_BASE_URL', 'http://localhost:11434/v1')
        key = os.environ.get('LLM_OPENAI_API_KEY', 'no-key')
        model = os.environ.get('LLM_OPENAI_MODEL', 'qwen2.5:1.5b')
        msgs = [{'role': 'system', 'content': self._SYSTEM_PROMPT}]
        msgs.extend(self._FEW_SHOT)
        msgs.append({'role': 'user', 'content': prompt})
        body = {
            'model': model,
            'messages': msgs,
            'max_tokens': self._max_new_tokens,
            'temperature': 0.0,
        }
        req = urllib.request.Request(
            f'{base}/chat/completions',
            data=json.dumps(body).encode('utf-8'),
            headers={'Content-Type': 'application/json',
                     'Authorization': f'Bearer {key}'},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
        ms_elapsed = (time.perf_counter() - t0) * 1000.0
        text = payload['choices'][0]['message']['content'].strip()
        gen_meta = {
            'model_id': model,
            'backend': 'openai',
            'max_new_tokens': self._max_new_tokens,
            'do_sample': False,
            'temperature': 0.0,
            'eos_token_ids': None,
            'device': None,
            'base_url': base,
        }
        return text, ms_elapsed, msgs, gen_meta

    def _query_openai(self, prompt: str) -> str:
        import os, json, urllib.request
        base = os.environ.get('LLM_OPENAI_BASE_URL', 'http://localhost:11434/v1')
        key = os.environ.get('LLM_OPENAI_API_KEY', 'no-key')
        model = os.environ.get('LLM_OPENAI_MODEL', 'qwen2.5:1.5b')
        msgs = [{'role': 'system', 'content': self._SYSTEM_PROMPT}]
        msgs.extend(self._FEW_SHOT)
        msgs.append({'role': 'user', 'content': prompt})
        body = {
            'model': model,
            'messages': msgs,
            'max_tokens': self._max_new_tokens,
            'temperature': 0.0,
        }
        req = urllib.request.Request(
            f'{base}/chat/completions',
            data=json.dumps(body).encode('utf-8'),
            headers={'Content-Type': 'application/json',
                     'Authorization': f'Bearer {key}'},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
        return payload['choices'][0]['message']['content'].strip()

    # ------------------------------------------------------------------ #
    # Decision                                                              #
    # ------------------------------------------------------------------ #

    def _build_buy_prompt(self, prop) -> str:
        owned_count = len(self.owned)
        same_group_self = sum(1 for c in self.owned if c.group == prop.group)
        # Opponent context (if board/players are cached by make_a_move)
        opp_in_group = 0
        opp_total = 0
        group_size = 0
        if self._board_ref is not None and self._players_ref is not None:
            from monopoly.core.cell import Property
            group_cells = [c for c in self._board_ref.cells
                           if isinstance(c, Property) and c.group == prop.group]
            group_size = len(group_cells)
            for other in self._players_ref:
                if other is self:
                    continue
                opp_in_group += sum(1 for c in other.owned if c.group == prop.group)
                opp_total    += len(other.owned)
        # Structured key:value block. Small LLMs read these far more
        # reliably than the prose form did — see notes/llm_hallucination_*.
        # Numeric fields are int-coerced before formatting because Player.money
        # can be a float in some games (rent×multiplier rounding, tax/fine
        # interactions). A float like 411.79999999999995 in the prompt makes
        # the model echo "$411.79999999999995", which then trips the echo
        # validator's int() parse — see the 2026-04-29 Task 1 post-mortem in
        # notes/. Coercing to int here means STATE always shows whole dollars.
        cash_i      = int(round(self.money))
        cost_i      = int(round(prop.cost_base))
        base_rent_i = int(round(prop.rent_base))
        return (
            "STATE:\n"
            f"  cash: ${cash_i}\n"
            f"  property: {prop.name}\n"
            f"  group: {prop.group}\n"
            f"  cost: ${cost_i}\n"
            f"  base_rent: ${base_rent_i}\n"
            f"  you_own_total: {owned_count}\n"
            f"  you_own_in_group: {same_group_self}\n"
            f"  group_size: {group_size}\n"
            f"  opp_own_in_group: {opp_in_group}\n"
            f"  opp_own_total: {opp_total}"
        )

    # ------------------------------------------------------------------ #
    # Hallucination detection                                              #
    # ------------------------------------------------------------------ #
    # Patterns we look for in the REASON sentence. Each entry is
    # (regex, predicate(ctx) -> bool, issue_label). The regex picks up
    # phrasings the model recycles from few-shot exemplars; the predicate
    # answers "is this claim contradicted by STATE?".
    #
    # NB: We deliberately keep this list short and conservative — false
    # positives mean wasted retries, but false negatives (missed
    # hallucinations) are not a correctness bug for the optimiser, just a
    # missed flag in the analysis. So when in doubt we DO NOT flag.
    import re as _re
    _HALLUC_PATTERNS = [
        # "opponent already owns all/all three/most of <group>" — claim that
        # opp_own_in_group is large (or equals group_size). Contradicts when
        # opp_own_in_group is actually 0.
        (_re.compile(
            r'\b(?:opp(?:onent)?|they)\s+(?:already\s+)?(?:owns?|holds?|controls?)\s+'
            r'(?:all(?:\s+\w+)?|every|most|two[-\s]?thirds|half|the\s+entire|'
            r'\d+\s+of\s+\d+|\d+/\d+)\b',
            _re.IGNORECASE),
         lambda ctx: ctx.get('opp_in_group', 0) == 0,
         'reason claims opponent ownership but opp_own_in_group=0'),
        # "I have no cash" / "cash is too low" / "cannot afford" when
        # cash >= cost AND cash >= 500 (well above any reasonable cash floor).
        (_re.compile(
            r'\b(?:i\s+have\s+no\s+cash|cash\s+is\s+(?:too\s+)?low|'
            r'low\s+(?:on\s+)?cash|cannot\s+afford|can[\'’]t\s+afford|'
            r'no\s+money|out\s+of\s+(?:cash|money))\b',
            _re.IGNORECASE),
         lambda ctx: ctx.get('cash', 0) >= max(500, ctx.get('cost', 0) + 200),
         'reason claims low cash but cash is plentiful'),
        # "completes my X monopoly" when you_own_in_group is NOT the
        # second-to-last in the group (i.e. group_size - you_own_in_group != 1).
        (_re.compile(r'\bcompletes?\s+(?:my|the)\s+\w*\s*monopoly', _re.IGNORECASE),
         lambda ctx: (ctx.get('group_size', 0) - ctx.get('same_group_self', 0)) != 1,
         'reason claims monopoly completion but you are not one short'),
    ]

    # ECHO-block parser: each line "  key: value" → (key, value). Tolerates
    # an optional leading "$" on numeric values, leading/trailing whitespace,
    # and case-insensitive keys. Crucially uses [ \t] (horizontal whitespace
    # only) around the colon so that `ECHO:\n  cash: $1500` does NOT match
    # ECHO=>"cash: $1500" — that bug ate the first echoed field on every
    # validation pass.
    _ECHO_LINE_RE = _re.compile(
        r'^[ \t]*([a-z_][a-z0-9_]*)[ \t]*:[ \t]*([^\n]+?)[ \t]*$',
        _re.MULTILINE | _re.IGNORECASE,
    )
    # Numeric ECHO fields and their corresponding context keys.
    _ECHO_NUM_FIELDS = (
        'cash', 'cost', 'base_rent',
        'you_own_total', 'you_own_in_group', 'group_size',
        'opp_own_in_group', 'opp_own_total',
    )
    _ECHO_STR_FIELDS = ('property', 'group')

    @staticmethod
    def _parse_echo(response: str) -> dict:
        """Pull every ``key: value`` pair from the response.

        Returns a dict (lower-cased keys → raw string values). The caller
        decides whether each value is parseable as int / matches STATE.
        Note this scans the *entire* response, including REASON and
        ANSWER lines — but those don't have ``key: value`` shapes (REASON
        and ANSWER values are sentences/words without colons embedded in
        a "key:value" pattern), so the only matches are the ECHO block.
        """
        out: dict = {}
        for k, v in LLMPlayer._ECHO_LINE_RE.findall(response or ''):
            key = k.lower()
            # Skip the response's own structural keys so they don't pollute
            # the dict (these aren't ECHO fields).
            if key in ('echo', 'reason', 'answer'):
                continue
            # First wins (small models occasionally repeat the block).
            if key not in out:
                out[key] = v.strip()
        return out

    @staticmethod
    def _check_echo(response: str, ctx: dict):
        """Validate the ECHO block against ground-truth STATE.

        Returns ``(issues: list[str], echoed: dict[str, raw])``. Empty
        ``issues`` means the model echoed all 10 fields with values that
        match STATE. Each entry in ``issues`` is a short sentence the
        retry path can paste verbatim into the corrective user turn.

        Numeric fields tolerate a leading ``$`` and minus sign. String
        fields must match exactly (case-sensitive — property/group names
        are canonical).
        """
        echoed = LLMPlayer._parse_echo(response)
        issues = []

        for k in LLMPlayer._ECHO_NUM_FIELDS:
            expected = ctx.get(k)
            if k not in echoed:
                issues.append(f'echo missing field {k!r} (STATE.{k}={expected})')
                continue
            raw = echoed[k]
            cleaned = raw.lstrip('$').replace('$', '').strip()
            # Try int first; fall through to float so legitimate float values
            # (e.g. cash=$411.79999... from rent×multiplier rounding) are
            # accepted as long as they round-match STATE within 0.5 dollars.
            got = None
            try:
                got = int(cleaned)
            except ValueError:
                try:
                    got = float(cleaned)
                except ValueError:
                    issues.append(
                        f'echo unparseable for {k!r}: {raw!r} (STATE.{k}={expected})')
                    continue
            if expected is None:
                issues.append(
                    f'echo mismatch on {k!r}: model echoed {got}, '
                    f'STATE.{k} is missing')
                continue
            # Compare with float tolerance — covers both pure-int matches
            # and legitimate float echoes (e.g. 411.8 vs 411.79999999999995).
            try:
                if abs(float(got) - float(expected)) > 0.5:
                    issues.append(
                        f'echo mismatch on {k!r}: model echoed {got}, '
                        f'STATE.{k}={expected}')
            except (TypeError, ValueError):
                issues.append(
                    f'echo unparseable for {k!r}: {raw!r} (STATE.{k}={expected})')

        for k in LLMPlayer._ECHO_STR_FIELDS:
            expected = ctx.get(k)
            if k not in echoed:
                issues.append(f'echo missing field {k!r} (STATE.{k}={expected!r})')
                continue
            got = echoed[k]
            if expected is None or got != expected:
                issues.append(
                    f'echo mismatch on {k!r}: model echoed {got!r}, '
                    f'STATE.{k}={expected!r}')

        return issues, echoed

    @staticmethod
    def _check_reason_consistency(reason_text: str, ctx: dict):
        """Return list[str] of detected contradictions; empty if clean.

        ``ctx`` keys used: ``cash``, ``cost``, ``opp_in_group``,
        ``same_group_self``, ``group_size``. Missing keys default to 0.
        """
        if not reason_text:
            return []
        issues = []
        for regex, predicate, label in LLMPlayer._HALLUC_PATTERNS:
            if regex.search(reason_text) and predicate(ctx):
                issues.append(label)
        return issues

    @staticmethod
    def _extract_reason(response: str):
        """Return the REASON sentence from the response, or None if absent.

        Uses the FIRST 'REASON:' line (case-insensitive) and stops at the next
        newline. Mirrors _parse_decision's first-occurrence rule so the
        decision and the reason are extracted from the same span of the model
        output.
        """
        if not response:
            return None
        upper = response.upper()
        idx = upper.find('REASON:')
        if idx < 0:
            return None
        tail = response[idx + len('REASON:'):]
        # Stop at first newline; trim whitespace.
        nl = tail.find('\n')
        sentence = tail if nl < 0 else tail[:nl]
        return sentence.strip()

    def _parse_decision_traced(self, response: str):
        """Like _parse_decision but also reports which branch produced the
        decision: 'first_answer_tag', 'last_token_fallback', or
        'default_buy'. Used only by the decision-log path.
        """
        upper = (response or '').upper()
        buy_idx  = upper.find('ANSWER: BUY')
        pass_idx = upper.find('ANSWER: PASS')
        if buy_idx >= 0 and (pass_idx < 0 or buy_idx < pass_idx):
            return True, 'first_answer_tag'
        if pass_idx >= 0 and (buy_idx < 0 or pass_idx < buy_idx):
            return False, 'first_answer_tag'
        last_buy  = upper.rfind('BUY')
        last_pass = upper.rfind('PASS')
        if last_buy < 0 and last_pass < 0:
            return True, 'default_buy'
        return (last_buy > last_pass), 'last_token_fallback'

    def _log_decision(self, record: dict) -> None:
        """Append one JSONL record to the configured decision_log_path.

        Failures are swallowed (with a single warning printed once) so that a
        broken log file never crashes a long simulation run.
        """
        path = self._decision_log_path
        if not path:
            return
        try:
            import json, os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as exc:
            if not getattr(self, '_decision_log_warned', False):
                print(f'[LLMPlayer] decision-log write failed for '
                      f'{path}: {exc}; further failures suppressed')
                self._decision_log_warned = True

    @staticmethod
    def _parse_decision(response: str) -> bool:
        """Extract the LLM's BUY/PASS decision from the structured response.

        We look for the FIRST occurrence of 'ANSWER: BUY' or 'ANSWER: PASS'
        (case-insensitive) because small models often hallucinate a fake
        new user turn after their answer, and that hallucinated text can
        contain the word 'buy' or 'pass' unrelated to the actual decision.
        Falls back to a last-token heuristic if neither structured form is
        found, then defaults to True (BUY) so a malformed response leaves
        the player playing roughly like a RuleBasedPlayer.
        """
        upper = response.upper()
        buy_idx  = upper.find('ANSWER: BUY')
        pass_idx = upper.find('ANSWER: PASS')
        if buy_idx >= 0 and (pass_idx < 0 or buy_idx < pass_idx):
            return True
        if pass_idx >= 0 and (buy_idx < 0 or pass_idx < buy_idx):
            return False
        # Fallback: no structured ANSWER tag — use last-token heuristic.
        last_buy  = upper.rfind('BUY')
        last_pass = upper.rfind('PASS')
        if last_buy < 0 and last_pass < 0:
            return True
        return last_buy > last_pass

    def _should_buy(self, property_to_buy) -> bool:
        # Logging branch: if a decision_log_path is set, route through the
        # traced query helpers and append one JSONL record per call. The
        # non-logging branch below is byte-identical to the pre-logging
        # behaviour so existing callers see no change.
        if self._decision_log_path:
            return self._should_buy_logged(property_to_buy)

        # Cheap pre-filter that avoids the LLM for trivially-impossible buys.
        if property_to_buy.cost_base > self.money:
            return False
        if self.money - property_to_buy.cost_base < self.settings.unspendable_cash:
            return False
        if property_to_buy.group in self.settings.ignore_property_groups:
            return False
        # Heuristic backend: shortcut to "always buy when affordable".
        if self._backend == 'heuristic':
            self._n_buy_decisions += 1
            self._n_buy_yes += 1
            return True

        prompt = self._build_buy_prompt(property_to_buy)
        try:
            if self._backend == 'openai':
                response = self._query_openai(prompt)
            else:
                response = self._query_local(prompt)
        except Exception:
            # Network / model failure: fall back to "buy if affordable".
            response = 'BUY'
        decision = self._parse_decision(response)
        self._n_buy_decisions += 1
        if decision:
            self._n_buy_yes += 1
        return decision

    def _should_buy_logged(self, property_to_buy) -> bool:
        """Logged variant of _should_buy. Same control flow as the original
        but writes one JSONL record per decision.

        Records prefilter PASSes (cant_afford / cash_floor / ignore_group)
        without a prompt or response — this lets the analyser count how
        many decisions were actually shown to the LLM versus short-circuited
        by the rule-based prefilter.
        """
        prop = property_to_buy
        # Build the same context fields the prompt would use, so the log row
        # is meaningful even when the LLM was never called (prefilter PASS).
        owned_count = len(self.owned)
        same_group_self = sum(1 for c in self.owned if c.group == prop.group)
        opp_in_group = 0
        opp_total = 0
        group_size = 0
        if self._board_ref is not None and self._players_ref is not None:
            from monopoly.core.cell import Property
            group_cells = [c for c in self._board_ref.cells
                           if isinstance(c, Property) and c.group == prop.group]
            group_size = len(group_cells)
            for other in self._players_ref:
                if other is self:
                    continue
                opp_in_group += sum(1 for c in other.owned if c.group == prop.group)
                opp_total    += len(other.owned)

        base_record = dict(self._decision_log_meta)
        base_record.update({
            'decision_idx': self._decision_idx,
            'player_name': self.name,
            'cash': self.money,
            'prop_name': prop.name,
            'prop_group': prop.group,
            'prop_cost': prop.cost_base,
            'prop_rent_base': prop.rent_base,
            'owned_total': owned_count,
            'same_group_self': same_group_self,
            'group_size': group_size,
            'opp_in_group': opp_in_group,
            'opp_total': opp_total,
        })
        self._decision_idx += 1

        # Prefilter PASSes — log and return without an LLM call.
        if prop.cost_base > self.money:
            base_record.update({'prefilter': 'cant_afford', 'parsed': 'PASS',
                                'reason_text': None, 'fallback_used': False,
                                'parse_path': None,
                                'prompt': None, 'raw_response': None,
                                'ms_elapsed': 0.0, 'gen_meta': None})
            self._log_decision(base_record)
            return False
        if self.money - prop.cost_base < self.settings.unspendable_cash:
            base_record.update({'prefilter': 'cash_floor', 'parsed': 'PASS',
                                'reason_text': None, 'fallback_used': False,
                                'parse_path': None,
                                'prompt': None, 'raw_response': None,
                                'ms_elapsed': 0.0, 'gen_meta': None})
            self._log_decision(base_record)
            return False
        if prop.group in self.settings.ignore_property_groups:
            base_record.update({'prefilter': 'ignore_group', 'parsed': 'PASS',
                                'reason_text': None, 'fallback_used': False,
                                'parse_path': None,
                                'prompt': None, 'raw_response': None,
                                'ms_elapsed': 0.0, 'gen_meta': None})
            self._log_decision(base_record)
            return False

        # Heuristic backend: still log, but no LLM call.
        if self._backend == 'heuristic':
            self._n_buy_decisions += 1
            self._n_buy_yes += 1
            base_record.update({'prefilter': 'sent_to_llm', 'parsed': 'BUY',
                                'reason_text': 'heuristic backend',
                                'fallback_used': False,
                                'parse_path': 'heuristic',
                                'prompt': None, 'raw_response': None,
                                'ms_elapsed': 0.0, 'gen_meta': {'backend': 'heuristic'}})
            self._log_decision(base_record)
            return True

        prompt = self._build_buy_prompt(prop)

        # Ground-truth context for echo validation. Keys must match
        # _ECHO_NUM_FIELDS / _ECHO_STR_FIELDS in LLMPlayer above.
        echo_ctx = {
            'cash':              self.money,
            'cost':              prop.cost_base,
            'base_rent':         prop.rent_base,
            'you_own_total':     owned_count,
            'you_own_in_group':  same_group_self,
            'group_size':        group_size,
            'opp_own_in_group':  opp_in_group,
            'opp_own_total':     opp_total,
            'property':          prop.name,
            'group':             prop.group,
        }

        # Run up to MAX_ATTEMPTS = 1 initial + MAX_RETRIES = 4 retries on echo
        # mismatches. The LAST attempt's parsed answer is the decision; all
        # attempts go into the JSONL so the analyser can recover the trace.
        MAX_RETRIES = 4
        attempts = []
        api_error = None
        first_msgs = None
        first_gen_meta = None
        # First attempt — no tail messages.
        try:
            if self._backend == 'openai':
                response, ms_elapsed, msgs, gen_meta = (
                    self._query_openai_traced(prompt))
            else:
                response, ms_elapsed, msgs, gen_meta = (
                    self._query_local_traced(prompt))
        except Exception as exc:
            response = 'BUY'
            ms_elapsed = 0.0
            msgs = None
            gen_meta = None
            api_error = repr(exc)
        first_msgs = msgs
        first_gen_meta = gen_meta
        decision, parse_path = self._parse_decision_traced(response)
        reason_text = self._extract_reason(response)
        echo_issues, echoed_values = self._check_echo(response, echo_ctx)
        attempts.append({
            'attempt_idx':       0,
            'response':          response,
            'reason_text':       reason_text,
            'parsed':            'BUY' if decision else 'PASS',
            'parse_path':        parse_path,
            'echo_mismatches':   echo_issues,
            'echoed_values':     echoed_values,
            'ms_elapsed':        ms_elapsed,
        })

        # Retry loop. Each retry's tail re-uses ALL prior attempts so the
        # model sees its full history of wrong answers, plus the latest
        # corrective. We stop as soon as echo is clean OR retries exhausted.
        retry_idx = 0
        last_attempt = attempts[0]
        while last_attempt['echo_mismatches'] and retry_idx < MAX_RETRIES \
                and api_error is None:
            retry_idx += 1
            issues_str = '; '.join(last_attempt['echo_mismatches'])
            corrective = (
                f"Your previous response failed echo validation. "
                f"Mismatches: {issues_str}. Re-emit the FULL ECHO block "
                f"with values copied EXACTLY from STATE, then REASON, "
                f"then ANSWER. Do not invent any number or name."
            )
            # Build tail: replay each prior attempt's response and a single
            # corrective user message after the latest. The earlier
            # corrections persist as plain user turns to keep the history
            # honest, but only the final user message contains the
            # latest mismatches.
            tail = []
            for prior in attempts:
                tail.append({'role': 'assistant', 'content': prior['response']})
                if prior is attempts[-1]:
                    tail.append({'role': 'user', 'content': corrective})
                else:
                    tail.append({'role': 'user',
                                  'content': 'Your previous response failed '
                                             'echo validation. Try again.'})
            try:
                if self._backend == 'openai':
                    r_response, r_ms, _, _ = self._query_openai_traced(prompt)
                else:
                    r_response, r_ms, _, _ = (
                        self._query_local_traced(prompt, tail_msgs=tail))
            except Exception as exc:
                api_error = (api_error or '') + f' | retry-{retry_idx}-error: {exc!r}'
                break
            r_decision, r_parse_path = self._parse_decision_traced(r_response)
            r_reason = self._extract_reason(r_response)
            r_issues, r_echoed = self._check_echo(r_response, echo_ctx)
            attempts.append({
                'attempt_idx':       retry_idx,
                'response':          r_response,
                'reason_text':       r_reason,
                'parsed':            'BUY' if r_decision else 'PASS',
                'parse_path':        r_parse_path,
                'echo_mismatches':   r_issues,
                'echoed_values':     r_echoed,
                'ms_elapsed':        r_ms,
            })
            decision = r_decision
            last_attempt = attempts[-1]

        # Aggregate trace.
        n_retries = len(attempts) - 1
        final_resolved = bool(attempts) and not attempts[-1]['echo_mismatches']
        first = attempts[0]
        last = attempts[-1] if attempts else first

        self._n_buy_decisions += 1
        if decision:
            self._n_buy_yes += 1
        base_record.update({
            'prefilter': 'sent_to_llm',
            'prompt_text':       prompt,
            'prompt':            first_msgs,
            # Original (first attempt) — keeps the analyser's old field shape
            # working when it groups by raw_response / reason_text / parsed.
            'raw_response':      first['response'],
            'reason_text':       first['reason_text'],
            'parsed':            'BUY' if decision else 'PASS',
            'parse_path':        first['parse_path'],
            'fallback_used':     first['parse_path'] != 'first_answer_tag',
            'ms_elapsed':        first['ms_elapsed'],
            'gen_meta':          first_gen_meta,
            'api_error':         api_error,
            # Echo-validation trace.
            'echo_mismatches':           first['echo_mismatches'],
            'echoed_values':             first['echoed_values'],
            'hallucination_detected':    bool(first['echo_mismatches']),
            # Compatibility with the previous (pre-2026-04-28) hallucination_issues
            # field; lets older analysers keep counting "any flag" without code
            # changes.
            'hallucination_issues':      first['echo_mismatches'],
            # Retry summary.
            'retry_attempted':           n_retries > 0,
            'n_retries':                 n_retries,
            'final_resolved':            final_resolved,
            # The "retry_*" suffix mirrors the old single-retry schema and now
            # tracks the FINAL attempt (not necessarily index 1).
            'retry_response':            last['response'] if n_retries > 0 else None,
            'retry_reason_text':         last['reason_text'] if n_retries > 0 else None,
            'retry_parse_path':          last['parse_path'] if n_retries > 0 else None,
            'retry_ms_elapsed':          last['ms_elapsed'] if n_retries > 0 else 0.0,
            'retry_issues':              last['echo_mismatches'] if n_retries > 0 else None,
            'retry_resolved':            n_retries > 0 and final_resolved,
            # Full attempt list (excluding the prompt msgs to keep records small).
            'echo_attempts':             attempts,
        })
        self._log_decision(base_record)
        return decision


class RandomPlayer(Player):
    """Buys each unowned property it can afford with probability *buy_probability*.

    All other behaviour (building, trading, bankruptcy) inherits from Player.
    """

    def __init__(self, name: str, settings=None, buy_probability: float = 0.5,
                 seed: int = None):
        super().__init__(name, settings or RandomPlayerSettings())
        self._rng = _random.Random(seed)
        self._buy_probability = buy_probability

    def _should_buy(self, property_to_buy) -> bool:
        if property_to_buy.cost_base > self.money:
            return False
        return self._rng.random() < self._buy_probability
