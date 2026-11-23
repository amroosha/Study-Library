"""Improved Pokemon TCG agent for the Kaggle PTCG AI Battle Challenge Playground.

Deck: a Dragapult-ex evolution deck (see deck.csv, 60 cards, 25 distinct IDs).
Win condition: evolve Dreepy -> Drakloak -> Dragapult ex and attack with Phantom
Dive (200 damage + 6 damage counters spread on the opponent's Bench). Budew provides
a free early Itchy Pollen Item-lock; Drakloak/Dudunsparce provide draw/filtering.

Architecture
------------
1. A card-aware, context-aware rule-based policy is the reliable backbone. It
   handles every selection context the engine can emit (setup, switch, search,
   discard, damage-counter placement, energy discard, yes/no, counts, ...),
   evaluates the actual meaning of each legal option (which card, which target,
   which attack), and always returns a legal selection.
2. A conservative, time-budgeted Monte-Carlo lookahead (MC_ENABLED) can refine
   single-choice MAIN decisions. It uses the SDK search API with a *correct*
   hidden-state estimate (deck multiset subtraction), the same heuristic as the
   rollout policy, and a richer state evaluator. It only searches the top cluster
   of near-equivalent heuristic options and only overrides the heuristic when the
   rollouts show a meaningful advantage, so it never overturns a clearly-best
   heuristic choice. It is gated by the remaining overage-time bank and degrades
   gracefully to the heuristic on any failure.

Key strategic improvements over a type-priority baseline
--------------------------------------------------------
* Options are scored by what they DO, not a fixed action-type priority.
* Attack selection computes real damage (weakness/resistance), knockout value and
  prize trade, and prefers Phantom Dive's Bench spread; attack ends the turn, so
  beneficial setup is done first.
* Energy attachment is planned toward the Dragapult line's Fire+Psychic cost.
* Donk protection: with an empty Bench, fetching/playing a Basic is prioritised.
* Draw-supporter timing (Lillie/Judge) is based on current hand quality.
* Phantom Dive's 6 damage counters are allocated to maximise Bench knockouts.

Reliability
-----------
Every code path returns indices within [0, len(option)), respecting
minCount/maxCount with no duplicates. A global guard falls back to a safe legal
choice on any unexpected observation. The policy keeps no per-game mutable state,
so it is safe across Best-of-3 rounds.

Only agent.py is required; deck.csv is read unchanged. The ``cg`` engine package
is imported lazily (bundled with the submission on Kaggle); if its metadata is
unavailable a built-in fallback card database keeps the agent functional.
"""

import os
import random

# ---------------------------------------------------------------------------
# Enum constants (plain ints so the core policy does not depend on cg import).
# Values mirror cg.api enums.
# ---------------------------------------------------------------------------
# OptionType
OT_NUMBER, OT_YES, OT_NO, OT_CARD, OT_TOOL_CARD, OT_ENERGY_CARD, OT_ENERGY = 0, 1, 2, 3, 4, 5, 6
OT_PLAY, OT_ATTACH, OT_EVOLVE, OT_ABILITY, OT_DISCARD, OT_RETREAT, OT_ATTACK, OT_END = 7, 8, 9, 10, 11, 12, 13, 14
OT_SKILL, OT_SPECIAL_CONDITION = 15, 16
# AreaType
AR_DECK, AR_HAND, AR_DISCARD, AR_ACTIVE, AR_BENCH, AR_PRIZE = 1, 2, 3, 4, 5, 6
AR_STADIUM, AR_ENERGY, AR_TOOL, AR_PRE_EVOLUTION, AR_PLAYER, AR_LOOKING = 7, 8, 9, 10, 11, 12
# EnergyType
EN_COLORLESS, EN_GRASS, EN_FIRE, EN_WATER, EN_LIGHTNING, EN_PSYCHIC = 0, 1, 2, 3, 4, 5
EN_FIGHTING, EN_DARKNESS, EN_METAL, EN_DRAGON, EN_RAINBOW, EN_TEAM_ROCKET = 6, 7, 8, 9, 10, 11
# CardType
CT_POKEMON, CT_ITEM, CT_TOOL, CT_SUPPORTER, CT_STADIUM, CT_BASIC_ENERGY, CT_SPECIAL_ENERGY = 0, 1, 2, 3, 4, 5, 6
# SelectContext
CX_MAIN = 0
CX_SETUP_ACTIVE, CX_SETUP_BENCH, CX_SWITCH, CX_TO_ACTIVE, CX_TO_BENCH, CX_TO_FIELD = 1, 2, 3, 4, 5, 6
CX_TO_HAND, CX_DISCARD, CX_TO_DECK, CX_TO_DECK_BOTTOM, CX_TO_PRIZE, CX_NOT_MOVE = 7, 8, 9, 10, 11, 12
CX_DAMAGE_COUNTER, CX_DAMAGE_COUNTER_ANY, CX_DAMAGE, CX_REMOVE_DAMAGE_COUNTER, CX_HEAL = 13, 14, 15, 16, 17
CX_EVOLVES_FROM, CX_EVOLVES_TO, CX_DEVOLVE, CX_ATTACH_FROM, CX_ATTACH_TO, CX_DETACH_FROM = 18, 19, 20, 21, 22, 23
CX_LOOK, CX_EFFECT_TARGET = 24, 25
CX_DISCARD_ENERGY_CARD, CX_DISCARD_TOOL_CARD, CX_SWITCH_ENERGY_CARD, CX_DISCARD_CARD_OR_ATTACHED = 26, 27, 28, 29
CX_DISCARD_ENERGY, CX_TO_HAND_ENERGY, CX_TO_DECK_ENERGY, CX_SWITCH_ENERGY = 30, 31, 32, 33
CX_SKILL_ORDER, CX_ATTACK, CX_DISABLE_ATTACK, CX_EVOLVE = 34, 35, 36, 37
CX_DRAW_COUNT, CX_DAMAGE_COUNTER_COUNT, CX_REMOVE_DAMAGE_COUNTER_COUNT = 38, 39, 40
CX_IS_FIRST, CX_MULLIGAN, CX_ACTIVATE, CX_FIRST_EFFECT, CX_MORE_DEVOLVE, CX_COIN_HEAD = 41, 42, 43, 44, 45, 46
CX_AFFECT_SPECIAL_CONDITION, CX_RECOVER_SPECIAL_CONDITION = 47, 48

# ---------------------------------------------------------------------------
# Deck-specific strategic knowledge. Card IDs are fixed by deck.csv.
# ---------------------------------------------------------------------------
DREEPY, DRAKLOAK, DRAGAPULT = 119, 120, 121
DUNSPARCE, DUDUNSPARCE, DUDUNSPARCE_EX = 65, 66, 306
MUNKIDORI, FEZANDIPIRI_EX, MEOWTH_EX, BUDEW = 112, 140, 1071, 235
FIRE_ENERGY, PSYCHIC_ENERGY, DARKNESS_ENERGY = 2, 5, 7
POFFIN, NIGHT_STRETCHER, CRUSHING_HAMMER, ULTRA_BALL, POKE_PAD = 1086, 1097, 1120, 1121, 1152
BOSS_ORDERS, CRISPIN, JUDGE, LILLIE, UNFAIR_STAMP, ROSA, RISKY_RUINS = 1182, 1198, 1213, 1227, 1080, 1240, 1260

DRAGAPULT_LINE = {DREEPY, DRAKLOAK, DRAGAPULT}
EVOLVES_FROM = {DRAKLOAK: DREEPY, DRAGAPULT: DRAKLOAK, DUDUNSPARCE: DUNSPARCE, DUDUNSPARCE_EX: DUNSPARCE}
# Attack IDs we care about (from cg.api.all_attack()).
ATK_PHANTOM_DIVE, ATK_JET_HEADBUTT = 154, 153
ATK_ITCHY_POLLEN, ATK_MIND_BEND, ATK_CRUEL_ARROW = 323, 141, 183
ATK_TENACIOUS_TAIL, ATK_DESTRUCTIVE_DRILL = 425, 426

# Turn-order preference. Going second lets Budew attack on turn 2 (free Itchy
# Pollen -> opponent Item-lock) and attacks first; validated empirically in self-
# play. Set False to prefer going first.
PREFER_GO_SECOND = True

# Monte-Carlo lookahead configuration.
# NOTE: MC is implemented and its hidden-state construction is verified correct,
# but A/B self-play vs the baseline showed no reliable win-rate gain (short-horizon
# rollouts undervalue this deck's long-term energy/evolution setup and can override
# good heuristic decisions). It is therefore DISABLED by default; the strong
# heuristic is the reliable policy. Set MC_ENABLED=True to experiment.
MC_ENABLED = False
MC_MAX_OPTIONS = 14          # skip search if MAIN has more options than this
MC_ROLLOUTS_PER_OPTION = 4   # rollouts per candidate option
MC_DEPTH = 10                # max search steps per rollout
MC_TIME_CAP_S = 0.45         # hard per-decision wall-clock cap
MC_TIME_RESERVE_S = 60.0     # keep this much overage bank in reserve


# ---------------------------------------------------------------------------
# Metadata loading (authoritative from cg; resilient fallback for our deck).
# ---------------------------------------------------------------------------
_META = None


def _fallback_meta():
    """Minimal card/attack database for our 25 deck cards if cg is unavailable."""
    def C(cid, name, ct, et, hp, basic, s1, s2, ex, retreat, weak, resist, efrom, attacks):
        return {"cardId": cid, "name": name, "cardType": ct, "energyType": et, "hp": hp,
                "basic": basic, "stage1": s1, "stage2": s2, "ex": ex, "megaEx": False,
                "tera": cid == DRAGAPULT, "aceSpec": cid == UNFAIR_STAMP, "retreatCost": retreat,
                "weakness": weak, "resistance": resist, "evolvesFrom": efrom,
                "attacks": attacks, "skills": []}
    cards = {
        DREEPY: C(DREEPY, "Dreepy", CT_POKEMON, EN_DRAGON, 70, True, False, False, False, 1, None, None, None, [150, 151]),
        DRAKLOAK: C(DRAKLOAK, "Drakloak", CT_POKEMON, EN_DRAGON, 90, False, True, False, False, 1, None, None, "Dreepy", [152]),
        DRAGAPULT: C(DRAGAPULT, "Dragapult ex", CT_POKEMON, EN_DRAGON, 320, False, False, True, True, 1, None, None, "Drakloak", [153, 154]),
        DUNSPARCE: C(DUNSPARCE, "Dunsparce", CT_POKEMON, EN_COLORLESS, 60, True, False, False, False, 0, EN_FIGHTING, None, None, [74, 75]),
        DUDUNSPARCE: C(DUDUNSPARCE, "Dudunsparce", CT_POKEMON, EN_COLORLESS, 140, False, True, False, False, 3, EN_FIGHTING, None, "Dunsparce", [76]),
        DUDUNSPARCE_EX: C(DUDUNSPARCE_EX, "Dudunsparce ex", CT_POKEMON, EN_COLORLESS, 270, False, True, False, True, 3, EN_FIGHTING, None, "Dunsparce", [425, 426]),
        MUNKIDORI: C(MUNKIDORI, "Munkidori", CT_POKEMON, EN_PSYCHIC, 110, True, False, False, False, 1, EN_DARKNESS, EN_FIGHTING, None, [141]),
        FEZANDIPIRI_EX: C(FEZANDIPIRI_EX, "Fezandipiti ex", CT_POKEMON, EN_DARKNESS, 210, True, False, False, True, 1, EN_FIGHTING, None, None, [183]),
        MEOWTH_EX: C(MEOWTH_EX, "Meowth ex", CT_POKEMON, EN_COLORLESS, 170, True, False, False, True, 1, EN_FIGHTING, None, None, [1546]),
        BUDEW: C(BUDEW, "Budew", CT_POKEMON, EN_GRASS, 30, True, False, False, False, 0, EN_FIRE, None, None, [323]),
        FIRE_ENERGY: C(FIRE_ENERGY, "Basic {R} Energy", CT_BASIC_ENERGY, EN_FIRE, 0, False, False, False, False, 0, None, None, None, []),
        PSYCHIC_ENERGY: C(PSYCHIC_ENERGY, "Basic {P} Energy", CT_BASIC_ENERGY, EN_PSYCHIC, 0, False, False, False, False, 0, None, None, None, []),
        DARKNESS_ENERGY: C(DARKNESS_ENERGY, "Basic {D} Energy", CT_BASIC_ENERGY, EN_DARKNESS, 0, False, False, False, False, 0, None, None, None, []),
    }
    for cid, name in [(POFFIN, "Buddy-Buddy Poffin"), (NIGHT_STRETCHER, "Night Stretcher"),
                      (CRUSHING_HAMMER, "Crushing Hammer"), (ULTRA_BALL, "Ultra Ball"),
                      (POKE_PAD, "Poke Pad"), (BOSS_ORDERS, "Boss's Orders"), (CRISPIN, "Crispin"),
                      (JUDGE, "Judge"), (LILLIE, "Lillie's Determination"), (UNFAIR_STAMP, "Unfair Stamp"),
                      (ROSA, "Rosa's Encouragement"), (RISKY_RUINS, "Risky Ruins")]:
        ct = CT_SUPPORTER if cid in (BOSS_ORDERS, CRISPIN, JUDGE, LILLIE, ROSA) else (CT_STADIUM if cid == RISKY_RUINS else CT_ITEM)
        cards[cid] = C(cid, name, ct, EN_COLORLESS, 0, False, False, False, False, 0, None, None, None, [])

    def A(aid, name, dmg, energies, text=""):
        return {"attackId": aid, "name": name, "damage": dmg, "energies": energies, "text": text}
    attacks = {
        150: A(150, "Petty Grudge", 10, [EN_PSYCHIC]),
        151: A(151, "Bite", 40, [EN_FIRE, EN_PSYCHIC]),
        152: A(152, "Dragon Headbutt", 70, [EN_FIRE, EN_PSYCHIC]),
        ATK_JET_HEADBUTT: A(ATK_JET_HEADBUTT, "Jet Headbutt", 70, [EN_COLORLESS]),
        ATK_PHANTOM_DIVE: A(ATK_PHANTOM_DIVE, "Phantom Dive", 200, [EN_FIRE, EN_PSYCHIC], "Put 6 damage counters on your opponent's Benched Pokémon in any way you like."),
        74: A(74, "Gnaw", 10, [EN_COLORLESS]),
        75: A(75, "Dig", 30, [EN_COLORLESS, EN_COLORLESS]),
        76: A(76, "Land Crush", 90, [EN_COLORLESS, EN_COLORLESS, EN_COLORLESS]),
        425: A(425, "Tenacious Tail", 0, [EN_COLORLESS], "60 damage for each of your opponent's Pokémon ex in play."),
        426: A(426, "Destructive Drill", 150, [EN_COLORLESS, EN_COLORLESS, EN_COLORLESS]),
        ATK_MIND_BEND: A(ATK_MIND_BEND, "Mind Bend", 60, [EN_PSYCHIC, EN_COLORLESS], "Your opponent's Active Pokémon is now Confused."),
        ATK_CRUEL_ARROW: A(ATK_CRUEL_ARROW, "Cruel Arrow", 0, [EN_COLORLESS, EN_COLORLESS, EN_COLORLESS], "100 damage to 1 of your opponent's Pokémon."),
        1546: A(1546, "Tuck Tail", 60, [EN_COLORLESS, EN_COLORLESS, EN_COLORLESS]),
        ATK_ITCHY_POLLEN: A(ATK_ITCHY_POLLEN, "Itchy Pollen", 10, [], "During your opponent's next turn, they can't play any Item cards from their hand."),
    }
    return cards, attacks


def _meta():
    """Return (cards_by_id, attacks_by_id) as plain dicts, cached."""
    global _META
    if _META is not None:
        return _META
    try:
        from cg import api
        cards = {}
        for c in api.all_card_data():
            cards[c.cardId] = {
                "cardId": c.cardId, "name": c.name, "cardType": int(c.cardType),
                "energyType": int(c.energyType), "hp": c.hp, "basic": c.basic,
                "stage1": c.stage1, "stage2": c.stage2, "ex": c.ex, "megaEx": c.megaEx,
                "tera": c.tera, "aceSpec": c.aceSpec, "retreatCost": c.retreatCost,
                "weakness": None if c.weakness is None else int(c.weakness),
                "resistance": None if c.resistance is None else int(c.resistance),
                "evolvesFrom": c.evolvesFrom, "attacks": list(c.attacks),
                "skills": [(s.name, s.text) for s in c.skills],
            }
        attacks = {}
        for a in api.all_attack():
            attacks[a.attackId] = {"attackId": a.attackId, "name": a.name, "damage": a.damage,
                                   "energies": [int(e) for e in a.energies], "text": a.text}
        _META = (cards, attacks)
    except Exception:
        _META = _fallback_meta()
    return _META


def card(cid):
    return _meta()[0].get(cid)


def attack(aid):
    return _meta()[1].get(aid)


# ---------------------------------------------------------------------------
# Deck loading (unchanged deck.csv).
# ---------------------------------------------------------------------------
_DECK = None


def read_deck_csv():
    global _DECK
    if _DECK is not None:
        return list(_DECK)
    candidates = (
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv"),
        "deck.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pokemon-agent", "deck.csv"),
        "/kaggle_simulations/agent/deck.csv",
    )
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                deck = [int(line.strip()) for line in f if line.strip()]
            if len(deck) == 60:
                _DECK = deck
                return list(deck)
        except (OSError, ValueError):
            continue
    raise ValueError("deck.csv must contain exactly 60 card IDs")


# ---------------------------------------------------------------------------
# Observation helpers (defensive dict access).
# ---------------------------------------------------------------------------
def _unwrap(observation):
    if isinstance(observation, dict):
        inner = observation.get("observation")
        if isinstance(inner, dict):
            observation = inner
        inner = observation.get("obs")
        if isinstance(inner, dict):
            observation = inner
    return observation


def _select(obs):
    s = obs.get("select")
    return s if isinstance(s, dict) else None


def _current(obs):
    c = obs.get("current")
    return c if isinstance(c, dict) else None


def _players(cur):
    p = cur.get("players") or []
    return p if len(p) >= 2 else None


def _yi(cur):
    try:
        return int(cur.get("yourIndex", 0))
    except (TypeError, ValueError):
        return 0


def _active(player):
    a = player.get("active") or []
    if a and isinstance(a[0], dict):
        return a[0]
    return None


def _bench(player):
    return [b for b in (player.get("bench") or []) if isinstance(b, dict)]


def _hand(player):
    return [c for c in (player.get("hand") or []) if isinstance(c, dict)]


def _prize_count(player):
    return len(player.get("prize") or [])


def _pokemon_at(cur, player_index, area, index):
    """Resolve a Pokemon dict from an option's area/index, or None."""
    ps = _players(cur)
    if ps is None or player_index not in (0, 1):
        return None
    p = ps[player_index]
    if area == AR_ACTIVE:
        return _active(p)
    if area == AR_BENCH:
        b = _bench(p)
        if isinstance(index, int) and 0 <= index < len(b):
            return b[index]
    return None


def _option_card_id(obs, opt):
    """Best-effort card ID an option refers to (hand/play/discard/deck)."""
    cur = _current(obs)
    if cur is None:
        return None
    area = opt.get("area")
    index = opt.get("index")
    # MAIN hand-action options (PLAY/EVOLVE/ATTACH) reference the hand by `index`;
    # PLAY omits the `area` field entirely, so infer it here.
    if area is None and opt.get("type") in (OT_PLAY, OT_EVOLVE, OT_ATTACH):
        area = AR_HAND
    pi = opt.get("playerIndex", _yi(cur))
    ps = _players(cur)
    if ps is None or pi not in (0, 1):
        return None
    p = ps[pi]
    if area == AR_HAND:
        h = _hand(p)
        if isinstance(index, int) and 0 <= index < len(h):
            return h[index].get("id")
    elif area == AR_DISCARD:
        d = p.get("discard") or []
        if isinstance(index, int) and 0 <= index < len(d):
            return d[index].get("id")
    elif area == AR_DECK:
        sel = _select(obs)
        deck = (sel or {}).get("deck")
        if deck and isinstance(index, int) and 0 <= index < len(deck):
            return deck[index].get("id")
    elif area in (AR_ACTIVE, AR_BENCH):
        pk = _pokemon_at(cur, pi, area, index)
        if pk:
            return pk.get("id")
    return None


def _energies(pokemon):
    return [int(e) for e in (pokemon.get("energies") or [])]


def _hp(pokemon):
    try:
        return int(pokemon.get("hp", 0))
    except (TypeError, ValueError):
        return 0


def _maxhp(pokemon):
    try:
        return int(pokemon.get("maxHp", 0))
    except (TypeError, ValueError):
        return 0


def _damage(pokemon):
    return max(0, _maxhp(pokemon) - _hp(pokemon))


def _prizes_for(pokemon):
    c = card(pokemon.get("id"))
    if not c:
        return 1
    if c.get("megaEx"):
        return 3
    if c.get("ex"):
        return 2
    return 1


def _in_play_pokemon(player):
    out = []
    a = _active(player)
    if a:
        out.append(a)
    out.extend(_bench(player))
    return out


# ---------------------------------------------------------------------------
# Energy / attack planning helpers.
# ---------------------------------------------------------------------------
def _energy_matches(cost, available):
    """Can `available` (list of energy types) pay `cost` (list w/ COLORLESS wild)?

    Returns (can_pay, leftover_available). Greedy: match specific types first.
    """
    avail = list(available)
    specific = [e for e in cost if e != EN_COLORLESS]
    wild = cost.count(EN_COLORLESS)
    for e in specific:
        if e in avail:
            avail.remove(e)
        elif EN_RAINBOW in avail:
            avail.remove(EN_RAINBOW)
        else:
            return False, avail
    # colorless can be paid by any remaining energy
    if len(avail) < wild:
        return False, avail
    return True, avail


def _usable_attacks(pokemon):
    """Attack dicts usable by this Pokemon given its attached energies."""
    c = card(pokemon.get("id"))
    if not c:
        return []
    en = _energies(pokemon)
    out = []
    for aid in c.get("attacks", []):
        a = attack(aid)
        if a and _energy_matches(a["energies"], en)[0]:
            out.append(a)
    return out


def _best_attack(pokemon):
    us = _usable_attacks(pokemon)
    if not us:
        return None
    return max(us, key=lambda a: a["damage"])


def _energy_needed(pokemon):
    """Energy types still needed to enable this Pokemon's best (highest dmg) attack.

    Returns a list of needed types (COLORLESS means 'any'). Empty if already able.
    """
    c = card(pokemon.get("id"))
    if not c:
        return []
    aids = c.get("attacks", [])
    if not aids:
        return []
    # target the highest-damage attack
    target = None
    for aid in aids:
        a = attack(aid)
        if a and (target is None or a["damage"] > target["damage"]):
            target = a
    if target is None:
        return []
    en = _energies(pokemon)
    cost = list(target["energies"])
    # remove what we already have (specific first)
    for e in [x for x in cost if x != EN_COLORLESS]:
        if e in en:
            en.remove(e)
            cost.remove(e)
    # remaining cost (specific + colorless) is what we still need
    return cost


def _expected_damage(attacker, defender, atk):
    """Damage of atk from attacker to defender, applying weakness/resistance."""
    base = atk["damage"]
    ac = card(attacker.get("id"))
    dc = card(defender.get("id"))
    if ac and dc:
        et = ac.get("energyType")
        if dc.get("weakness") is not None and dc["weakness"] == et:
            base *= 2
        if dc.get("resistance") is not None and dc["resistance"] == et:
            base = max(0, base - 30)
    return base


def _opp_threat(cur, yi):
    """Estimated max damage opponent's active can deal to our active next turn."""
    ps = _players(cur)
    if ps is None:
        return 0
    opp_active = _active(ps[1 - yi])
    own_active = _active(ps[yi])
    if not opp_active or not own_active:
        return 0
    best = 0
    for a in _usable_attacks(opp_active):
        best = max(best, _expected_damage(opp_active, own_active, a))
    return best


# ---------------------------------------------------------------------------
# Hand / card valuation (for discard, draw timing, search targets).
# ---------------------------------------------------------------------------
def _board_ids(player):
    ids = []
    for pk in _in_play_pokemon(player):
        ids.append(pk.get("id"))
    return ids


def _hand_quality(obs):
    """Rough 0..1 measure of how good our current hand is (higher = keep it)."""
    cur = _current(obs)
    if cur is None:
        return 0.5
    ps = _players(cur)
    yi = _yi(cur)
    own = ps[yi]
    hand = _hand(own)
    if not hand:
        return 0.0
    board = _board_ids(own)
    score = 0.0
    for c in hand:
        score += _keep_value(c.get("id"), board, obs)
    return max(0.0, min(1.0, score / (len(hand) * 60.0)))


def _keep_value(cid, board, obs):
    """How much we want to KEEP this card in hand (0..~100). Used for discard."""
    c = card(cid)
    if not c:
        return 20
    ct = c.get("cardType")
    cur = _current(obs)
    ps = _players(cur) if cur else None
    yi = _yi(cur) if cur else 0
    own = ps[yi] if ps else None
    opp = ps[1 - yi] if ps else None

    if ct == CT_POKEMON:
        if cid == DRAGAPULT:
            # keep if we can evolve into it soon
            if DRAKLOAK in board or DRAKLOAK in [x.get("id") for x in _hand(own)]:
                return 95
            return 70
        if cid == DRAKLOAK:
            return 85 if DREEPY in board else 55
        if cid == DREEPY:
            n = board.count(DREEPY) + board.count(DRAKLOAK) + board.count(DRAGAPULT)
            return 70 if n == 0 else (45 if n == 1 else 25)
        if cid in (DUDUNSPARCE, DUDUNSPARCE_EX):
            return 60 if DUNSPARCE in board else 35
        if cid == DUNSPARCE:
            return 40 if (DUDUNSPARCE in board or DUDUNSPARCE_EX in board) else 30
        if cid == MUNKIDORI:
            return 45 if MUNKIDORI not in board else 20
        if cid in (FEZANDIPIRI_EX, MEOWTH_EX):
            return 50 if cid not in board else 20
        if cid == BUDEW:
            turn = cur.get("turn", 0) if cur else 0
            return 40 if turn <= 4 else 15
        return 30
    if ct in (CT_BASIC_ENERGY, CT_SPECIAL_ENERGY):
        et = c.get("energyType")
        # premium energies for the Dragapult line
        if et in (EN_FIRE, EN_PSYCHIC):
            return 55
        if et == EN_DARKNESS:
            return 40 if MUNKIDORI in board else 20
        return 25
    if ct == CT_SUPPORTER:
        if cid == LILLIE:
            return 45
        if cid == BOSS_ORDERS:
            return 60
        if cid == CRISPIN:
            return 50
        if cid == JUDGE:
            return 30
        if cid == ROSA:
            return 45 if _rosa_usable(obs) else 10
        return 30
    if ct == CT_ITEM:
        if cid == ULTRA_BALL:
            return 55
        if cid == POKE_PAD:
            return 50
        if cid == POFFIN:
            return 50
        if cid == NIGHT_STRETCHER:
            return 45 if _night_stretcher_target(obs) else 15
        if cid == CRUSHING_HAMMER:
            return 35 if _opp_has_energy(obs) else 15
        if cid == UNFAIR_STAMP:
            return 45
        return 30
    if ct == CT_STADIUM:
        return 25
    return 20


def _ko_recent(obs):
    """Best-effort: did one of our Pokemon just move to the discard (a KO)?

    Reads only the current observation's logs. The engine gates the legality of
    KO-triggered cards (Unfair Stamp / Flip the Script), so this is advisory only.
    """
    cur = _current(obs)
    if cur is None:
        return False
    yi = _yi(cur)
    for log in (obs.get("logs") or []):
        if not isinstance(log, dict):
            continue
        if log.get("type") == 6 and log.get("playerIndex") == yi:
            if log.get("fromArea") in (AR_ACTIVE, AR_BENCH) and log.get("toArea") == AR_DISCARD:
                c = card(log.get("cardId"))
                if c and c.get("cardType") == CT_POKEMON:
                    return True
    return False


def _rosa_usable(obs):
    cur = _current(obs)
    if cur is None:
        return False
    ps = _players(cur)
    yi = _yi(cur)
    own, opp = ps[yi], ps[1 - yi]
    if _prize_count(own) <= _prize_count(opp):
        return False
    if DRAGAPULT not in _board_ids(own):
        return False
    discard = own.get("discard") or []
    return any(card(x.get("id")) and card(x.get("id")).get("cardType") == CT_BASIC_ENERGY for x in discard)


def _night_stretcher_target(obs):
    cur = _current(obs)
    if cur is None:
        return None
    ps = _players(cur)
    own = ps[_yi(cur)]
    best = None
    for x in (own.get("discard") or []):
        cid = x.get("id")
        c = card(cid)
        if not c:
            continue
        if c.get("cardType") == CT_POKEMON and cid in DRAGAPULT_LINE:
            return cid
        if c.get("cardType") == CT_BASIC_ENERGY and c.get("energyType") in (EN_FIRE, EN_PSYCHIC):
            best = best or cid
    return best


def _opp_has_energy(obs):
    cur = _current(obs)
    if cur is None:
        return False
    ps = _players(cur)
    opp = ps[1 - _yi(cur)]
    return any(_energies(pk) for pk in _in_play_pokemon(opp))


# ---------------------------------------------------------------------------
# MAIN option scoring.
# ---------------------------------------------------------------------------
def _score_attack(obs, opt):
    cur = _current(obs)
    ps = _players(cur)
    yi = _yi(cur)
    own, opp = ps[yi], ps[1 - yi]
    attacker = _active(own)
    defender = _active(opp)
    a = attack(opt.get("attackId"))
    if not a or not attacker:
        return 1.0
    aid = a["attackId"]
    # Dynamic-damage attacks whose base damage is 0 in the metadata.
    if aid == ATK_TENACIOUS_TAIL:
        ex_n = sum(1 for pk in _in_play_pokemon(opp) if _prizes_for(pk) >= 2)
        base = 60 * ex_n
        dmg = base  # colourless attacker; apply weakness/resistance vs active
        if defender:
            dmg = _expected_damage(attacker, defender, {"damage": base})
    elif aid == ATK_CRUEL_ARROW:
        dmg = 100  # snipe any opponent Pokemon (target chosen in a later select)
    else:
        dmg = _expected_damage(attacker, defender, a) if defender else a["damage"]
    score = dmg * 0.5
    if defender and dmg >= _hp(defender) and _hp(defender) > 0:
        prizes = _prizes_for(defender)
        score += 300 + 250 * prizes
        if _prize_count(opp) <= prizes:
            score += 10000  # game-winning KO
    # Phantom Dive bench spread value
    if aid == ATK_PHANTOM_DIVE:
        score += _spread_value(obs, 6)
    # Itchy Pollen item-lock
    if aid == ATK_ITCHY_POLLEN:
        score += 80
    # Mind Bend confusion
    if aid == ATK_MIND_BEND:
        score += 30
    # Cruel Arrow snipe bonus (can hit a damaged Bench target for a KO)
    if aid == ATK_CRUEL_ARROW:
        score += 60 + _snipe_value(obs, 100)
    return score


def _snipe_value(obs, dmg):
    """Best knockout value from hitting any opponent Pokemon with `dmg` (Cruel Arrow)."""
    cur = _current(obs)
    ps = _players(cur)
    opp = ps[1 - _yi(cur)]
    best = 0
    for pk in _in_play_pokemon(opp):
        if 0 < _hp(pk) <= dmg:
            best = max(best, 250 * _prizes_for(pk))
    return best


def _spread_value(obs, counters):
    """Value of placing `counters` damage counters on opponent bench (Phantom Dive)."""
    cur = _current(obs)
    ps = _players(cur)
    opp = ps[1 - _yi(cur)]
    bench = _bench(opp)
    if not bench:
        return 0
    # estimate KOs achievable
    hp_list = sorted((_hp(b), _prizes_for(b)) for b in bench)
    remaining = counters
    val = 0
    for hp, prizes in hp_list:
        need = (hp + 9) // 10
        if need <= remaining and hp > 0:
            val += 250 * prizes
            remaining -= need
    val += min(counters, sum(max(0, _hp(b)) for b in bench)) * 0.3
    return val


def _score_evolve(obs, opt):
    cur = _current(obs)
    ps = _players(cur)
    yi = _yi(cur)
    own = ps[yi]
    evo_id = _option_card_id(obs, opt)
    target = _pokemon_at(cur, opt.get("playerIndex", yi), opt.get("inPlayArea"), opt.get("inPlayIndex"))
    base = 150
    if evo_id == DRAGAPULT:
        base = 320
    elif evo_id == DRAKLOAK:
        base = 230
    elif evo_id == DUDUNSPARCE_EX:
        base = 200
    elif evo_id == DUDUNSPARCE:
        base = 150
    # evolving the active is slightly more urgent if it's under threat
    if target is not None and opt.get("inPlayArea") == AR_ACTIVE:
        base += 10
    return base


def _score_ability(obs, opt):
    cur = _current(obs)
    ps = _players(cur)
    yi = _yi(cur)
    own = ps[yi]
    cid = _option_card_id(obs, opt)
    pk = _pokemon_at(cur, opt.get("playerIndex", yi), opt.get("area"), opt.get("index"))
    hand_n = len(_hand(own))
    if cid == DRAKLOAK:      # Recon Directive: look top 2, keep 1 -> always good
        return 130
    if cid == DUDUNSPARCE:   # Run Away Draw: draw 3, shuffle self back
        # good when hand is low and we don't lose invested energy; avoid deck-out
        deck_n = own.get("deckCount", 0)
        if deck_n < 4:
            return -20
        en = len(_energies(pk)) if pk else 0
        if hand_n <= 3 and en == 0:
            return 160
        if hand_n <= 2:
            return 120
        return 20
    if cid == MUNKIDORI:     # Adrena-Brain: move damage counters
        if _damage_move_value(obs) > 0:
            return 150
        return 10
    if cid == FEZANDIPIRI_EX:  # Flip the Script: draw 3 (only legal after KO)
        return 140
    if cid == MEOWTH_EX:     # Last-Ditch Catch: search Supporter
        return 120
    return 60


def _damage_move_value(obs):
    """Value of Munkidori moving damage from our Pokemon to opponent's."""
    cur = _current(obs)
    ps = _players(cur)
    yi = _yi(cur)
    own, opp = ps[yi], ps[1 - yi]
    damaged = [pk for pk in _in_play_pokemon(own) if _damage(pk) > 0]
    if not damaged:
        return 0
    targets = _in_play_pokemon(opp)
    if not targets:
        return 0
    return 1


def _score_attach(obs, opt):
    cur = _current(obs)
    ps = _players(cur)
    yi = _yi(cur)
    own = ps[yi]
    energy_id = _option_card_id(obs, opt)
    target = _pokemon_at(cur, opt.get("playerIndex", yi), opt.get("inPlayArea"), opt.get("inPlayIndex"))
    if not target:
        return 5
    ec = card(energy_id)
    et = ec.get("energyType") if ec else None
    return _attach_value(et, target, own, obs)


def _attach_value(et, target, own, obs):
    """How good is attaching energy type `et` to `target`."""
    tid = target.get("id")
    needed = _energy_needed(target)
    # does this energy progress the target toward its best attack?
    helps = False
    if et in needed:
        helps = True
    elif EN_COLORLESS in needed and et is not None:
        helps = True
    # priority of the target as an attacker
    if tid == DRAGAPULT:
        prio = 3.0
    elif tid in (DRAKLOAK, DREEPY):
        prio = 2.2
    elif tid in (DUDUNSPARCE_EX, FEZANDIPIRI_EX, MEOWTH_EX):
        prio = 1.6
    elif tid == MUNKIDORI:
        prio = 1.4 if et == EN_DARKNESS else 0.8
    elif tid == DUDUNSPARCE:
        prio = 1.0
    else:
        prio = 0.7
    # Darkness energy is only useful on Munkidori
    if et == EN_DARKNESS and tid != MUNKIDORI:
        return 5
    if not helps:
        # over-attaching / wrong type: low value
        return 8 * prio * 0.3
    # active attacker that will attack soon is best
    cur = _current(obs)
    is_active = False
    if cur:
        ps = _players(cur)
        a = _active(ps[_yi(cur)])
        if a and a.get("serial") == target.get("serial"):
            is_active = True
    base = 90 * prio
    if is_active:
        base *= 1.25
    # if this attachment completes the best attack, big bonus
    en_after = _energies(target) + ([et] if et is not None else [])
    c = card(tid)
    if c:
        for aid in c.get("attacks", []):
            a = attack(aid)
            if a and _energy_matches(a["energies"], en_after)[0] and not _energy_matches(a["energies"], _energies(target))[0]:
                base += 60
                break
    return base


def _score_play(obs, opt):
    cur = _current(obs)
    ps = _players(cur)
    yi = _yi(cur)
    own, opp = ps[yi], ps[1 - yi]
    cid = _option_card_id(obs, opt)
    c = card(cid)
    if not c:
        return 40
    ct = c.get("cardType")
    hand = _hand(own)
    hand_n = len(hand)
    board = _board_ids(own)
    bench_n = len(_bench(own))
    bench_max = own.get("benchMax", 5)

    if ct == CT_POKEMON:
        # playing a basic to bench (or it could be evolve handled separately)
        if c.get("basic"):
            if bench_n >= bench_max:
                return -20
            # DONK PROTECTION: a second Pokemon in play is critical for survival
            survival = 150 if bench_n == 0 else 0
            if cid in DRAGAPULT_LINE:
                return (180 if cid == DREEPY else 120) + survival
            if cid == BUDEW:
                return (90 if cur.get("turn", 0) <= 4 else 30) + survival
            if cid == MUNKIDORI:
                return (100 if MUNKIDORI not in board else 30) + survival
            if cid in (FEZANDIPIRI_EX, MEOWTH_EX):
                return (80 if cid not in board else 20) + survival
            if cid == DUNSPARCE:
                return 70 + survival
            return 60 + survival
        return 40

    if ct == CT_ITEM:
        if cid == POFFIN:
            if bench_n >= bench_max:
                return -10
            return 230 if bench_n == 0 else 170
        if cid == ULTRA_BALL:
            # need 2 other cards to discard; worth it to find evolution piece
            return 160 if hand_n >= 3 else 40
        if cid == POKE_PAD:
            return 140
        if cid == NIGHT_STRETCHER:
            return 150 if _night_stretcher_target(obs) else 20
        if cid == CRUSHING_HAMMER:
            return 90 if _opp_has_energy(obs) else 20
        if cid == UNFAIR_STAMP:
            # Only offered by the engine when legal (right after our KO), so it
            # is always a strong tempo play when it appears as an option.
            return 180
        return 60

    if ct == CT_SUPPORTER:
        if cur.get("supporterPlayed"):
            return -50
        if cid == LILLIE:
            hq = _hand_quality(obs)
            # shuffle away a bad hand; keep a good one
            return 200 * (1.0 - hq) - 40
        if cid == JUDGE:
            opp_hand = opp.get("handCount", 0)
            hq = _hand_quality(obs)
            return 120 * (1.0 - hq) + (30 if opp_hand >= 6 else 0) - 30
        if cid == CRISPIN:
            return 170 if _needs_energy(obs) else 60
        if cid == BOSS_ORDERS:
            return 190 if _boss_target(obs) else 70
        if cid == ROSA:
            return 200 if _rosa_usable(obs) else -30
        return 60

    if ct == CT_STADIUM:
        if cid == RISKY_RUINS:
            if cur.get("stadiumPlayed"):
                return -40
            return _stadium_value(obs)
        return 40

    if ct in (CT_BASIC_ENERGY, CT_SPECIAL_ENERGY):
        # playing energy from hand as a "play" (rare; usually attach)
        return 30
    return 40


def _needs_energy(obs):
    cur = _current(obs)
    ps = _players(cur)
    own = ps[_yi(cur)]
    for pk in _in_play_pokemon(own):
        if pk.get("id") in DRAGAPULT_LINE and _energy_needed(pk):
            return True
    return False


def _boss_target(obs):
    """Is there a good Boss's Orders gust target (KO or high threat)?"""
    cur = _current(obs)
    ps = _players(cur)
    yi = _yi(cur)
    own, opp = ps[yi], ps[1 - yi]
    attacker = _active(own)
    if not attacker:
        return None
    best = None
    for b in _bench(opp):
        # can our active KO it?
        ba = _best_attack(attacker)
        if ba:
            dmg = _expected_damage(attacker, b, ba)
            if dmg >= _hp(b) and _hp(b) > 0:
                return b
        # high threat / low hp
        if best is None or _hp(b) < _hp(best):
            best = b
    return best


def _stadium_value(obs):
    cur = _current(obs)
    ps = _players(cur)
    yi = _yi(cur)
    own, opp = ps[yi], ps[1 - yi]
    # Risky Ruins damages non-Dark basics put on bench. Good if opponent relies
    # on basics more than we do; mild self-harm to our Dreepy/Budew.
    opp_basics = sum(1 for pk in _in_play_pokemon(opp) if card(pk.get("id")) and card(pk.get("id")).get("basic"))
    own_basics = sum(1 for pk in _in_play_pokemon(own) if card(pk.get("id")) and card(pk.get("id")).get("basic"))
    if opp_basics > own_basics:
        return 90
    if opp_basics >= 2:
        return 60
    return 30


def _score_retreat(obs):
    cur = _current(obs)
    ps = _players(cur)
    yi = _yi(cur)
    own, opp = ps[yi], ps[1 - yi]
    active = _active(own)
    bench = _bench(own)
    if not active or not bench:
        return -50
    threat = _opp_threat(cur, yi)
    # will active die?
    dies = threat >= _hp(active) and _hp(active) > 0
    # is there a ready attacker on bench?
    ready = [b for b in bench if _usable_attacks(b)]
    active_can_attack = bool(_usable_attacks(active))
    retreat_cost = card(active.get("id")).get("retreatCost", 0) if card(active.get("id")) else 0
    invested = len(_energies(active))
    val = 0
    if dies and (ready or len(bench) > 0):
        val += 220  # save the Pokemon / deny prize
    if not active_can_attack and ready:
        val += 120
    if retreat_cost == 0 and ready and not active_can_attack:
        val += 60
    # cost: lose invested energy (goes to discard) and a prize-risk if none
    val -= invested * 25
    if not ready and not dies:
        val -= 80
    return val


def _score_discard_inplay(obs, opt):
    cid = _option_card_id(obs, opt)
    # discarding a card in play is usually costly; prefer least valuable
    return -50 - _keep_value(cid, [], obs) * 0.5


def score_main_option(obs, opt):
    t = opt.get("type")
    if t == OT_ATTACK:
        return _score_attack(obs, opt)
    if t == OT_EVOLVE:
        return _score_evolve(obs, opt)
    if t == OT_ABILITY:
        return _score_ability(obs, opt)
    if t == OT_ATTACH:
        return _score_attach(obs, opt)
    if t == OT_PLAY:
        return _score_play(obs, opt)
    if t == OT_RETREAT:
        return _score_retreat(obs)
    if t == OT_DISCARD:
        return _score_discard_inplay(obs, opt)
    if t == OT_END:
        return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Non-MAIN selection handlers.
# ---------------------------------------------------------------------------
def _choose_cards(obs, score_fn, min_count, max_count, options, maximize=True):
    scored = [(score_fn(i, o), i) for i, o in enumerate(options)]
    scored.sort(reverse=maximize)
    n = max(min_count, 0)
    n = min(n, max_count, len(options))
    if n <= 0:
        # optional selection: take only positive-value picks up to max
        pos = [i for s, i in scored if s > 0]
        return pos[:max_count]
    return [i for _, i in scored[:n]]


def _pokemon_option_value(obs, opt, mode):
    """Value of a Pokemon-target option for damage/heal/target contexts."""
    cur = _current(obs)
    pi = opt.get("playerIndex", _yi(cur))
    pk = _pokemon_at(cur, pi, opt.get("area"), opt.get("index"))
    if pk is None:
        return 0
    yi = _yi(cur)
    if mode == "damage":       # put damage on opponent -> prefer KO / low hp / high value
        hp = _hp(pk)
        prizes = _prizes_for(pk)
        if hp <= 0:
            return -1000
        # prizes per counter needed
        need = max(1, (hp + 9) // 10)
        return prizes * 1000.0 / need + (100 if pi != yi else -100)
    if mode == "heal":         # remove damage from our Pokemon -> most damaged, valuable
        return _damage(pk) + _prizes_for(pk) * 5
    if mode == "target":       # generic effect target (opponent) ->威胁 / low hp
        return -_hp(pk) + _prizes_for(pk) * 20
    return 0


def handle_selection(obs):
    """Return a list of option indices for the current selection, or None to
    signal 'use MAIN scoring'. Raises nothing; callers guard."""
    sel = _select(obs)
    cur = _current(obs)
    if sel is None or cur is None:
        return None
    options = sel.get("option") or []
    if not options:
        return []
    min_count = int(sel.get("minCount", 1) or 0)
    max_count = int(sel.get("maxCount", 1) or 0)
    if max_count <= 0:
        return []
    max_count = min(max_count, len(options))
    min_count = min(min_count, max_count)
    ctx = int(sel.get("context", 0) or 0)
    stype = int(sel.get("type", 0) or 0)
    yi = _yi(cur)
    ps = _players(cur)

    # --- Yes/No contexts ---
    if stype == 9:
        yes = next((i for i, o in enumerate(options) if o.get("type") == OT_YES), None)
        no = next((i for i, o in enumerate(options) if o.get("type") == OT_NO), None)
        if ctx == CX_IS_FIRST:
            want_no = PREFER_GO_SECOND
            if want_no and no is not None:
                return [no]
            if not want_no and yes is not None:
                return [yes]
            return [0]
        if ctx == CX_MULLIGAN:
            return [yes if yes is not None else 0]
        if ctx == CX_ACTIVATE:
            # optional effect activation; almost always beneficial for our cards
            cc = sel.get("contextCard") or {}
            cid = cc.get("id")
            if cid == DUDUNSPARCE:
                # Run Away Draw as an activate: decide by hand size
                own = ps[yi]
                if len(_hand(own)) <= 3:
                    return [yes if yes is not None else 0]
                return [no if no is not None else 0]
            return [yes if yes is not None else 0]
        if ctx == CX_MORE_DEVOLVE:
            return [no if no is not None else 0]
        if ctx in (CX_FIRST_EFFECT, CX_COIN_HEAD):
            return [yes if yes is not None else 0]
        return [yes if yes is not None else 0]

    # --- Number contexts ---
    if stype == 8:
        nums = [(o.get("number", 0), i) for i, o in enumerate(options)]
        if ctx == CX_DRAW_COUNT:
            # draw as many as safe (avoid deck-out)
            own = ps[yi]
            deck_n = own.get("deckCount", 0)
            safe = [i for n, i in nums if n <= max(0, deck_n - 1)]
            if safe:
                return [max(safe, key=lambda i: options[i].get("number", 0))]
            return [min(nums)[1]]
        # damage counters / remove counters: take max
        return [max(nums)[1]]

    # --- Skill order ---
    if stype == 5 or ctx == CX_SKILL_ORDER:
        n = max(min_count, 1)
        return list(range(min(n, max_count, len(options))))

    # --- Special condition ---
    if stype == 10:
        # recover: pick worst condition; affect: pick most disruptive
        order = {3: 0, 2: 1, 4: 2, 1: 3, 0: 4}  # paralyze>sleep>confuse>burn>poison
        if ctx == CX_RECOVER_SPECIAL_CONDITION:
            return [min(range(len(options)), key=lambda i: order.get(options[i].get("specialConditionType", 0), 9))]
        return [min(range(len(options)), key=lambda i: order.get(options[i].get("specialConditionType", 0), 9))]

    # --- MAIN ---
    if ctx == CX_MAIN and stype == 0:
        return None  # handled by MAIN scoring (+ optional search)

    # --- Attack selection (dedicated) ---
    if ctx == CX_ATTACK or stype == 6:
        best = max(range(len(options)), key=lambda i: _score_attack(obs, options[i]))
        return [best]

    # --- Evolution source/target ---
    if ctx in (CX_EVOLVES_FROM, CX_EVOLVES_TO, CX_EVOLVE) or stype == 7:
        return _choose_cards(obs, lambda i, o: _score_evolve(obs, o), min_count, max_count, options)

    # --- Card selections by context ---
    if ctx == CX_SETUP_ACTIVE:
        return _choose_cards(obs, lambda i, o: _setup_active_value(obs, o), 1, 1, options)
    if ctx == CX_SETUP_BENCH:
        return _choose_cards(obs, lambda i, o: _setup_bench_value(obs, o), min_count, max_count, options)
    if ctx in (CX_SWITCH, CX_TO_ACTIVE):
        return _choose_cards(obs, lambda i, o: _switch_value(obs, o), min_count, max_count, options)
    if ctx in (CX_TO_BENCH, CX_TO_FIELD):
        return _choose_cards(obs, lambda i, o: _to_bench_value(obs, o), min_count, max_count, options)
    if ctx == CX_TO_HAND:
        return _choose_cards(obs, lambda i, o: _to_hand_value(obs, o), min_count, max_count, options)
    if ctx in (CX_DISCARD, CX_DISCARD_CARD_OR_ATTACHED):
        # discard the LEAST valuable cards
        return _choose_cards(obs, lambda i, o: _keep_value(_option_card_id(obs, o), _board_ids(ps[yi]), obs),
                             min_count, max_count, options, maximize=False)
    if ctx in (CX_TO_DECK, CX_TO_DECK_BOTTOM, CX_TO_PRIZE, CX_NOT_MOVE):
        return _choose_cards(obs, lambda i, o: _keep_value(_option_card_id(obs, o), _board_ids(ps[yi]), obs),
                             min_count, max_count, options, maximize=False)
    if ctx in (CX_DAMAGE_COUNTER, CX_DAMAGE_COUNTER_ANY, CX_DAMAGE):
        return _choose_cards(obs, lambda i, o: _pokemon_option_value(obs, o, "damage"), min_count, max_count, options)
    if ctx in (CX_REMOVE_DAMAGE_COUNTER, CX_HEAL):
        return _choose_cards(obs, lambda i, o: _pokemon_option_value(obs, o, "heal"), min_count, max_count, options)
    if ctx == CX_EFFECT_TARGET:
        # could be opponent (damage) or own (buff); infer from playerIndex
        pi = options[0].get("playerIndex", yi)
        mode = "damage" if pi != yi else "heal"
        return _choose_cards(obs, lambda i, o: _pokemon_option_value(obs, o, mode), min_count, max_count, options)
    if ctx == CX_ATTACH_FROM:
        return _choose_cards(obs, lambda i, o: _attach_from_value(obs, o), min_count, max_count, options)
    if ctx == CX_ATTACH_TO:
        return _choose_cards(obs, lambda i, o: _to_hand_value(obs, o), min_count, max_count, options)
    if ctx in (CX_DISCARD_ENERGY, CX_DISCARD_ENERGY_CARD, CX_SWITCH_ENERGY_CARD, CX_SWITCH_ENERGY):
        # discard cheapest / least useful energy
        return _choose_cards(obs, lambda i, o: _energy_discard_value(obs, o), min_count, max_count, options, maximize=False)
    if ctx in (CX_TO_HAND_ENERGY, CX_TO_DECK_ENERGY):
        return _choose_cards(obs, lambda i, o: _energy_discard_value(obs, o), min_count, max_count, options, maximize=False)
    if ctx == CX_DISCARD_TOOL_CARD:
        return [0]
    if ctx == CX_DETACH_FROM:
        return [0]
    if ctx == CX_LOOK:
        return [0]
    if ctx == CX_DEVOLVE:
        return [0]
    if ctx == CX_DISABLE_ATTACK:
        # disable opponent's strongest attack
        return [max(range(len(options)), key=lambda i: (attack(options[i].get("attackId")) or {}).get("damage", 0))]

    # --- Fallback: generic card selection ---
    if stype in (1, 2, 3):
        return _choose_cards(obs, lambda i, o: _keep_value(_option_card_id(obs, o), _board_ids(ps[yi]), obs),
                             min_count, max_count, options)
    # energy selection fallback
    if stype == 4:
        return _choose_cards(obs, lambda i, o: _energy_discard_value(obs, o), min_count, max_count, options, maximize=False)
    # ultimate fallback: first min_count options
    n = max(min_count, 1)
    return list(range(min(n, max_count, len(options))))


def _setup_active_value(obs, opt):
    cid = _option_card_id(obs, opt)
    # prefer Budew (free attack, retreat 0) going second; else Dreepy (evolution),
    # else Dunsparce (retreat 0). Avoid leading with an ex (2 prizes).
    if cid == BUDEW:
        return 100 if PREFER_GO_SECOND else 70
    if cid == DREEPY:
        return 85
    if cid == DUNSPARCE:
        return 75
    if cid == MUNKIDORI:
        return 60
    if cid in (FEZANDIPIRI_EX, MEOWTH_EX):
        return 30
    c = card(cid)
    if c and c.get("basic"):
        return 50
    return 10


def _setup_bench_value(obs, opt):
    cid = _option_card_id(obs, opt)
    if cid == DREEPY:
        return 90
    if cid == DUNSPARCE:
        return 70
    if cid == MUNKIDORI:
        return 65
    if cid == BUDEW:
        return 55
    if cid in (FEZANDIPIRI_EX, MEOWTH_EX):
        return 35
    c = card(cid)
    if c and c.get("basic"):
        return 50
    return 10


def _switch_value(obs, opt):
    cur = _current(obs)
    pi = opt.get("playerIndex", _yi(cur))
    pk = _pokemon_at(cur, pi, opt.get("area"), opt.get("index"))
    if pk is None:
        return 0
    # prefer a ready attacker, else highest HP / lowest retreat
    ready = bool(_usable_attacks(pk))
    return (100 if ready else 0) + _hp(pk) * 0.2 - (card(pk.get("id")) or {}).get("retreatCost", 0) * 5


def _to_bench_value(obs, opt):
    cid = _option_card_id(obs, opt)
    if cid in DRAGAPULT_LINE:
        return 100
    if cid == DUNSPARCE:
        return 70
    if cid == MUNKIDORI:
        return 65
    if cid == BUDEW:
        return 50
    c = card(cid)
    if c and c.get("basic"):
        return 60
    return 20


def _vulnerable(obs):
    """True if we have no Benched Pokemon (a lone Active can be donked for the loss)."""
    cur = _current(obs)
    if cur is None:
        return False
    ps = _players(cur)
    if ps is None:
        return False
    own = ps[_yi(cur)]
    return len(_bench(own)) == 0 and _active(own) is not None


def _to_hand_value(obs, opt):
    cid = _option_card_id(obs, opt)
    c = card(cid)
    # DONK PROTECTION: with an empty bench we must fetch a Basic to survive,
    # prioritising it over evolution pieces.
    if _vulnerable(obs) and c and c.get("cardType") == CT_POKEMON and c.get("basic"):
        if cid == DREEPY:
            return 200
        if cid in (BUDEW, DUNSPARCE, MUNKIDORI):
            return 180
        return 170
    # search/recover: prioritize evolution pieces and premium energy
    if cid == DRAGAPULT:
        return 120
    if cid == DRAKLOAK:
        return 110
    if cid == DREEPY:
        return 100
    if cid in (DUDUNSPARCE_EX, DUDUNSPARCE):
        return 80
    if c and c.get("cardType") == CT_BASIC_ENERGY:
        et = c.get("energyType")
        return 90 if et in (EN_FIRE, EN_PSYCHIC) else 50
    if c and c.get("cardType") == CT_POKEMON:
        return 70
    return _keep_value(cid, [], obs)


def _attach_from_value(obs, opt):
    cur = _current(obs)
    pi = opt.get("playerIndex", _yi(cur))
    pk = _pokemon_at(cur, pi, opt.get("area"), opt.get("index"))
    if pk is None:
        return 0
    cc = (_select(obs) or {}).get("contextCard") or {}
    et = (card(cc.get("id")) or {}).get("energyType")
    own = _players(cur)[_yi(cur)]
    return _attach_value(et, pk, own, obs)


def _energy_discard_value(obs, opt):
    # value of the energy being discarded; lower = prefer to discard
    cur = _current(obs)
    pi = opt.get("playerIndex", _yi(cur))
    pk = _pokemon_at(cur, pi, opt.get("area"), opt.get("index"))
    ei = opt.get("energyIndex")
    if pk and isinstance(ei, int):
        ecs = pk.get("energyCards") or []
        if 0 <= ei < len(ecs):
            cid = ecs[ei].get("id")
            et = (card(cid) or {}).get("energyType")
            # prefer discarding Darkness (least useful) then surplus
            if et == EN_DARKNESS:
                return 0
            if et in (EN_FIRE, EN_PSYCHIC):
                return 10
            return 5
    return 5


# ---------------------------------------------------------------------------
# Monte-Carlo lookahead (verified hidden-state construction, budgeted).
# ---------------------------------------------------------------------------
def _known_ids(player):
    ids = []
    for c in _hand(player):
        ids.append(c.get("id"))
    for c in (player.get("discard") or []):
        ids.append(c.get("id"))
    for pk in _in_play_pokemon(player):
        ids.append(pk.get("id"))
        for c in (pk.get("energyCards") or []):
            ids.append(c.get("id"))
        for c in (pk.get("tools") or []):
            ids.append(c.get("id"))
        for c in (pk.get("preEvolution") or []):
            ids.append(c.get("id"))
    for c in (player.get("prize") or []):
        if isinstance(c, dict):
            ids.append(c.get("id"))
    return [i for i in ids if i is not None]


def _build_hidden(obs, rng):
    """Construct search_begin hidden-state predictions. Returns dict or None."""
    cur = _current(obs)
    if cur is None:
        return None
    ps = _players(cur)
    yi = _yi(cur)
    own, opp = ps[yi], ps[1 - yi]
    deck = read_deck_csv()
    from collections import Counter
    # The stadium is a shared in-play area outside hand/bench/discard/prize; it
    # must be attributed to its owner so the hidden-state multiset stays exact.
    stadium_ids = {0: [], 1: []}
    for sc in (cur.get("stadium") or []):
        if isinstance(sc, dict) and sc.get("id") is not None:
            owner = sc.get("playerIndex", yi)
            if owner in (0, 1):
                stadium_ids[owner].append(sc.get("id"))
    pool = Counter(deck)
    for cid in _known_ids(own):
        pool[cid] -= 1
    for cid in stadium_ids[yi]:
        pool[cid] -= 1
    remaining = [cid for cid, n in pool.items() for _ in range(max(0, n))]
    prize_n = _prize_count(own)
    deck_n = own.get("deckCount", 0)
    # Require an exact reconstruction; otherwise bail out to the heuristic.
    if len(remaining) != prize_n + deck_n:
        return None
    rng.shuffle(remaining)
    my_prize = remaining[:prize_n]
    my_deck = remaining[prize_n:prize_n + deck_n]
    # opponent: mirror-deck prior minus observed cards
    opool = Counter(deck)
    for cid in _known_ids(opp):
        opool[cid] -= 1
    for cid in stadium_ids[1 - yi]:
        opool[cid] -= 1
    orem = [cid for cid, n in opool.items() for _ in range(max(0, n))]
    rng.shuffle(orem)
    op_n = _prize_count(opp)
    oh_n = opp.get("handCount", 0)
    od_n = opp.get("deckCount", 0)
    opp_prize = orem[:op_n]
    opp_hand = orem[op_n:op_n + oh_n]
    opp_deck = orem[op_n + oh_n:op_n + oh_n + od_n]
    opp_active = []
    oa = opp.get("active") or []
    if oa and oa[0] is None:
        # face-down opponent active: predict a basic from our pool
        basics = [cid for cid in deck if card(cid) and card(cid).get("basic")]
        opp_active = [rng.choice(basics)] if basics else [DREEPY]
    return {"your_deck": my_deck, "your_prize": my_prize, "opponent_deck": opp_deck,
            "opponent_prize": opp_prize, "opponent_hand": opp_hand, "opponent_active": opp_active}


def _eval_state(native_obs, yi):
    """Richer state evaluation from the perspective of player yi."""
    cur = getattr(native_obs, "current", None)
    if cur is None:
        return 0.0
    result = int(getattr(cur, "result", -1))
    if result >= 0:
        if result == yi:
            return 100000.0
        if result == 1 - yi:
            return -100000.0
        return 0.0
    players = getattr(cur, "players", []) or []
    if len(players) < 2:
        return 0.0
    own, opp = players[yi], players[1 - yi]
    op = len(getattr(opp, "prize", []) or [])
    mp = len(getattr(own, "prize", []) or [])
    score = 150.0 * (op - mp)
    # board presence & quality
    def board_value(p):
        v = 0.0
        a = getattr(p, "active", []) or []
        act = a[0] if a and a[0] is not None else None
        bench = getattr(p, "bench", []) or []
        for pk in ([act] if act else []) + list(bench):
            cid = pk.id
            c = card(cid)
            hp = int(getattr(pk, "hp", 0) or 0)
            maxhp = int(getattr(pk, "maxHp", 0) or 0)
            base = 20
            if c:
                if c.get("ex"):
                    base = 35
                if cid in DRAGAPULT_LINE:
                    base = 45 if cid == DRAGAPULT else 30
            v += base + hp * 0.15
            # readiness to attack
            en = [int(e) for e in (getattr(pk, "energies", []) or [])]
            v += len(en) * 6
        return v
    score += board_value(own) - board_value(opp)
    score += 3.0 * (int(getattr(own, "handCount", 0) or 0) - int(getattr(opp, "handCount", 0) or 0))
    # active HP matchup
    oa = getattr(own, "active", []) or []
    ea = getattr(opp, "active", []) or []
    ohp = int(getattr(oa[0], "hp", 0) or 0) if oa and oa[0] is not None else 0
    ehp = int(getattr(ea[0], "hp", 0) or 0) if ea and ea[0] is not None else 0
    score += 0.2 * (ohp - ehp)
    return score


def _rollout_policy(native_obs):
    """Heuristic action for a search observation (both seats). Returns indices."""
    sel = getattr(native_obs, "select", None)
    if sel is None:
        return None
    options = getattr(sel, "option", None) or []
    if not options:
        return None
    try:
        obs_dict = _native_to_dict(native_obs)
        act = handle_selection(obs_dict)
        if act is None:
            act = _main_heuristic(obs_dict)
        if not act:
            mn = int(getattr(sel, "minCount", 1) or 1)
            act = list(range(min(max(mn, 1), len(options))))
        return act
    except Exception:
        mn = int(getattr(sel, "minCount", 1) or 1)
        mx = int(getattr(sel, "maxCount", 1) or 1)
        n = min(max(mn, 1), mx, len(options))
        return list(range(n))


def _native_to_dict(native_obs):
    """Convert a search Observation dataclass back to a plain dict for handlers."""
    from dataclasses import asdict
    try:
        return asdict(native_obs)
    except Exception:
        return {}


def monte_carlo_main(obs, heuristic_scores):
    """Refine MAIN single-choice scores with budgeted rollouts.

    Returns a list of combined scores (same length as options), or None if
    search is unavailable / over budget / fails. Never raises.
    """
    if not MC_ENABLED:
        return None
    sel = _select(obs)
    cur = _current(obs)
    if sel is None or cur is None:
        return None
    options = sel.get("option") or []
    if len(options) < 2 or len(options) > MC_MAX_OPTIONS:
        return None
    if int(sel.get("maxCount", 1)) != 1 or int(sel.get("minCount", 1)) != 1:
        return None
    if int(sel.get("context", 0)) != CX_MAIN or int(sel.get("type", 0)) != 0:
        return None
    # time budget from overage bank
    try:
        rot = float(obs.get("remainingOverageTime", 600.0))
    except (TypeError, ValueError):
        rot = 600.0
    budget = min(MC_TIME_CAP_S, max(0.0, (rot - MC_TIME_RESERVE_S) * 0.02))
    if budget <= 0.02:
        return None
    import time
    t_start = time.perf_counter()
    try:
        from cg import api
        native = api.to_observation_class(obs)
        if getattr(native, "search_begin_input", None) is None:
            return None
        if getattr(native, "current", None) is None or getattr(native, "select", None) is None:
            return None
    except Exception:
        return None
    yi = _yi(cur)
    rng = random.Random(12345)
    n = len(options)
    # Conservative refinement: only consider the top cluster of near-equivalent
    # heuristic options. A clearly-best heuristic choice is never overridden.
    hmax = max(heuristic_scores)
    hmin = min(heuristic_scores)
    margin = max(25.0, 0.25 * (hmax - hmin))
    candidates = [i for i in range(n) if heuristic_scores[i] >= hmax - margin]
    if len(candidates) < 2:
        return None
    sums = {i: 0.0 for i in candidates}
    counts = {i: 0 for i in candidates}
    total_rollouts = MC_ROLLOUTS_PER_OPTION * len(candidates)
    for r in range(total_rollouts):
        if time.perf_counter() - t_start > budget:
            break
        root = candidates[r % len(candidates)]
        hidden = _build_hidden(obs, rng)
        if hidden is None:
            return None
        ss = None
        try:
            ss = api.search_begin(native, hidden["your_deck"], hidden["your_prize"],
                                  hidden["opponent_deck"], hidden["opponent_prize"],
                                  hidden["opponent_hand"], hidden["opponent_active"], False)
            ss = api.search_step(ss.searchId, [root])
            for _ in range(MC_DEPTH):
                so = ss.observation
                if getattr(so, "select", None) is None:
                    break
                if getattr(so.current, "result", -1) >= 0:
                    break
                act = _rollout_policy(so)
                if act is None:
                    break
                ss = api.search_step(ss.searchId, act)
            sums[root] += _eval_state(ss.observation, yi)
            counts[root] += 1
        except Exception:
            pass
        finally:
            if ss is not None:
                try:
                    api.search_release(ss.searchId)
                except Exception:
                    pass
    try:
        api.search_end()
    except Exception:
        pass
    sampled = [i for i in candidates if counts[i] > 0]
    if len(sampled) < 2:
        return None
    means = {i: sums[i] / counts[i] for i in sampled}
    hbest = max(range(n), key=lambda i: heuristic_scores[i])
    best = max(sampled, key=lambda i: means[i])
    # Only flip to a different option when search shows a meaningful advantage;
    # otherwise keep the heuristic choice (guards against rollout noise).
    if best == hbest or hbest not in means:
        return None
    if means[best] - means[hbest] > 5.0:
        adjusted = list(heuristic_scores)
        adjusted[best] = hmax + 1.0
        return adjusted
    return None


def _main_heuristic(obs):
    sel = _select(obs)
    options = sel.get("option") or []
    if not options:
        return []
    scores = [score_main_option(obs, o) for o in options]
    best = max(range(len(options)), key=lambda i: (scores[i], -i))
    return [best]


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def agent(observation):
    try:
        observation = _unwrap(observation)
        if not isinstance(observation, dict):
            return []
        sel = observation.get("select")
        if sel is None:
            return read_deck_csv()
        if not isinstance(sel, dict):
            return []
        options = sel.get("option") or []
        if not options:
            return []
        # The policy keeps no per-game mutable state, so it is naturally safe
        # across Best-of-3 rounds (current.round / players[*].win change).
        # single-option fast path
        min_count = int(sel.get("minCount", 1) or 0)
        max_count = int(sel.get("maxCount", 1) or 0)
        if max_count <= 0:
            return []
        if len(options) == 1 and min_count >= 1:
            return [0]
        act = handle_selection(observation)
        if act is None:
            # MAIN single-choice: heuristic + optional search
            scores = [score_main_option(observation, o) for o in options]
            if len(options) >= 2 and max_count == 1 and min_count == 1:
                combined = monte_carlo_main(observation, scores)
                if combined is not None:
                    scores = combined
            best = max(range(len(options)), key=lambda i: (scores[i], -i))
            act = [best]
        return _sanitize(act, options, min_count, max_count)
    except Exception:
        # absolute fallback: return a legal selection no matter what
        try:
            sel = observation.get("select") if isinstance(observation, dict) else None
            if sel is None:
                return read_deck_csv()
            options = sel.get("option") or []
            if not options:
                return []
            mn = int(sel.get("minCount", 1) or 1)
            mx = int(sel.get("maxCount", 1) or 1)
            n = min(max(mn, 1), mx, len(options))
            return list(range(n))
        except Exception:
            return []


def _sanitize(act, options, min_count, max_count):
    n = len(options)
    max_count = min(max_count, n)
    min_count = min(min_count, max_count)
    seen = set()
    out = []
    for i in act or []:
        if isinstance(i, int) and 0 <= i < n and i not in seen:
            out.append(i)
            seen.add(i)
    if len(out) < min_count:
        for i in range(n):
            if i not in seen:
                out.append(i)
                seen.add(i)
            if len(out) >= min_count:
                break
    if len(out) > max_count:
        out = out[:max_count]
    return out
