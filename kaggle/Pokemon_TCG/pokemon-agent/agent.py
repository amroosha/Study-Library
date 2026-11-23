import os
import random


PRIORITY = {
    8: 1000,   # Attach
    9: 850,    # Evolve
    7: 700,    # Play
    10: 550,   # Ability
    13: 300,   # Attack
    14: 1,     # End turn
    12: -5,    # Retreat
    11: 50,    # Discard
}

MONTE_CARLO_ROLLOUTS = 16
MONTE_CARLO_DEPTH = 8


def _unwrap_observation(observation):
    if isinstance(observation, dict):
        if isinstance(observation.get("observation"), dict):
            observation = observation["observation"]
        if isinstance(observation.get("obs"), dict):
            observation = observation["obs"]
    return observation


def read_deck_csv():
    candidates = (
        os.path.join(os.path.dirname(__file__), "deck.csv"),
        "deck.csv",
        "/kaggle_simulations/agent/deck.csv",
    )
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as deck_file:
                deck = [
                    int(line.strip())
                    for line in deck_file
                    if line.strip()
                ]
            if len(deck) == 60:
                return deck
        except (OSError, ValueError):
            continue
    raise ValueError("deck.csv must contain exactly 60 card IDs")


def heuristic_action(observation):
    if not isinstance(observation, dict):
        return []

    select = observation.get("select")
    if select is None:
        return read_deck_csv()
    if not isinstance(select, dict):
        return []

    options = select.get("option") or []
    if not options:
        return []

    max_count = int(select.get("maxCount", 1) or 0)
    min_count = int(select.get("minCount", max_count) or 0)
    if max_count <= 0:
        return []

    context = int(select.get("context", 0) or 0)

    # Context 41 asks whether to go first. Prefer going second only in the
    # public sample's YES/NO convention when option types identify the choice.
    if context == 41:
        for index, option in enumerate(options):
            if int(option.get("type", -1)) == 1:
                return [index]

    scored = []
    for index, option in enumerate(options):
        if not isinstance(option, dict):
            scored.append((0, -index, index))
            continue
        option_type = int(option.get("type", -1))
        score = PRIORITY.get(option_type, 50)
        scored.append((score, -index, index))

    scored.sort(reverse=True)
    count = max(min_count if min_count > 0 else 1, 1)
    count = min(count, max_count, len(scored))
    return [item[2] for item in scored[:count]]


def _native_card_ids(cards, expected_count, fallback_ids):
    ids = []
    for card in cards or []:
        card_id = getattr(card, "id", None)
        if card_id is not None:
            ids.append(int(card_id))
    while len(ids) < expected_count:
        ids.append(fallback_ids[len(ids) % len(fallback_ids)])
    return ids[:expected_count]


def _native_option_action(native_observation, use_heuristic=True):
    select = getattr(native_observation, "select", None)
    if select is None:
        return None

    options = getattr(select, "option", None) or []
    if not options:
        return None

    max_count = int(getattr(select, "maxCount", 1) or 1)
    min_count = int(getattr(select, "minCount", 1) or 1)
    context = int(getattr(select, "context", 0) or 0)

    if context == 41:
        for index, option in enumerate(options):
            if int(getattr(option, "type", -1)) == 1:
                return [index]

    count = max(min_count, 1)
    count = min(count, max_count, len(options))

    if use_heuristic:
        scored = []
        for index, option in enumerate(options):
            option_type = int(getattr(option, "type", -1))
            scored.append((PRIORITY.get(option_type, 50), -index, index))
        scored.sort(reverse=True)
        return [item[2] for item in scored[:count]]

    if count == 1:
        return [random.randrange(len(options))]
    return random.sample(range(len(options)), count)


def _score_native_state(native_observation):
    current = getattr(native_observation, "current", None)
    if current is None:
        return 0.0

    result = int(getattr(current, "result", -1))
    if result >= 0:
        your_index = int(getattr(current, "yourIndex", 0))
        return 10000.0 if result == your_index else -10000.0

    players = getattr(current, "players", []) or []
    if len(players) < 2:
        return 0.0

    your_index = int(getattr(current, "yourIndex", 0))
    opponent_index = 1 - your_index
    own = players[your_index]
    opponent = players[opponent_index]

    own_prizes = len(getattr(own, "prize", []) or [])
    opponent_prizes = len(getattr(opponent, "prize", []) or [])
    own_bench = len(getattr(own, "bench", []) or [])
    opponent_bench = len(getattr(opponent, "bench", []) or [])
    own_hand = int(getattr(own, "handCount", 0) or 0)
    opponent_hand = int(getattr(opponent, "handCount", 0) or 0)

    own_active = getattr(own, "active", []) or []
    opponent_active = getattr(opponent, "active", []) or []
    own_hp = sum(int(getattr(pokemon, "hp", 0) or 0) for pokemon in own_active if pokemon is not None)
    opponent_hp = sum(int(getattr(pokemon, "hp", 0) or 0) for pokemon in opponent_active if pokemon is not None)

    return (
        120.0 * (opponent_prizes - own_prizes)
        + 20.0 * (own_bench - opponent_bench)
        + 2.0 * (own_hand - opponent_hand)
        + 0.1 * (own_hp - opponent_hp)
    )


def monte_carlo_action(observation, rollout_count=MONTE_CARLO_ROLLOUTS, depth=MONTE_CARLO_DEPTH):
    select = observation.get("select") if isinstance(observation, dict) else None
    options = select.get("option", []) if isinstance(select, dict) else []
    if not options:
        return []
    if len(options) == 1:
        return [0]

    max_count = int(select.get("maxCount", 1) or 1)
    if max_count != 1:
        return heuristic_action(observation)

    try:
        from cg import api

        native_observation = api.to_observation_class(observation)
        deck = read_deck_csv()
        current = native_observation.current
        if current is None or native_observation.select is None:
            return heuristic_action(observation)

        your_index = int(current.yourIndex)
        opponent_index = 1 - your_index
        own = current.players[your_index]
        opponent = current.players[opponent_index]

        fallback = deck or [1]
        your_prize = _native_card_ids(own.prize, len(own.prize or []), fallback)
        opponent_prize = _native_card_ids(opponent.prize, len(opponent.prize or []), fallback)
        opponent_deck = _native_card_ids([], opponent.deckCount, fallback)
        opponent_hand = _native_card_ids([], opponent.handCount, fallback)

        opponent_active = []
        active = opponent.active or []
        if active and active[0] is None:
            opponent_active = [fallback[0]]

        scores = [0.0] * len(options)
        root_action_counts = [0] * len(options)
        remaining = rollout_count

        if len(options) > 0:
            base = remaining // len(options)
            extra = remaining % len(options)
            for root_index in range(len(options)):
                root_action_counts[root_index] = base + (1 if root_index < extra else 0)

        for root_index, rollout_count_for_action in enumerate(root_action_counts):
            for _ in range(rollout_count_for_action):
                search_state = None
                try:
                    search_state = api.search_begin(
                        native_observation,
                        deck,
                        your_prize,
                        opponent_deck,
                        opponent_prize,
                        opponent_hand,
                        opponent_active,
                        False,
                    )
                    search_state = api.search_step(search_state.searchId, [root_index])

                    for _ in range(depth):
                        search_observation = search_state.observation
                        if getattr(search_observation, "select", None) is None:
                            break
                        rollout_action = _native_option_action(search_observation, use_heuristic=False)
                        if rollout_action is None:
                            break
                        search_state = api.search_step(search_state.searchId, rollout_action)

                    scores[root_index] += _score_native_state(search_state.observation)
                except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
                    scores[root_index] += -10000.0
                finally:
                    if search_state is not None:
                        try:
                            api.search_release(search_state.searchId)
                        except (AttributeError, RuntimeError):
                            pass

        if not any(root_action_counts):
            return heuristic_action(observation)

        best_index = max(
            range(len(options)),
            key=lambda index: (
                scores[index] / max(root_action_counts[index], 1),
                PRIORITY.get(int(options[index].get("type", -1)), 50),
                -index,
            ),
        )
        return [best_index]
    except (ImportError, ModuleNotFoundError):
        return heuristic_action(observation)


def agent(observation):
    observation = _unwrap_observation(observation)
    if not isinstance(observation, dict):
        return []
    if observation.get("select") is None:
        return read_deck_csv()
    return monte_carlo_action(observation)
