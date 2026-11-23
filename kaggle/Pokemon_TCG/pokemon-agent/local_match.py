import os
import random
import sys

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(AGENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from cg import game

from agent import agent as heuristic_agent
from agent import read_deck_csv


def random_agent(observation):
    select = observation.get("select")
    if select is None:
        return read_deck_csv()
    options = select.get("option", [])
    if not options:
        return []
    max_count = max(1, int(select.get("maxCount", 1)))
    count = random.randint(int(select.get("minCount", 1)), min(max_count, len(options)))
    return random.sample(range(len(options)), count)


def play_match(match_number, opponent=random_agent):
    deck = read_deck_csv()

    try:
        observation, start = game.battle_start(deck, deck)
        if observation is None:
            return {"match": match_number, "error": "battle_start failed"}

        for turn in range(5000):
            if observation is None:
                return {"match": match_number, "error": "null observation"}

            select = observation.get("select")
            current = observation.get("current")
            if select is None:
                if isinstance(current, dict):
                    winner = current.get("result")
                    if winner in (0, 1):
                        return {"match": match_number, "winner": winner}
                return {"match": match_number, "error": "game ended or has no selection"}
            if isinstance(select, dict) and not select.get("option"):
                if isinstance(current, dict):
                    winner = current.get("result")
                    if winner in (0, 1):
                        return {"match": match_number, "winner": winner}
                return {
                    "match": match_number,
                    "error": "no legal options",
                    "context": select.get("context"),
                }

            your_index = current.get("yourIndex", 0) if isinstance(current, dict) else 0
            if your_index == 0:
                action = heuristic_agent(observation)
            else:
                action = opponent(observation)

            try:
                observation = game.battle_select(action)
            except IndexError:
                select = observation.get("select", {})
                return {
                    "match": match_number,
                    "error": "invalid action",
                    "action": action,
                    "minCount": select.get("minCount"),
                    "maxCount": select.get("maxCount"),
                    "optionCount": len(select.get("option", [])),
                    "context": select.get("context"),
                }

        return {"match": match_number, "error": "step limit reached"}
    finally:
        try:
            game.battle_finish()
        except Exception:
            pass


def main():
    random.seed(2026)
    results = [play_match(index + 1) for index in range(10)]
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
