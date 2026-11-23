import unittest

import agent


class AgentBaselineTests(unittest.TestCase):
    def setUp(self):
        self.observation = {
            "player": 0,
            "select": {
                "option": ["Attach Energy", "End Turn", "Attack"],
                "minCount": 1,
                "maxCount": 1,
            },
            "current": {
                "players": [
                    {
                        "prize": [1, 2, 3, None, None, None],
                        "bench": [1, None, None, None, None],
                        "hand": [1, 2, 3],
                        "active": {"hp": 120},
                    },
                    {
                        "prize": [1, 2, 3, 4, None, None],
                        "bench": [1, 2, None, None, None],
                        "hand": [1, 2],
                        "active": {"hp": 80},
                    },
                ]
            },
        }

    def test_attach_energy_preferred_over_end_turn(self):
        action = agent.heuristic_action(self.observation)
        self.assertEqual(action, [0])

    def test_single_option_is_returned(self):
        observation = dict(self.observation)
        observation["select"] = {"option": ["Pass"], "minCount": 1, "maxCount": 1}
        self.assertEqual(agent.agent(observation), [0])

    def test_empty_selection_is_safe(self):
        observation = {"select": {"option": [], "minCount": 0, "maxCount": 0}}
        self.assertEqual(agent.agent(observation), [])

    def test_initial_selection_returns_exactly_60_cards(self):
        deck = agent.agent({"select": None, "current": None})
        self.assertEqual(len(deck), 60)

    def test_monte_carlo_falls_back_for_multi_select(self):
        observation = {
            "select": {
                "option": [{"type": 8}, {"type": 14}],
                "minCount": 2,
                "maxCount": 2,
                "context": 0,
            }
        }
        action = agent.monte_carlo_action(observation)
        self.assertEqual(len(action), 2)


if __name__ == "__main__":
    unittest.main()
