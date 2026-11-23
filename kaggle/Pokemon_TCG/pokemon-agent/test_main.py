import unittest

import main


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
        action = main.heuristic_action(self.observation)
        self.assertEqual(action, [0])

    def test_single_option_is_returned(self):
        observation = dict(self.observation)
        observation["select"] = {"option": ["Pass"], "minCount": 1, "maxCount": 1}
        self.assertEqual(main.agent(observation), [0])

    def test_empty_selection_is_safe(self):
        observation = {"select": {"option": [], "minCount": 0, "maxCount": 0}}
        self.assertEqual(main.agent(observation), [])


if __name__ == "__main__":
    unittest.main()
