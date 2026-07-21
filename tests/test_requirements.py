import unittest

import numpy as np

from escape_room.envs import (
    CONTINUOUS_ACTIONS,
    ContinuousEscapeRoom,
    DynamicObstacleRoom,
    GridEscapeRoom,
    SokobanEscapeRoom,
    continuous_room_config,
    obstacle_room_config,
    room1_config,
    room2_config,
    room3_config,
)


class ProjectRequirementTests(unittest.TestCase):
    def test_first_three_rooms_are_10_by_10(self):
        rooms = [
            GridEscapeRoom(room1_config()),
            SokobanEscapeRoom(room2_config()),
            GridEscapeRoom(room3_config()),
        ]
        self.assertTrue(all(room.rows == 10 and room.cols == 10 for room in rooms))

    def test_known_room_has_slip_and_complete_transition_model(self):
        room = GridEscapeRoom(room1_config())
        self.assertGreater(len(room.config.slippery), 0)
        self.assertGreater(len(room.config.guard_cycles), 0)
        outcomes = room.transition_model(room.reset(), 1)
        self.assertAlmostEqual(sum(item[0] for item in outcomes), 1.0)

    def test_sarsa_room_is_real_sokoban_and_has_slip(self):
        room = SokobanEscapeRoom(room2_config(slip_probability=0.0))
        self.assertGreater(len(room.config.slippery), 0)
        for _ in range(7):
            room.step(1)
        state, reward, done, info = room.step(1)
        self.assertEqual(state, (0, 8, 0, 9))
        self.assertTrue(info["pushed"])
        self.assertTrue(info["box_locked"])
        self.assertFalse(done)
        self.assertGreater(reward, 0)

    def test_rewards_cannot_be_farmed_from_bonus_or_portal_loops(self):
        self.assertEqual(room1_config().bonuses, {})
        self.assertEqual(room2_config().bonuses, {})
        self.assertEqual(room3_config().bonuses, {})
        self.assertEqual(room3_config().portal_reward, 0.0)

    def test_grid_rooms_have_one_canonical_terminal_state(self):
        for config in (room1_config(), room3_config()):
            room = GridEscapeRoom(config)
            terminals = [state for state in room.all_states() if room.is_terminal_state(state)]
            self.assertEqual(len(terminals), 1)

    def test_continuous_room_matches_required_dynamics(self):
        config = continuous_room_config()
        room = ContinuousEscapeRoom(config)
        self.assertEqual(config.room_size, 10.0)
        self.assertEqual(config.dt, 0.02)
        self.assertEqual(len(CONTINUOUS_ACTIONS), 9)
        self.assertEqual(set(CONTINUOUS_ACTIONS), {(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)})
        state, _, _, _ = room.step(8)
        np.testing.assert_allclose(state, np.array([0.52, 0.52, 1.0, 1.0]))
        room.x, room.y = config.goal
        terminal_state, _, done, _ = room.step(4)
        self.assertTrue(done)
        np.testing.assert_allclose(terminal_state, np.array([config.goal[0], config.goal[1], 0.0, 0.0]))

    def test_dynamic_obstacle_room_matches_optional_specification(self):
        config = obstacle_room_config(seed=9, obstacle_count=6, observation_range=4.0)
        room = DynamicObstacleRoom(config)
        state = room.reset()
        self.assertEqual(config.obstacle_width, 0.5)
        self.assertEqual(len(room.obstacles), 6)
        self.assertEqual(config.observation_range, 4.0)
        self.assertEqual(len(state), 7)


if __name__ == "__main__":
    unittest.main()
