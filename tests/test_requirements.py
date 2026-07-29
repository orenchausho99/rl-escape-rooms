import unittest
from pathlib import Path

import numpy as np

from escape_room.algorithms import train_approx_q_learning, value_iteration
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
from escape_room.replay import filter_replay_attempts, replay_library_rows


class ProjectRequirementTests(unittest.TestCase):
    def test_static_art_urls_work_behind_streamlit_cloud_prefix(self):
        app_source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
        self.assertIn("app/static/game_art/", app_source)
        self.assertNotIn("/app/static/game_art/", app_source)

    def test_player_profile_wraps_campaign_without_changing_room_order(self):
        app_source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
        self.assertIn('"Continue as guest"', app_source)
        self.assertIn("def render_campaign_dashboard", app_source)
        self.assertIn('[data-testid="stTextInput"] input', app_source)
        self.assertNotIn('"Training style"', app_source)
        self.assertNotIn("PLAYER_ROLES", app_source)
        self.assertIn('st.tabs(["Play Game", "Train Agent", "Episode Replay", "Analytics", "Room Specs"])', app_source)

    def test_training_status_messages_keep_readable_contrast(self):
        app_source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
        self.assertIn('[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p', app_source)
        self.assertIn("-webkit-text-fill-color: #eef2f7 !important", app_source)

    def test_episode_count_is_configurable_only_for_learning_rooms(self):
        app_source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
        self.assertIn("EPISODE_CONTROLS", app_source)
        self.assertIn('"Episodes - training attempts"', app_source)
        self.assertIn('episode_control = EPISODE_CONTROLS[room_kind]', app_source)
        self.assertIn("Value Iteration uses model sweeps, not training episodes", app_source)
        for room_kind in ("sarsa", "q_learning", "approx", "obstacles"):
            self.assertIn(f'"{room_kind}": {{"min":', app_source)
        self.assertNotIn('"dp": {"min":', app_source)

    def test_room_selection_thumbnails_are_packaged(self):
        art_dir = Path(__file__).parents[1] / "static" / "game_art"
        names = (
            "pacman-ice-thumbnail-v2.webp",
            "sokoban-vault-thumbnail-v2.webp",
            "bomberman-reactor-thumbnail-v2.webp",
            "lunar-lander-thumbnail-v2.webp",
            "portal-hazard-thumbnail-v2.webp",
            "pacman-ice-banner-v2.webp",
            "sokoban-vault-banner-v2.webp",
            "bomberman-reactor-banner-v2.webp",
            "lunar-lander-banner-v2.webp",
            "portal-hazard-banner-v2.webp",
        )
        self.assertTrue(all((art_dir / name).is_file() for name in names))

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

    def test_dp_rollout_step_limit_is_configurable(self):
        room = GridEscapeRoom(room1_config())
        result = value_iteration(room, max_iterations=1, rollout_max_steps=3)
        self.assertEqual(len(result["attempts"]), 12)
        self.assertTrue(all(attempt["steps"] <= 3 for attempt in result["attempts"]))

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

    def test_slippery_tiles_are_distributed_across_the_grid(self):
        for config in (room1_config(), room2_config()):
            slippery = config.slippery
            self.assertGreaterEqual(len(slippery), 8)
            self.assertLessEqual(min(row for row, _col in slippery), 2)
            self.assertGreaterEqual(max(row for row, _col in slippery), 8)
            self.assertLessEqual(min(col for _row, col in slippery), 1)
            self.assertGreaterEqual(max(col for _row, col in slippery), 8)
            self.assertFalse(slippery & config.walls)
            self.assertNotIn(config.start, slippery)
            self.assertNotIn(config.goal, slippery)
            for row, col in slippery:
                neighbors = {(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)}
                self.assertFalse(neighbors & slippery)

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
        self.assertEqual(len(config.hazards), 5)
        state, _, _, _ = room.step(8)
        np.testing.assert_allclose(state, np.array([0.52, 0.52, 1.0, 1.0]))
        hazard_x, hazard_y, _, hazard_top = config.hazards[0]
        room.x, room.y = hazard_x - 0.01, (hazard_y + hazard_top) / 2
        collision_state, _, _, collision_info = room.step(5)
        self.assertTrue(collision_info["hit_hazard"])
        np.testing.assert_allclose(
            collision_state,
            np.array([hazard_x - 0.01, (hazard_y + hazard_top) / 2, 0.0, 0.0]),
        )
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

        room.x, room.y = 4.0, 4.0
        room.vx = room.vy = 0
        room.obstacles = [{"x": 4.0, "y": 4.0, "axis": 0.0, "direction": 1.0}]
        teleported_state, reward, done, info = room.step(4)
        self.assertTrue(info["hit_obstacle"])
        self.assertTrue(info["teleported"])
        self.assertNotEqual(tuple(teleported_state[:2]), config.start)
        self.assertEqual(tuple(teleported_state[2:4]), (0.0, 0.0))
        self.assertLess(reward, 0.0)
        self.assertFalse(done)

    def test_replay_library_contains_every_episode(self):
        attempts = [
            {"episode": 1, "reward": -4.0, "steps": 5, "success": False, "epsilon": 0.8},
            {"episode": 2, "reward": 12.0, "steps": 4, "success": True, "epsilon": 0.5},
            {"episode": 3, "reward": 20.0, "steps": 3, "success": True, "epsilon": 0.2},
        ]
        filtered = filter_replay_attempts(attempts, "All episodes", "Episode: oldest first")
        rows = replay_library_rows(filtered, total_episodes=3)
        self.assertEqual([row["Episode"] for row in rows], [1, 2, 3])
        self.assertEqual(len(rows), len(attempts))
        self.assertEqual(
            [item["episode"] for item in filter_replay_attempts(attempts, "Successful only", "Score: highest first")],
            [3, 2],
        )

    def test_continuous_replay_uses_environment_time_step(self):
        app_source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
        self.assertIn("const replayBaseDelay = cfg.mode === 'continuous' ? 20 : 450", app_source)
        self.assertIn("replayIndex + frameAdvance", app_source)
        self.assertIn("Step ' + replayIndex + ' of '", app_source)
        self.assertIn("function createLanderMeteors()", app_source)
        self.assertIn("if (!cont.meteors.length) cont.meteors = createLanderMeteors()", app_source)
        self.assertIn("if (!Number.isFinite(now)) return", app_source)
        self.assertIn("requestAnimationFrame(loop);", app_source)

    def test_fast_tuning_skips_replay_recording_and_reports_progress(self):
        room = DynamicObstacleRoom(obstacle_room_config(seed=12, obstacle_count=3))
        progress = []
        result = train_approx_q_learning(
            room,
            episodes=3,
            max_steps=8,
            seed=12,
            record_replay=False,
            progress_callback=lambda completed, total: progress.append((completed, total)),
        )
        self.assertEqual(len(result["metrics"]), 3)
        self.assertEqual(result["attempts"], [])
        self.assertEqual(result["snapshots"], [])
        self.assertEqual(progress[-1], (3, 3))


if __name__ == "__main__":
    unittest.main()
