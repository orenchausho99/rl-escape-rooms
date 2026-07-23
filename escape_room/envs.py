from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


GridPos = Tuple[int, int]
GridState = Tuple[int, ...]

ACTIONS = ("up", "right", "down", "left")
ACTION_LABELS_EN = ("Up", "Right", "Down", "Left")
ACTION_TO_DELTA = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}
LEFT_TURN = {0: 3, 1: 0, 2: 1, 3: 2}
RIGHT_TURN = {0: 1, 1: 2, 2: 3, 3: 0}


@dataclass(frozen=True)
class GridRoomConfig:
    name: str
    start: GridPos
    goal: GridPos
    walls: frozenset[GridPos] = field(default_factory=frozenset)
    slippery: frozenset[GridPos] = field(default_factory=frozenset)
    traps: Dict[GridPos, float] = field(default_factory=dict)
    bonuses: Dict[GridPos, float] = field(default_factory=dict)
    keys: Tuple[GridPos, ...] = ()
    portals: Dict[GridPos, GridPos] = field(default_factory=dict)
    guard_cycles: Tuple[Tuple[GridPos, ...], ...] = ()
    box_start: Optional[GridPos] = None
    box_target: Optional[GridPos] = None
    slip_probability: float = 0.2
    step_reward: float = -1.0
    goal_reward: float = 100.0
    key_reward: float = 20.0
    blocked_goal_penalty: float = -6.0
    guard_reward: float = -55.0
    portal_reward: float = 4.0
    seed: int = 7


class GridEscapeRoom:
    """A 10x10 tabular escape-room environment.

    State is always a tuple: (row, col, key_mask, guard_phase).
    For simple rooms key_mask and guard_phase are 0.
    """

    rows = 10
    cols = 10
    action_count = 4

    def __init__(self, config: GridRoomConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.position: GridPos = config.start
        self.key_mask = 0
        self.guard_phase = 0
        self.steps = 0

    @property
    def required_key_mask(self) -> int:
        return (1 << len(self.config.keys)) - 1

    @property
    def guard_period(self) -> int:
        if not self.config.guard_cycles:
            return 1
        period = 1
        for cycle in self.config.guard_cycles:
            period = math.lcm(period, len(cycle))
        return period

    def copy_with(self, *, seed: Optional[int] = None) -> "GridEscapeRoom":
        config = self.config if seed is None else replace(self.config, seed=seed)
        return GridEscapeRoom(config)

    def reset(self) -> GridState:
        self.position = self.config.start
        self.key_mask = 0
        self.guard_phase = 0
        self.steps = 0
        return self.state()

    def state(self) -> GridState:
        row, col = self.position
        return (row, col, self.key_mask, self.guard_phase)

    def all_states(self) -> List[GridState]:
        states: List[GridState] = []
        key_masks = range(1 << len(self.config.keys))
        phases = range(self.guard_period)
        for row in range(self.rows):
            for col in range(self.cols):
                if (row, col) in self.config.walls:
                    continue
                for key_mask in key_masks:
                    for phase in phases:
                        if (
                            (row, col) == self.config.goal
                            and key_mask == self.required_key_mask
                            and phase != 0
                        ):
                            continue
                        states.append((row, col, key_mask, phase))
        return states

    def available_actions(self, _state: Optional[GridState] = None) -> range:
        return range(self.action_count)

    def is_terminal_state(self, state: GridState) -> bool:
        row, col, key_mask, phase = state
        return (row, col) == self.config.goal and key_mask == self.required_key_mask and phase == 0

    def guard_positions(self, phase: Optional[int] = None) -> Tuple[GridPos, ...]:
        if not self.config.guard_cycles:
            return ()
        current_phase = self.guard_phase if phase is None else phase
        return tuple(cycle[current_phase % len(cycle)] for cycle in self.config.guard_cycles)

    def _move(self, position: GridPos, action: int) -> GridPos:
        dr, dc = ACTION_TO_DELTA[action]
        row = min(max(position[0] + dr, 0), self.rows - 1)
        col = min(max(position[1] + dc, 0), self.cols - 1)
        candidate = (row, col)
        if candidate in self.config.walls:
            return position
        return candidate

    def _action_outcomes(self, state: GridState, action: int) -> List[Tuple[float, int]]:
        row, col, _key_mask, _phase = state
        position = (row, col)
        slip = self.config.slip_probability if position in self.config.slippery else 0.0
        if slip <= 0:
            return [(1.0, action)]
        intended = max(0.0, 1.0 - slip)
        side = slip / 2.0
        return [(intended, action), (side, LEFT_TURN[action]), (side, RIGHT_TURN[action])]

    def transition_model(self, state: GridState, action: int) -> List[Tuple[float, GridState, float, bool]]:
        if self.is_terminal_state(state):
            return [(1.0, state, 0.0, True)]

        row, col, key_mask, phase = state
        position = (row, col)
        outcomes: List[Tuple[float, GridState, float, bool]] = []

        for probability, realized_action in self._action_outcomes(state, action):
            new_pos = self._move(position, realized_action)
            reward = self.config.step_reward

            if new_pos in self.config.portals:
                new_pos = self.config.portals[new_pos]
                reward += self.config.portal_reward

            if new_pos in self.config.traps:
                reward += self.config.traps[new_pos]

            if new_pos in self.config.bonuses:
                reward += self.config.bonuses[new_pos]

            new_mask = key_mask
            for index, key_pos in enumerate(self.config.keys):
                if new_pos == key_pos and not (new_mask & (1 << index)):
                    new_mask |= 1 << index
                    reward += self.config.key_reward

            new_phase = (phase + 1) % self.guard_period
            if new_pos in self.guard_positions(new_phase):
                reward += self.config.guard_reward
                new_pos = self.config.start

            done = new_pos == self.config.goal and new_mask == self.required_key_mask
            if new_pos == self.config.goal and not done:
                reward += self.config.blocked_goal_penalty
            if done:
                reward += self.config.goal_reward
                new_phase = 0

            next_state = (new_pos[0], new_pos[1], new_mask, new_phase)
            outcomes.append((probability, next_state, reward, done))

        return outcomes

    def step(self, action: int) -> Tuple[GridState, float, bool, Dict[str, object]]:
        state = self.state()
        model = self.transition_model(state, action)
        threshold = self.rng.random()
        total = 0.0
        selected = model[-1]
        for outcome in model:
            total += outcome[0]
            if threshold <= total:
                selected = outcome
                break
        _probability, next_state, reward, done = selected
        self.position = (next_state[0], next_state[1])
        self.key_mask = next_state[2]
        self.guard_phase = next_state[3]
        self.steps += 1
        return next_state, reward, done, {}


def room1_config(slip_probability: float = 0.25, seed: int = 7) -> GridRoomConfig:
    walls = frozenset(
        {
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 7),
            (4, 7),
            (5, 7),
            (6, 2),
            (6, 3),
            (6, 4),
            (7, 6),
            (8, 6),
        }
    )
    return GridRoomConfig(
        name="Room 1 - Dynamic Programming",
        start=(0, 0),
        goal=(9, 9),
        walls=walls,
        slippery=frozenset({(0, 6), (2, 2), (2, 3), (4, 4), (5, 5), (7, 4), (8, 8)}),
        traps={(4, 1): -22.0, (6, 8): -25.0},
        guard_cycles=(
            ((4, 1), (4, 2), (4, 3), (5, 3), (5, 2), (5, 1)),
            ((6, 8), (6, 9), (7, 9), (8, 9), (8, 8), (7, 8)),
        ),
        slip_probability=slip_probability,
        step_reward=-1.0,
        goal_reward=110.0,
        guard_reward=-45.0,
        seed=seed,
    )


def room2_config(slip_probability: float = 0.2, seed: int = 11) -> GridRoomConfig:
    walls = frozenset(
        {
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 6),
            (3, 6),
            (4, 6),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (7, 5),
            (7, 6),
            (7, 7),
            (8, 2),
        }
    )
    return GridRoomConfig(
        name="Room 2 - SARSA",
        start=(0, 0),
        goal=(9, 9),
        walls=walls,
        slippery=frozenset({(2, 2), (2, 3), (3, 3), (4, 3), (6, 6), (8, 8)}),
        traps={(4, 8): -30.0, (8, 4): -35.0},
        box_start=(0, 8),
        box_target=(0, 9),
        slip_probability=slip_probability,
        step_reward=-1.0,
        goal_reward=130.0,
        key_reward=28.0,
        seed=seed,
    )


def room3_config(slip_probability: float = 0.12, seed: int = 19) -> GridRoomConfig:
    walls = frozenset(
        {
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 1),
            (4, 2),
            (4, 3),
            (5, 6),
            (5, 7),
            (6, 6),
            (7, 1),
            (7, 2),
            (8, 7),
            (8, 8),
        }
    )
    return GridRoomConfig(
        name="Room 3 - Q-Learning",
        start=(0, 0),
        goal=(9, 9),
        keys=((0, 9), (8, 1)),
        walls=walls,
        slippery=frozenset({(2, 2), (3, 8), (6, 3), (8, 5)}),
        traps={(1, 8): -26.0, (6, 8): -38.0, (9, 3): -30.0},
        portals={(3, 0): (8, 6), (8, 6): (3, 0)},
        guard_cycles=(
            ((4, 4), (4, 5), (4, 6), (4, 5)),
            ((7, 3), (7, 4), (7, 5), (7, 6), (7, 5), (7, 4)),
        ),
        slip_probability=slip_probability,
        step_reward=-1.2,
        goal_reward=170.0,
        key_reward=30.0,
        guard_reward=-60.0,
        portal_reward=0.0,
        seed=seed,
    )


class SokobanEscapeRoom(GridEscapeRoom):
    """Model-free 10x10 Sokoban room.

    State is (player_row, player_col, box_row, box_col). The box must be on
    box_target before the single SAFE terminal state can be entered.
    """

    def __init__(self, config: GridRoomConfig):
        if config.box_start is None or config.box_target is None:
            raise ValueError("Sokoban requires box_start and box_target")
        super().__init__(config)
        self.box_position = config.box_start

    def copy_with(self, *, seed: Optional[int] = None) -> "SokobanEscapeRoom":
        config = self.config if seed is None else replace(self.config, seed=seed)
        return SokobanEscapeRoom(config)

    def reset(self) -> GridState:
        self.position = self.config.start
        self.box_position = self.config.box_start or (0, 0)
        self.steps = 0
        return self.state()

    def state(self) -> GridState:
        return (self.position[0], self.position[1], self.box_position[0], self.box_position[1])

    def is_terminal_state(self, state: GridState) -> bool:
        return (state[0], state[1]) == self.config.goal and (state[2], state[3]) == self.config.box_target

    def step(self, action: int) -> Tuple[GridState, float, bool, Dict[str, object]]:
        outcomes = self._action_outcomes((self.position[0], self.position[1], 0, 0), action)
        threshold = self.rng.random()
        total = 0.0
        realized_action = outcomes[-1][1]
        for probability, candidate_action in outcomes:
            total += probability
            if threshold <= total:
                realized_action = candidate_action
                break

        dr, dc = ACTION_TO_DELTA[realized_action]
        candidate = (self.position[0] + dr, self.position[1] + dc)
        reward = self.config.step_reward
        pushed = False
        blocked = False

        if not (0 <= candidate[0] < self.rows and 0 <= candidate[1] < self.cols) or candidate in self.config.walls:
            blocked = True
        elif candidate == self.box_position:
            box_candidate = (candidate[0] + dr, candidate[1] + dc)
            if (
                not (0 <= box_candidate[0] < self.rows and 0 <= box_candidate[1] < self.cols)
                or box_candidate in self.config.walls
            ):
                blocked = True
            else:
                was_on_target = self.box_position == self.config.box_target
                self.position = candidate
                self.box_position = box_candidate
                pushed = True
                if not was_on_target and self.box_position == self.config.box_target:
                    reward += self.config.key_reward
                elif was_on_target and self.box_position != self.config.box_target:
                    reward -= self.config.key_reward
        else:
            self.position = candidate

        if blocked:
            reward += self.config.blocked_goal_penalty
        if self.position in self.config.traps:
            reward += self.config.traps[self.position]

        box_locked = self.box_position == self.config.box_target
        done = self.position == self.config.goal and box_locked
        if self.position == self.config.goal and not done:
            reward += self.config.blocked_goal_penalty
        if done:
            reward += self.config.goal_reward

        self.steps += 1
        return self.state(), reward, done, {"pushed": pushed, "blocked": blocked, "box_locked": box_locked}


CONTINUOUS_ACTIONS: Tuple[Tuple[int, int], ...] = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (0, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


@dataclass(frozen=True)
class ContinuousRoomConfig:
    name: str = "Room 4 - Approximate Q-Learning"
    room_size: float = 10.0
    dt: float = 0.02
    speed: float = 1.0
    start: Tuple[float, float] = (0.5, 0.5)
    goal: Tuple[float, float] = (9.45, 9.45)
    goal_radius: float = 0.42
    hazards: Tuple[Tuple[float, float, float, float], ...] = (
        (1.7, 2.2, 3.8, 2.8),
        (4.3, 4.0, 5.1, 6.5),
        (6.0, 7.1, 8.3, 7.7),
    )
    step_reward: float = -0.015
    goal_reward: float = 35.0
    wall_penalty: float = -0.6
    hazard_penalty: float = -2.5
    progress_scale: float = 4.0
    seed: int = 23


class ContinuousEscapeRoom:
    action_count = len(CONTINUOUS_ACTIONS)

    def __init__(self, config: ContinuousRoomConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.x = config.start[0]
        self.y = config.start[1]
        self.vx = 0
        self.vy = 0
        self.steps = 0

    def copy_with(self, *, seed: Optional[int] = None) -> "ContinuousEscapeRoom":
        config = self.config if seed is None else replace(self.config, seed=seed)
        return ContinuousEscapeRoom(config)

    def reset(self) -> np.ndarray:
        self.x, self.y = self.config.start
        self.vx = 0
        self.vy = 0
        self.steps = 0
        return self.state()

    def state(self) -> np.ndarray:
        return np.array([self.x, self.y, float(self.vx), float(self.vy)], dtype=float)

    def distance_to_goal(self, x: Optional[float] = None, y: Optional[float] = None) -> float:
        px = self.x if x is None else x
        py = self.y if y is None else y
        gx, gy = self.config.goal
        return math.hypot(gx - px, gy - py)

    def _inside_hazard(self, x: float, y: float) -> bool:
        return any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in self.config.hazards)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        self.vx, self.vy = CONTINUOUS_ACTIONS[action]
        old_x, old_y = self.x, self.y
        old_distance = self.distance_to_goal()
        nx = self.x + self.vx * self.config.speed * self.config.dt
        ny = self.y + self.vy * self.config.speed * self.config.dt

        reward = self.config.step_reward
        hit_wall = False
        if nx < 0 or nx > self.config.room_size:
            hit_wall = True
            nx = min(max(nx, 0.0), self.config.room_size)
        if ny < 0 or ny > self.config.room_size:
            hit_wall = True
            ny = min(max(ny, 0.0), self.config.room_size)
        if hit_wall:
            reward += self.config.wall_penalty

        hit_hazard = self._inside_hazard(nx, ny)
        if hit_hazard:
            reward += self.config.hazard_penalty
            nx, ny = old_x, old_y
            self.vx = 0
            self.vy = 0

        self.x, self.y = nx, ny
        new_distance = self.distance_to_goal()
        reward += self.config.progress_scale * (old_distance - new_distance)

        self.steps += 1
        done = new_distance <= self.config.goal_radius
        if done:
            reward += self.config.goal_reward
            self.x, self.y = self.config.goal
            self.vx = 0
            self.vy = 0
        return self.state(), reward, done, {"hit_wall": hit_wall, "hit_hazard": hit_hazard}


def continuous_room_config(seed: int = 23) -> ContinuousRoomConfig:
    return ContinuousRoomConfig(seed=seed, goal_radius=0.55, goal_reward=55.0, progress_scale=5.0)


@dataclass(frozen=True)
class ObstacleRoomConfig(ContinuousRoomConfig):
    name: str = "Room 5 - Dynamic Obstacles"
    obstacle_count: int = 7
    obstacle_width: float = 0.5
    observation_range: float = 3.0
    obstacle_penalty: float = -8.0


class DynamicObstacleRoom(ContinuousEscapeRoom):
    def __init__(self, config: ObstacleRoomConfig):
        super().__init__(config)
        self.config: ObstacleRoomConfig
        self.obstacles: List[Dict[str, float]] = []

    def copy_with(self, *, seed: Optional[int] = None) -> "DynamicObstacleRoom":
        config = self.config if seed is None else replace(self.config, seed=seed)
        return DynamicObstacleRoom(config)

    def reset(self) -> np.ndarray:
        super().reset()
        self.obstacles = []
        attempts = 0
        while len(self.obstacles) < self.config.obstacle_count and attempts < 500:
            attempts += 1
            x = self.rng.uniform(1.4, self.config.room_size - 1.4)
            y = self.rng.uniform(1.4, self.config.room_size - 1.4)
            if self.distance_to_goal(x, y) < 1.2:
                continue
            if math.hypot(x - self.config.start[0], y - self.config.start[1]) < 1.2:
                continue
            self.obstacles.append(
                {
                    "x": x,
                    "y": y,
                    "axis": float(self.rng.choice([0, 1])),
                    "direction": float(self.rng.choice([-1, 1])),
                }
            )
        return self.state()

    def _move_obstacles(self) -> None:
        for obstacle in self.obstacles:
            axis = int(obstacle["axis"])
            key = "x" if axis == 0 else "y"
            obstacle[key] += obstacle["direction"] * 0.01
            low = 0.7
            high = self.config.room_size - 0.7
            if obstacle[key] < low or obstacle[key] > high:
                obstacle[key] = min(max(obstacle[key], low), high)
                obstacle["direction"] *= -1.0

    def _collides_obstacle(self) -> bool:
        half = self.config.obstacle_width / 2.0
        for obstacle in self.obstacles:
            if abs(self.x - obstacle["x"]) <= half and abs(self.y - obstacle["y"]) <= half:
                return True
        return False

    def _nearest_forward_obstacle(self) -> Tuple[float, float, float]:
        heading = np.array([float(self.vx), float(self.vy)], dtype=float)
        if np.linalg.norm(heading) < 1e-9:
            gx, gy = self.config.goal
            heading = np.array([gx - self.x, gy - self.y], dtype=float)
        heading = heading / max(np.linalg.norm(heading), 1e-9)

        best_distance = self.config.observation_range
        best_dx = 0.0
        best_dy = 0.0
        visible = 0.0
        for obstacle in self.obstacles:
            rel = np.array([obstacle["x"] - self.x, obstacle["y"] - self.y], dtype=float)
            distance = float(np.linalg.norm(rel))
            if distance <= self.config.observation_range and float(np.dot(rel, heading)) > 0:
                if distance < best_distance:
                    best_distance = distance
                    best_dx = rel[0] / self.config.observation_range
                    best_dy = rel[1] / self.config.observation_range
                    visible = 1.0
        return best_dx, best_dy, visible

    def state(self) -> np.ndarray:
        base = super().state()
        dx, dy, visible = self._nearest_forward_obstacle()
        return np.array([base[0], base[1], base[2], base[3], dx, dy, visible], dtype=float)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        next_state, reward, done, info = super().step(action)
        self._move_obstacles()
        hit_obstacle = self._collides_obstacle()
        if hit_obstacle:
            reward += self.config.obstacle_penalty
            collision_distance = self.distance_to_goal()
            self.x, self.y = self.config.start
            self.vx = 0
            self.vy = 0
            reward += self.config.progress_scale * (collision_distance - self.distance_to_goal())
            done = False
        next_state = self.state()
        info["hit_obstacle"] = hit_obstacle
        return next_state, reward, done, info


def obstacle_room_config(seed: int = 31, observation_range: float = 3.0, obstacle_count: int = 7) -> ObstacleRoomConfig:
    return ObstacleRoomConfig(
        hazards=((3.0, 7.2, 6.0, 7.5),),
        seed=seed,
        observation_range=observation_range,
        obstacle_count=obstacle_count,
        goal_radius=0.65,
        goal_reward=45.0,
        progress_scale=4.5,
    )
