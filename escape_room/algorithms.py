from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import DefaultDict, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .envs import ContinuousEscapeRoom, GridEscapeRoom, GridState


Trajectory = List[Dict[str, object]]


def _compact_attempt(
    episode: int,
    states: Sequence[object],
    actions: Sequence[int],
    rewards: Sequence[float],
    *,
    success: bool,
    epsilon: Optional[float] = None,
    label: Optional[str] = None,
    obstacles: Optional[Sequence[Sequence[Dict[str, float]]]] = None,
) -> Dict[str, object]:
    """Store a complete episode without the overhead of one dict per step."""
    state_array = np.asarray(states)
    if np.issubdtype(state_array.dtype, np.integer):
        state_array = state_array.astype(np.int16, copy=False)
    else:
        state_array = state_array.astype(np.float32, copy=False)
    attempt: Dict[str, object] = {
        "episode": episode,
        "label": label or f"Training episode {episode}",
        "states": state_array,
        "actions": np.asarray(actions, dtype=np.int8),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "reward": float(np.sum(rewards, dtype=np.float64)),
        "steps": max(0, len(states) - 1),
        "success": bool(success),
    }
    if epsilon is not None:
        attempt["epsilon"] = float(epsilon)
    if obstacles and any(obstacles):
        attempt["obstacles"] = np.asarray(
            [
                [[item["x"], item["y"], item["axis"], item["direction"]] for item in frame]
                for frame in obstacles
            ],
            dtype=np.float32,
        )
    return attempt


def moving_average(values: Sequence[float], window: int = 25) -> List[float]:
    if not values:
        return []
    window = max(1, window)
    result = []
    total = 0.0
    queue: List[float] = []
    for value in values:
        queue.append(float(value))
        total += float(value)
        if len(queue) > window:
            total -= queue.pop(0)
        result.append(total / len(queue))
    return result


def _argmax_random_tie(values: Sequence[float], rng: random.Random) -> int:
    best = max(values)
    candidates = [index for index, value in enumerate(values) if abs(value - best) < 1e-12]
    return rng.choice(candidates)


def epsilon_greedy(q_values: Sequence[float], epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(q_values))
    return _argmax_random_tie(q_values, rng)


def value_iteration(
    env: GridEscapeRoom,
    gamma: float = 0.96,
    theta: float = 1e-4,
    max_iterations: int = 1000,
) -> Dict[str, object]:
    states = env.all_states()
    values: Dict[GridState, float] = {state: 0.0 for state in states}
    policy: Dict[GridState, int] = {}
    q_table: Dict[GridState, np.ndarray] = {}
    deltas: List[float] = []

    for iteration in range(1, max_iterations + 1):
        delta = 0.0
        for state in states:
            if env.is_terminal_state(state):
                continue
            action_values = []
            for action in env.available_actions(state):
                q_value = sum(
                    probability * (reward + gamma * values[next_state] * (not done))
                    for probability, next_state, reward, done in env.transition_model(state, action)
                )
                action_values.append(q_value)
            best_value = max(action_values)
            delta = max(delta, abs(values[state] - best_value))
            values[state] = best_value
        deltas.append(delta)
        if delta < theta:
            break

    rng = random.Random(env.config.seed + 1000)
    for state in states:
        action_values = []
        for action in env.available_actions(state):
            q_value = sum(
                probability * (reward + gamma * values[next_state] * (not done))
                for probability, next_state, reward, done in env.transition_model(state, action)
            )
            action_values.append(q_value)
        q_table[state] = np.array(action_values, dtype=float)
        policy[state] = _argmax_random_tie(action_values, rng)

    metrics = [
        {"iteration": index + 1, "delta": delta, "value_start": values[env.reset()]}
        for index, delta in enumerate(deltas)
    ]
    attempts: List[Dict[str, object]] = []
    for attempt_number in range(1, 13):
        attempt_trajectory = run_grid_policy(
            env.copy_with(seed=env.config.seed + 76 + attempt_number),
            policy=policy,
            max_steps=220,
            seed=env.config.seed + attempt_number,
        )
        attempts.append(
            _compact_attempt(
                attempt_number,
                [step["state"] for step in attempt_trajectory],
                [-1 if step["action"] is None else int(step["action"]) for step in attempt_trajectory],
                [float(step["reward"]) for step in attempt_trajectory],
                success=bool(attempt_trajectory and attempt_trajectory[-1]["done"]),
                label=f"Policy rollout {attempt_number}",
            )
        )
    trajectory = run_grid_policy(env.copy_with(seed=env.config.seed + 77), policy=policy, max_steps=220)
    return {
        "values": values,
        "policy": policy,
        "q_table": q_table,
        "metrics": metrics,
        "snapshots": [
            {
                "episode": len(deltas),
                "label": "Policy after Value Iteration",
                "trajectory": trajectory,
                "reward": sum(float(step["reward"]) for step in trajectory),
                "success": bool(trajectory and trajectory[-1]["done"]),
            }
        ],
        "attempts": attempts,
    }


def run_grid_policy(
    env: GridEscapeRoom,
    *,
    policy: Optional[Dict[GridState, int]] = None,
    q_table: Optional[MutableMapping[GridState, np.ndarray]] = None,
    max_steps: int = 250,
    epsilon: float = 0.0,
    seed: int = 0,
) -> Trajectory:
    rng = random.Random(seed)
    state = env.reset()
    trajectory: Trajectory = [{"state": state, "action": None, "reward": 0.0, "done": False}]
    for _ in range(max_steps):
        if policy is not None:
            action = policy.get(state, 0)
        elif q_table is not None:
            action = epsilon_greedy(q_table[state], epsilon, rng)
        else:
            action = rng.randrange(env.action_count)
        next_state, reward, done, info = env.step(action)
        trajectory.append(
            {
                "state": next_state,
                "action": action,
                "reward": reward,
                "done": done,
                "info": info,
            }
        )
        state = next_state
        if done:
            break
    return trajectory


def _snapshot_episodes(episodes: int) -> set[int]:
    if episodes <= 1:
        return {1}
    anchors = {1, episodes}
    for ratio in (0.1, 0.25, 0.5, 0.75):
        anchors.add(max(1, min(episodes, int(round(episodes * ratio)))))
    return anchors


def train_sarsa(
    env: GridEscapeRoom,
    *,
    episodes: int = 600,
    max_steps: int = 250,
    alpha: float = 0.15,
    gamma: float = 0.96,
    epsilon: float = 0.35,
    epsilon_min: float = 0.03,
    epsilon_decay: float = 0.992,
    seed: int = 3,
) -> Dict[str, object]:
    rng = random.Random(seed)
    q_table: DefaultDict[GridState, np.ndarray] = defaultdict(lambda: np.zeros(env.action_count, dtype=float))
    metrics: List[Dict[str, object]] = []
    snapshots: List[Dict[str, object]] = []
    attempts: List[Dict[str, object]] = []
    snapshot_points = _snapshot_episodes(episodes)

    for episode in range(1, episodes + 1):
        state = env.reset()
        action = epsilon_greedy(q_table[state], epsilon, rng)
        total_reward = 0.0
        done = False
        attempt_states: List[object] = [state]
        attempt_actions = [-1]
        attempt_rewards = [0.0]

        for step in range(1, max_steps + 1):
            next_state, reward, done, _info = env.step(action)
            attempt_states.append(next_state)
            attempt_actions.append(action)
            attempt_rewards.append(reward)
            next_action = epsilon_greedy(q_table[next_state], epsilon, rng)
            target = reward + gamma * q_table[next_state][next_action] * (not done)
            q_table[state][action] += alpha * (target - q_table[state][action])
            state, action = next_state, next_action
            total_reward += reward
            if done:
                break

        metrics.append(
            {
                "episode": episode,
                "reward": total_reward,
                "steps": step,
                "success": done,
                "epsilon": epsilon,
            }
        )
        attempts.append(
            _compact_attempt(
                episode,
                attempt_states,
                attempt_actions,
                attempt_rewards,
                success=done,
                epsilon=epsilon,
            )
        )
        if episode in snapshot_points:
            eval_env = env.copy_with(seed=seed + episode + 5000)
            trajectory = run_grid_policy(eval_env, q_table=q_table, max_steps=max_steps, seed=seed + episode)
            snapshots.append(
                {
                    "episode": episode,
                    "label": f"Learned policy after episode {episode}",
                    "trajectory": trajectory,
                    "reward": sum(float(item["reward"]) for item in trajectory),
                    "success": bool(trajectory and trajectory[-1]["done"]),
                }
            )
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return {"q_table": q_table, "metrics": metrics, "snapshots": snapshots, "attempts": attempts}


def train_q_learning(
    env: GridEscapeRoom,
    *,
    episodes: int = 800,
    max_steps: int = 280,
    alpha: float = 0.18,
    gamma: float = 0.97,
    epsilon: float = 0.4,
    epsilon_min: float = 0.03,
    epsilon_decay: float = 0.994,
    seed: int = 5,
) -> Dict[str, object]:
    rng = random.Random(seed)
    q_table: DefaultDict[GridState, np.ndarray] = defaultdict(lambda: np.zeros(env.action_count, dtype=float))
    metrics: List[Dict[str, object]] = []
    snapshots: List[Dict[str, object]] = []
    attempts: List[Dict[str, object]] = []
    snapshot_points = _snapshot_episodes(episodes)

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        attempt_states: List[object] = [state]
        attempt_actions = [-1]
        attempt_rewards = [0.0]

        for step in range(1, max_steps + 1):
            action = epsilon_greedy(q_table[state], epsilon, rng)
            next_state, reward, done, _info = env.step(action)
            attempt_states.append(next_state)
            attempt_actions.append(action)
            attempt_rewards.append(reward)
            target = reward + gamma * float(np.max(q_table[next_state])) * (not done)
            q_table[state][action] += alpha * (target - q_table[state][action])
            state = next_state
            total_reward += reward
            if done:
                break

        metrics.append(
            {
                "episode": episode,
                "reward": total_reward,
                "steps": step,
                "success": done,
                "epsilon": epsilon,
            }
        )
        attempts.append(
            _compact_attempt(
                episode,
                attempt_states,
                attempt_actions,
                attempt_rewards,
                success=done,
                epsilon=epsilon,
            )
        )
        if episode in snapshot_points:
            eval_env = env.copy_with(seed=seed + episode + 6000)
            trajectory = run_grid_policy(eval_env, q_table=q_table, max_steps=max_steps, seed=seed + episode)
            snapshots.append(
                {
                    "episode": episode,
                    "label": f"Learned policy after episode {episode}",
                    "trajectory": trajectory,
                    "reward": sum(float(item["reward"]) for item in trajectory),
                    "success": bool(trajectory and trajectory[-1]["done"]),
                }
            )
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return {"q_table": q_table, "metrics": metrics, "snapshots": snapshots, "attempts": attempts}


@dataclass
class LinearApproxQ:
    action_count: int
    room_size: float
    goal: Tuple[float, float]
    tile_count: int = 10
    tilings: int = 6

    def __post_init__(self) -> None:
        self.weights: List[DefaultDict[str, float]] = [
            defaultdict(float) for _ in range(self.action_count)
        ]

    def features(self, state: np.ndarray) -> Dict[str, float]:
        x = float(state[0])
        y = float(state[1])
        vx = int(round(float(state[2])))
        vy = int(round(float(state[3])))
        xn = min(max(x / self.room_size, 0.0), 0.999999)
        yn = min(max(y / self.room_size, 0.0), 0.999999)
        gx, gy = self.goal
        dx = (gx - x) / self.room_size
        dy = (gy - y) / self.room_size
        dist = min(1.5, float(np.hypot(dx, dy)))

        features: Dict[str, float] = {
            "bias": 1.0,
            "x": xn,
            "y": yn,
            "goal_dx": dx,
            "goal_dy": dy,
            "goal_distance": dist,
            f"vx:{vx}": 1.0,
            f"vy:{vy}": 1.0,
        }
        for tiling in range(self.tilings):
            offset = tiling / (self.tilings * self.tile_count)
            ix = min(self.tile_count - 1, int((xn + offset) * self.tile_count))
            iy = min(self.tile_count - 1, int((yn + offset) * self.tile_count))
            features[f"tile:{tiling}:{ix}:{iy}:v{vx},{vy}"] = 1.0

        for index in range(4, len(state)):
            value = float(state[index])
            features[f"obs:{index}"] = value
            bucket = int(np.clip(np.floor((value + 1.0) * 4), 0, 7))
            features[f"obs_bucket:{index}:{bucket}"] = 1.0
        return features

    def q_value(self, state: np.ndarray, action: int) -> float:
        features = self.features(state)
        return sum(self.weights[action][name] * value for name, value in features.items())

    def q_values(self, state: np.ndarray) -> np.ndarray:
        features = self.features(state)
        return np.array(
            [sum(weights[name] * value for name, value in features.items()) for weights in self.weights],
            dtype=float,
        )

    def update(self, state: np.ndarray, action: int, target: float, alpha: float) -> float:
        features = self.features(state)
        prediction = sum(self.weights[action][name] * value for name, value in features.items())
        td_error = target - prediction
        scale = max(1.0, sum(abs(value) for value in features.values()))
        for name, value in features.items():
            self.weights[action][name] += alpha * td_error * value / scale
        return td_error


def run_continuous_policy(
    env: ContinuousEscapeRoom,
    agent: LinearApproxQ,
    *,
    max_steps: int = 800,
    epsilon: float = 0.0,
    seed: int = 0,
) -> Trajectory:
    rng = random.Random(seed)
    state = env.reset()
    trajectory: Trajectory = [
        {
            "state": state.copy(),
            "action": None,
            "reward": 0.0,
            "done": False,
            "obstacles": _obstacle_snapshot(env),
        }
    ]
    for _ in range(max_steps):
        action = epsilon_greedy(agent.q_values(state), epsilon, rng)
        next_state, reward, done, info = env.step(action)
        trajectory.append(
            {
                "state": next_state.copy(),
                "action": action,
                "reward": reward,
                "done": done,
                "info": info,
                "obstacles": _obstacle_snapshot(env),
            }
        )
        state = next_state
        if done:
            break
    return trajectory


def _obstacle_snapshot(env: ContinuousEscapeRoom) -> List[Dict[str, float]]:
    obstacles = getattr(env, "obstacles", None)
    if not obstacles:
        return []
    return [
        {
            "x": float(obstacle["x"]),
            "y": float(obstacle["y"]),
            "axis": float(obstacle["axis"]),
            "direction": float(obstacle["direction"]),
        }
        for obstacle in obstacles
    ]


def train_approx_q_learning(
    env: ContinuousEscapeRoom,
    *,
    episodes: int = 450,
    max_steps: int = 850,
    alpha: float = 0.08,
    gamma: float = 0.985,
    epsilon: float = 0.45,
    epsilon_min: float = 0.04,
    epsilon_decay: float = 0.993,
    seed: int = 13,
) -> Dict[str, object]:
    rng = random.Random(seed)
    agent = LinearApproxQ(
        action_count=env.action_count,
        room_size=env.config.room_size,
        goal=env.config.goal,
    )
    metrics: List[Dict[str, object]] = []
    snapshots: List[Dict[str, object]] = []
    attempts: List[Dict[str, object]] = []
    snapshot_points = _snapshot_episodes(episodes)

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        td_errors: List[float] = []
        attempt_states: List[object] = [state.copy()]
        attempt_actions = [-1]
        attempt_rewards = [0.0]
        attempt_obstacles = [_obstacle_snapshot(env)]

        for step in range(1, max_steps + 1):
            action = epsilon_greedy(agent.q_values(state), epsilon, rng)
            next_state, reward, done, _info = env.step(action)
            attempt_states.append(next_state.copy())
            attempt_actions.append(action)
            attempt_rewards.append(reward)
            attempt_obstacles.append(_obstacle_snapshot(env))
            target = reward + gamma * float(np.max(agent.q_values(next_state))) * (not done)
            td_errors.append(agent.update(state, action, target, alpha))
            state = next_state
            total_reward += reward
            if done:
                break

        metrics.append(
            {
                "episode": episode,
                "reward": total_reward,
                "steps": step,
                "success": done,
                "epsilon": epsilon,
                "mean_abs_td_error": float(np.mean(np.abs(td_errors))) if td_errors else 0.0,
            }
        )
        attempts.append(
            _compact_attempt(
                episode,
                attempt_states,
                attempt_actions,
                attempt_rewards,
                success=done,
                epsilon=epsilon,
                obstacles=attempt_obstacles,
            )
        )
        if episode in snapshot_points:
            eval_env = env.copy_with(seed=seed + episode + 7000)
            trajectory = run_continuous_policy(eval_env, agent, max_steps=max_steps, seed=seed + episode)
            snapshots.append(
                {
                    "episode": episode,
                    "label": f"Learned policy after episode {episode}",
                    "trajectory": trajectory,
                    "reward": sum(float(item["reward"]) for item in trajectory),
                    "success": bool(trajectory and trajectory[-1]["done"]),
                }
            )
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return {"agent": agent, "metrics": metrics, "snapshots": snapshots, "attempts": attempts}
