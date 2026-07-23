# Reinforcement Learning Escape Rooms

A Python and Streamlit final project for experimenting with reinforcement-learning algorithms, reward design, exploration, replay, and hyperparameter optimization.

The app contains a five-room escape campaign. Each room has a different task and visual game style, but the manual game, training environment, replay, state definition, terminal condition, and reward function share the same rules.

## Requirement Coverage

| Requirement | Implementation |
|---|---|
| At least four rooms | Five rooms are included |
| First three rooms are 10x10 grids | `GridEscapeRoom` and `SokobanEscapeRoom` |
| Known model | Room 1, Value Iteration |
| Unknown model | Room 2 SARSA and Room 3 Q-Learning use only `env.step()` |
| Function approximation | Rooms 4 and 5 use linear Approximate Q-Learning with tile coding |
| Continuous 10x10 room | Rooms 4 and 5 |
| State `X,Y,Vx,Vy` | Room 4 uses exactly these four values |
| Time step 0.02 seconds | `ContinuousRoomConfig.dt = 0.02` |
| Discrete velocity | Nine `(Vx,Vy)` combinations from `{-1,0,1}` |
| Dynamic 0.5m obstacles | Room 5 |
| Configurable forward observation | Room 5, measured center to center |
| Training graphs | Reward, moving average, steps, success, epsilon, and TD error |
| Individual episode replay | Every training attempt can be selected and replayed |
| Increasing difficulty | Ice maze, Sokoban, items and guards, continuous control, partial observation |
| Hyperparameter optimization | `Optimize and train` compares candidates and trains the best one |
| Saved experiment output | CSV, JSON, and PNG reports under `runs/` |

## Run Locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`.

Run the automated requirement tests:

```bash
python -m unittest discover -s tests -v
```

## Application Workflow

1. Select a room.
2. Use `Play` to understand its rules.
3. Use `Train` to change parameters and train the matching agent.
4. A room is completed when recent training success reaches at least 60%.
5. Continue directly to the next room.
6. Use `Replay` to inspect any training episode.
7. Use `Analytics` to inspect learning and exploration.

The fifth room also supports generating an unseen random room and testing the learned policy without additional training.

## Manual Game Controls

- Click `Start Mission`, then use the keyboard directly inside the arena.
- Grid rooms: press the arrow keys or `W`, `A`, `S`, `D` once per tile.
- Continuous rooms: hold the arrow keys or `W`, `A`, `S`, `D` to set the discrete velocity.
- On mobile, the same actions are available through large touch controls below the arena.
- The live HUD shows keyboard focus, mission progress, score, steps, goal, and current status.

## Project Structure

```text
app.py                         Streamlit UI, Canvas games, tuning, replay, analytics
escape_room/envs.py            Environment states, transitions, rewards, obstacles
escape_room/algorithms.py      Value Iteration, SARSA, Q-Learning, Approximate Q
static/game_art/               Local optimized WebP arena artwork
tests/test_requirements.py     Automated rubric checks
.streamlit/config.toml         Theme and static-file serving
requirements.txt               Python dependencies
runtime.txt                    Streamlit Cloud Python version
```

## Room 1: Pac-Man Ice Maze

Algorithm: Dynamic Programming using Value Iteration.

Model: known. The algorithm reads the complete stochastic transition model, including slippery outcomes and deterministic ghost phases.

State:

```text
(row, col, collected_mask, ghost_phase)
```

`collected_mask` is always zero in this room. `ghost_phase` makes moving enemies Markovian.

Actions: `UP`, `RIGHT`, `DOWN`, `LEFT`.

Terminal condition: reach the single `EXIT` location `(9,9)`.

Rewards:

| Event | Reward |
|---|---:|
| Every step | -1 |
| Reach EXIT | +110 |
| Crack traps | -22 or -25 |
| Ghost collision | -45 and return to start |
| Enter blocked cell | The step penalty remains active |

Slippery cells redirect the intended action to the left or right with the configured probability. A negative reward per step means shorter solutions receive more return.

Verified parameters: `gamma=0.96`, `theta=0.0001`, `slip=0.25`, `max_iterations=1000`.

Verification result: 12/12 stochastic policy rollouts succeeded, averaging 18 steps.

## Room 2: Sokoban Vault

Algorithm: SARSA.

Model: unknown to the learner. SARSA receives only sampled transitions from `step()`.

State:

```text
(player_row, player_col, box_row, box_col)
```

The box position is part of the actual Q-table state. This is a real crate-pushing environment, not a visual replacement for key collection.

Actions: `UP`, `RIGHT`, `DOWN`, `LEFT`.

Terminal condition: the box must be on `TARGET`, then the player must enter `SAFE`.

Rewards:

| Event | Reward |
|---|---:|
| Every step | -1 |
| Box reaches TARGET | +28 |
| Box leaves TARGET | -28 |
| Reach SAFE after solving | +130 |
| Invalid push or blocked move | -6 additional |
| Laser tiles | -30 or -35 |

Verified parameters: `episodes=650`, `max_steps=250`, `alpha=0.15`, `gamma=0.96`, `epsilon=0.40`, `epsilon_min=0.03`, `epsilon_decay=0.993`, `slip=0.18`.

Verification result: 50/50 of the final episodes succeeded, averaging 18.96 steps.

Why SARSA: the update uses the next action selected by the current exploratory policy. It therefore learns the value of the behavior it actually follows.

## Room 3: Bomberman Reactor

Algorithm: Q-Learning.

Model: unknown to the learner.

State:

```text
(row, col, core_mask, guard_phase)
```

Actions: `UP`, `RIGHT`, `DOWN`, `LEFT`.

Terminal condition: collect both `CORE` items and enter `GATE`.

Rewards:

| Event | Reward |
|---|---:|
| Every step | -1.2 |
| Each CORE, once | +30 |
| Reach GATE after both cores | +170 |
| Bomb traps | -26, -30, or -38 |
| Patrol collision | -60 and return to start |
| WARP tunnel | 0, preventing reward loops |

Verified parameters: `episodes=650`, `max_steps=850`, `alpha=0.15`, `gamma=0.96`, `epsilon=0.40`, `epsilon_min=0.03`, `epsilon_decay=0.993`, `slip=0.18`.

Verification result: 50/50 of the final episodes succeeded, averaging 71.68 steps.

Why Q-Learning: it is off-policy. Exploration selects behavior actions, while the target uses the maximum estimated next-state value.

## Room 4: Lunar Lander Pad

Algorithm: linear Approximate Q-Learning with multiple offset tile codings.

State:

```text
X, Y, Vx, Vy
```

Room size: 10x10 meters.

Time step: `0.02` seconds.

Actions:

```text
Vx, Vy in {-1, 0, 1}
```

There are nine discrete actions. Position remains continuous.

Terminal condition: enter the radius around `PAD`.

Rewards:

| Event | Reward |
|---|---:|
| Every time step | -0.015 |
| Progress | `5 * (old_distance - new_distance)` |
| Reach PAD | +55 |
| Wall collision | -0.6 |

Verified parameters: `episodes=450`, `max_steps=850`, `alpha=0.08`, `gamma=0.985`, `epsilon=0.40`, `epsilon_min=0.03`, `epsilon_decay=0.993`.

Verification result: 45/50 of the final episodes succeeded, averaging 551.38 time steps.

Why approximation is required: continuous positions create too many states for a tabular Q-table. Features include normalized position, goal direction, distance, discrete velocity, and six offset tile codings.

## Room 5: Portal Hazard Run

Algorithm: Approximate Q-Learning with partial local observation.

Moving portal fields are the dynamic obstacles. Touching one produces a penalty and teleports the player back to the start.

State:

```text
X, Y, Vx, Vy, obstacle_dx, obstacle_dy, visible
```

`obstacle_dx` and `obstacle_dy` describe the nearest visible obstacle in front of the agent, normalized by the observation range. `visible` is zero or one.

Obstacle properties:

| Property | Value |
|---|---|
| Width | 0.5 meters |
| Default count | 7, configurable from 2 to 15 |
| Default observation | 3 meters, configurable from 1 to 6 |
| Position | Randomized at every reset |
| Motion | Horizontal or vertical, direction chosen randomly |

Rewards:

| Event | Reward |
|---|---:|
| Every time step | -0.015 |
| Progress | `4.5 * (old_distance - new_distance)` |
| Reach EXIT | +45 |
| Dynamic obstacle collision | -8 and return to start |
| Static hazard | -2.5 |
| Wall collision | -0.6 |

Verified parameters: `episodes=450`, `max_steps=1400`, `alpha=0.08`, `gamma=0.985`, `epsilon=0.40`, `epsilon_min=0.03`, `epsilon_decay=0.993`, `obstacle_count=7`, `observation_range=3.0`.

Verification result: 35/50 of the final episodes succeeded, averaging 1081.9 steps. The best failed episode scored 31.99, while successful episodes reached more than 90, confirming that collisions cannot be used to farm more reward than completing the room. The room remains intentionally harder than the four mandatory rooms.

## Hyperparameter Optimization

Every room has two training commands:

- `Start training` uses the visible values.
- `Optimize and train` evaluates four candidates, ranks them by success rate, reward, and solution length, then performs a full run with the best candidate.

Optimization tables are saved under `runs/tuning/`. Use the same seed when comparing individual settings so that the environment randomness is controlled.

The documented values are the best verified defaults used by the included benchmark. They are empirical settings, not a claim of a mathematical global optimum.

## Replay and Saved Results

SARSA, Q-Learning, and Approximate Q-Learning record every exploratory episode. Value Iteration records 12 stochastic policy rollouts because it converges by sweeps rather than training episodes.

Replay supports:

- A complete table containing every recorded training episode.
- Direct episode selection plus previous/next episode navigation.
- Successful or failed filtering.
- Sorting by episode or reward.
- Play, pause, single-step navigation, timeline scrubbing, and playback speed.
- Exact player, box, core, guard, velocity, and obstacle states from training.

Every completed training run automatically creates:

```text
runs/<room_timestamp>/metrics.csv
runs/<room_timestamp>/attempts.csv
runs/<room_timestamp>/parameters.json
runs/<room_timestamp>/summary.json
runs/<room_timestamp>/learning_report.png
```

## GitHub and Streamlit Community Cloud

Create the repository:

```bash
git init
git add .
git commit -m "RL escape rooms final project"
git branch -M main
git remote add origin https://github.com/YOUR_USER/YOUR_REPOSITORY.git
git push -u origin main
```

Deploy:

1. Open Streamlit Community Cloud.
2. Select `Create app`.
3. Select the GitHub repository and branch `main`.
4. Set the main file to `app.py`.
5. Deploy and add the public URL to this README.

No secret keys are required.
