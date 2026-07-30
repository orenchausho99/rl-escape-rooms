# RL Escape Rooms

This is my final reinforcement-learning project. I built an escape-room campaign with five different games. Each room has a different task, state, reward function, and learning algorithm. The rooms become harder as the campaign continues, and the agent gets a better score when it reaches the final state in fewer steps.

- **Live app:** [rl-escape-rooms.streamlit.app](https://rl-escape-rooms.streamlit.app/)
- **GitHub:** [github.com/orenchausho99/rl-escape-rooms](https://github.com/orenchausho99/rl-escape-rooms)

The project is written in Python. I used Streamlit for the interface and HTML Canvas for the games.

## Main features

- Five different rooms and four reinforcement-learning methods.
- `10x10` grids in the first three rooms.
- A continuous `10x10` meter environment in Rooms 4 and 5.
- Controls for the training parameters and number of episodes.
- Hyperparameter comparison before the full training run.
- Analytics for reward, steps, success, exploration, and convergence.
- Replay of every training episode.
- Keyboard controls with the arrow keys or `WASD`.

The application flow is:

```text
Player Profile -> Campaign Dashboard -> Room
Play Game | Train Agent | Episode Replay | Analytics | Room Specs
```

## Room 1 - Pac-Man Ice Maze

The agent starts at `(0,0)` and must reach `EXIT` at `(9,9)`. It has to avoid cracks and moving ghosts and handle slippery tiles.

- **Algorithm:** Dynamic Programming with Value Iteration.
- **Model:** Known. The algorithm can use all transition probabilities and rewards.
- **State:** `(row, col, collected_mask, ghost_phase)`.
- **Actions:** `UP`, `RIGHT`, `DOWN`, `LEFT`.
- **Final state:** The agent reaches `EXIT` with `ghost_phase=0`.

On a slippery tile, the requested direction can change to the left or right. `ghost_phase` is included because the ghost positions affect the next transition.

| Event | Reward |
|---|---:|
| Every step | `-1` |
| Reach EXIT | `+110` |
| Crack | `-22` or `-25` |
| Ghost collision | `-45` and return to start |

Best values I found:

```text
gamma=0.96, theta=0.0001, slip=0.25
max_iterations=1000, max_steps=220
```

All `12/12` policy rollouts succeeded in my verification.

## Room 2 - Sokoban Vault

The agent must push the box from `(0,8)` to `TARGET` at `(0,9)`. It then has to avoid two moving laser patrols and reach `SAFE` at `(9,9)`.

- **Algorithm:** SARSA.
- **Model:** Unknown. The agent learns only from sampled actions and rewards.
- **State:** `(player_row, player_col, box_row, box_col, laser_phase)`.
- **Actions:** `UP`, `RIGHT`, `DOWN`, `LEFT`.
- **Final state:** The box is on `TARGET`, the player is in `SAFE`, and `laser_phase=0`.

The box position is part of the real environment state. `laser_phase` stores the current position of the moving lasers. This lets SARSA learn when it is safe to cross their paths. A laser hit returns the player to the start but does not reset the box.

| Event | Reward |
|---|---:|
| Every step | `-1` |
| Box reaches TARGET | `+28` |
| Box leaves TARGET | `-28` |
| Reach SAFE after solving the box | `+130` |
| Invalid move or push | `-6` additional penalty |
| Moving laser hit | `-38` and return to start |

Best values I found:

```text
episodes=650, max_steps=250, alpha=0.15, gamma=0.96
epsilon=0.40, epsilon_min=0.03, epsilon_decay=0.993, slip=0.18
```

The last `50/50` episodes succeeded, with about `23.74` steps on average.

## Room 3 - Bomberman Reactor

The agent must collect two `CORE` items, avoid bombs and patrol bots, use the `WARP` tunnel when useful, and reach `GATE`.

- **Algorithm:** Q-Learning.
- **Model:** Unknown.
- **State:** `(row, col, core_mask, guard_phase)`.
- **Actions:** `UP`, `RIGHT`, `DOWN`, `LEFT`.
- **Final state:** Both cores are collected and the agent reaches `GATE`.

`core_mask` stores the collected items, and `guard_phase` stores the positions of the moving guards.

| Event | Reward |
|---|---:|
| Every step | `-1.2` |
| Collect a CORE | `+30` |
| Reach GATE with both COREs | `+170` |
| Bomb | `-26`, `-30`, or `-38` |
| Patrol collision | `-60` and return to start |
| Use WARP | `0` |

The WARP reward is zero so the agent cannot collect unlimited rewards from a portal loop.

Best values I found:

```text
episodes=650, max_steps=850, alpha=0.15, gamma=0.96
epsilon=0.40, epsilon_min=0.03, epsilon_decay=0.993, slip=0.18
```

The last `50/50` episodes succeeded in my verification.

## Room 4 - Lunar Lander Pad

This room is not a grid. The agent moves through a continuous `10x10` meter area, avoids asteroid fields, and reaches the landing pad.

- **Algorithm:** Linear Approximate Q-Learning.
- **Model:** Unknown.
- **State:** `(X, Y, Vx, Vy)`.
- **Actions:** Nine velocity pairs where `Vx,Vy` are in `{-1,0,1}`.
- **Time step:** `0.02` seconds.
- **Final state:** The agent reaches `PAD`, and the final velocity is `(0,0)`.

A regular Q-table is not practical because `X` and `Y` are continuous. I use normalized position, goal distance, velocity, and tile-coding features to approximate the Q-value.

| Event | Reward |
|---|---:|
| Every time step | `-0.015` |
| Progress toward PAD | `5 * distance improvement` |
| Reach PAD | `+55` |
| Wall collision | `-0.6` |
| Asteroid collision | `-2.5` and velocity reset |

Best values I found:

```text
episodes=450, max_steps=850, alpha=0.08, gamma=0.985
epsilon=0.40, epsilon_min=0.03, epsilon_decay=0.993
```

In my verification, `37/50` of the final episodes succeeded with all five asteroid fields enabled.

## Room 5 - Portal Hazard Run

This is the optional and hardest room. The agent must reach `EXIT` while moving portal hazards cross the room. The agent can only observe the nearest obstacle in front of it within a selected distance.

- **Algorithm:** Approximate Q-Learning.
- **State:** `(X, Y, Vx, Vy, obstacle_dx, obstacle_dy, visible)`.
- **Obstacle width:** `0.5` meters.
- **Default obstacle count:** `7`, configurable from 2 to 15.
- **Observation range:** Configurable from 1 to 6 meters.
- **Final state:** The agent reaches `EXIT`.

Portal contact gives a penalty and teleports the agent to a random safe position. After training, I can test the learned policy in a new random room with a different seed.

| Event | Reward |
|---|---:|
| Every time step | `-0.015` |
| Progress toward EXIT | `4.5 * distance improvement` |
| Reach EXIT | `+45` |
| Moving portal collision | `-8`, random teleport, and velocity reset |
| Static hazard collision | `-2.5` |
| Wall collision | `-0.6` |

Best values I found:

```text
episodes=450, max_steps=1400, alpha=0.08, gamma=0.985
epsilon=0.40, epsilon_min=0.03, epsilon_decay=0.993
obstacle_count=7, observation_range=3.0
```

The last `49/50` episodes succeeded in my verification.

## Training and optimization

The important parameters are:

- `episodes`: number of complete training attempts.
- `max_steps`: maximum actions in one episode.
- `alpha`: learning rate.
- `gamma`: importance of future rewards.
- `epsilon`: probability of a random exploratory action.
- `epsilon_decay`: how exploration decreases during training.
- `seed`: makes experiments reproducible.
- `theta`: convergence threshold for Value Iteration.

`Start training` uses the selected values. `Optimize and train` compares four candidate configurations and then trains the best one for the requested number of episodes. I treat these as the best experimental values I found, not a guaranteed mathematical global optimum.

## Replay and analytics

The application stores every episode from SARSA, Q-Learning, and Approximate Q-Learning. Value Iteration stores 12 policy rollouts because it uses model sweeps instead of training episodes.

The Replay tab can show successful or failed episodes, play or pause the agent, move one step at a time, and change the speed.

The Analytics tab shows:

- Reward and reward moving average over 25 episodes.
- Steps and recent success rate.
- Epsilon during training.
- TD error for function approximation.
- Convergence delta for Value Iteration.

Completed runs save CSV, JSON, and PNG reports under `runs/`.

## Run locally

Python 3.11 is recommended.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

Run the automated tests with:

```bash
python -m unittest discover -s tests -v
```

## Publishing with Streamlit Community Cloud

Live application: [RL Escape Rooms](https://rl-escape-rooms.streamlit.app/)

1. Push the project to a public GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select the repository and the `main` branch.
4. Set the main file path to `app.py`.
5. Select Python 3.11 and deploy the application.

The project does not require API keys or other secrets.

## Main files

```text
app.py                     Streamlit interface, games, training, replay, and analytics
escape_room/envs.py        Environment states, transitions, and rewards
escape_room/algorithms.py  Value Iteration, SARSA, Q-Learning, and Approximate Q-Learning
escape_room/replay.py      Episode replay data and filters
static/game_art/           Backgrounds, thumbnails, and banners
tests/                     Automated project tests
```

## What I learned

This project helped me understand the difference between planning with a known model, on-policy and off-policy learning, and function approximation. I also learned that the state representation, reward design, exploration schedule, and hyperparameters can be just as important as the algorithm itself.
