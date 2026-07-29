# RL Escape Rooms

## About the project

For my final project, I built an escape-room game that lets me train and compare different reinforcement-learning algorithms.

The project has five rooms. Each room has a different game, state representation, reward function, and learning algorithm. The rooms become harder as the campaign continues. The goal in every room is to reach one final state, and the agent receives a better total reward when it solves the room in fewer steps.

The first three rooms use a `10x10` grid. The fourth room uses continuous position and discrete velocity. The fifth room is an extra challenge with moving obstacles and limited observation.

The application is written in Python and uses Streamlit for the interface. The games themselves are displayed with an HTML Canvas component inside Streamlit.

Live application: [https://rl-escape-rooms.streamlit.app/](https://rl-escape-rooms.streamlit.app/)

GitHub repository: [https://github.com/orenchausho99/rl-escape-rooms](https://github.com/orenchausho99/rl-escape-rooms)

## What I wanted to test

The main purpose of the project was to experiment with:

- Dynamic Programming when the environment model is known.
- SARSA and Q-Learning when the model is unknown.
- Approximate Q-Learning when the state space is continuous.
- Exploration with an epsilon-greedy policy.
- Reward design and the effect of step penalties.
- Hyperparameter tuning and reproducible experiments.
- Saving learning results and replaying individual episodes.
- Generalization to a new random room.

## How to use the application

1. Create a Player Profile or select `Continue as guest`.
2. Use the Campaign Dashboard to continue the current campaign or choose one of the five rooms.
3. Open `Play Game` to understand the room and play it with the keyboard.
4. Open `Train Agent` and choose the training parameters.
5. Start a normal training run or use `Optimize and train`.
6. Open `Episode Replay` to watch any recorded episode.
7. Open `Analytics` to see the learning graphs.
8. Open `Room Specs` to see the state, actions, terminal condition, and rewards.

The Player Profile is a session profile, not a password-based authentication system. It personalizes the dashboard without changing any environment, reward, state, or learning algorithm. Guest mode keeps the application immediately available during demonstrations.

A room is marked as completed when the trained agent reaches at least a 60% success rate in its recent attempts. For the first room, all 12 policy rollouts are used.

## Keyboard controls

- Grid rooms: arrow keys or `W`, `A`, `S`, `D` move one tile at a time.
- Continuous rooms: hold the arrow keys or `W`, `A`, `S`, `D` to choose the velocity.
- On a phone or tablet, touch controls are shown below the game.
- In the Pac-Man room, the ghosts keep moving and chasing the player even when the player is standing still.

## Project requirements

| Requirement | How it is implemented |
|---|---|
| At least four different rooms | The project contains five rooms |
| First three rooms are `10x10` grids | Rooms 1, 2, and 3 use grid environments |
| Known environment model | Room 1 uses Value Iteration |
| Unknown environment model | Rooms 2 and 3 learn only from sampled `step()` results |
| SARSA | Used in Room 2 |
| Q-Learning | Used in Room 3 |
| Function approximation | Used in Rooms 4 and 5 |
| Continuous `10x10` meter room | Used in Rooms 4 and 5 |
| State `X,Y,Vx,Vy` | Room 4 uses exactly these four values |
| Time step of `0.02` seconds | Defined in `ContinuousRoomConfig` |
| Discrete velocity values | Nine combinations from `Vx,Vy in {-1,0,1}` |
| Optional dynamic obstacle room | Implemented as Room 5 |
| Obstacle width of `0.5` meters | Used in Room 5 |
| Configurable forward observation | The user can select the observation distance |
| Random room test | Room 5 can test the learned policy with a new seed |
| Training and exploration graphs | Reward, steps, success, epsilon, convergence, and TD error |
| Individual episode replay | Every training attempt is stored and can be replayed |
| Hyperparameter control | Available in the training tab of every room |
| Hyperparameter optimization | Four candidates are compared before the best full run |
| Saved experiment results | CSV, JSON, and PNG files are saved under `runs/` |

## Room 1: Pac-Man Ice Maze

### Task

The agent starts at `(0,0)` and must reach the exit at `(9,9)`. It has to move through a maze, avoid cracks and moving ghosts, and handle slippery ice tiles.

### Algorithm

I used Dynamic Programming with Value Iteration because the complete environment model is known. The algorithm can access all possible transitions, their probabilities, and their rewards.

Value Iteration repeatedly updates the value of each state:

```text
V(s) = max_a sum P(s'|s,a) * [reward + gamma * V(s')]
```

The process stops when the largest value change is smaller than `theta`, or when it reaches the maximum number of iterations.

### State

```text
(row, col, collected_mask, ghost_phase)
```

`collected_mask` stays zero in this room. `ghost_phase` is included because the position of the moving ghosts affects the result of the next action.

### Actions

```text
UP, RIGHT, DOWN, LEFT
```

### Final state

The single final state is reaching `EXIT` at `(9,9)`.

### Slippery tiles

On a slippery tile, the requested action happens with probability `1-slip`. With the remaining probability, the action is redirected to the left or right.

### Rewards

| Event | Reward |
|---|---:|
| Every step | `-1` |
| Reach EXIT | `+110` |
| Crack traps | `-22` or `-25` |
| Ghost collision | `-45` and return to start |

### Parameters that worked well

```text
gamma = 0.96
theta = 0.0001
slip_probability = 0.25
max_iterations = 1000
max_steps = 220
```

In my verification, all `12/12` stochastic policy rollouts succeeded, with an average of about 18 steps.

The manual Pac-Man game uses a real-time chasing system to make the game more interactive. The reinforcement-learning environment uses the deterministic `ghost_phase` transition cycle so the known model stays reproducible.

## Room 2: Sokoban Vault

### Task

The agent must push the box from `(0,8)` onto the target at `(0,9)`. After the box is locked on the target, the player must reach the safe at `(9,9)`.

### Algorithm

I used SARSA because the agent does not know the transition model. It learns only from the states and rewards returned by `env.step()`.

SARSA is an on-policy algorithm. Its update uses the next action that the current epsilon-greedy policy actually chooses:

```text
Q(S,A) = Q(S,A) + alpha * [R + gamma*Q(S',A') - Q(S,A)]
```

### State

```text
(player_row, player_col, box_row, box_col)
```

The box position is part of the actual state. The environment checks whether the box can be pushed into the next tile, so it is a real Sokoban task and not only a visual object.

### Actions

```text
UP, RIGHT, DOWN, LEFT
```

### Final state

The box must be on `TARGET` and the player must be in `SAFE`.

### Rewards

| Event | Reward |
|---|---:|
| Every step | `-1` |
| Box reaches TARGET | `+28` |
| Box leaves TARGET | `-28` |
| Reach SAFE after solving the box | `+130` |
| Invalid push or blocked move | `-6` additional penalty |
| Laser tiles | `-30` or `-35` |

### Parameters that worked well

```text
episodes = 650
max_steps = 250
alpha = 0.15
gamma = 0.96
epsilon = 0.40
epsilon_min = 0.03
epsilon_decay = 0.993
slip_probability = 0.18
```

In my verification, the final `50/50` episodes succeeded, with an average of about 18.96 steps.

## Room 3: Bomberman Reactor

### Task

The agent must collect two CORE items, avoid bombs and moving patrol bots, use the WARP tunnel when useful, and then reach the GATE.

### Algorithm

I used Q-Learning because the environment model is unknown. Q-Learning is off-policy: the behavior still uses epsilon-greedy exploration, but the update target uses the best estimated next action.

```text
Q(S,A) = Q(S,A) + alpha * [R + gamma*max Q(S',a) - Q(S,A)]
```

### State

```text
(row, col, core_mask, guard_phase)
```

`core_mask` stores which CORE items were already collected. `guard_phase` stores the movement phase of the patrol bots.

### Actions

```text
UP, RIGHT, DOWN, LEFT
```

### Final state

Both CORE items must be collected before the agent enters `GATE`.

### Rewards

| Event | Reward |
|---|---:|
| Every step | `-1.2` |
| Collect each CORE once | `+30` |
| Reach GATE with both COREs | `+170` |
| Bomb traps | `-26`, `-30`, or `-38` |
| Patrol collision | `-60` and return to start |
| Use WARP | `0` |

The WARP reward is zero because I did not want the agent to earn an unlimited reward by moving through a portal loop.

### Parameters that worked well

```text
episodes = 650
max_steps = 850
alpha = 0.15
gamma = 0.96
epsilon = 0.40
epsilon_min = 0.03
epsilon_decay = 0.993
slip_probability = 0.18
```

In my verification, the final `50/50` episodes succeeded, with an average of about 71.68 steps.

## Room 4: Lunar Lander Pad

### Task

The fourth room is not based on a grid. The agent moves through a continuous `10x10` meter area, avoids asteroid fields, and tries to reach the landing pad.

### Algorithm

I used linear Approximate Q-Learning. A normal Q-table is not practical here because `X` and `Y` are continuous and can have a very large number of possible values.

The approximation uses:

```text
Q(s,a) = sum weights(a,i) * features(s,i)
```

The features include normalized position, direction and distance to the goal, discrete velocity, and six offset tile codings.

### State

```text
(X, Y, Vx, Vy)
```

### Actions and movement

`Vx` and `Vy` can each be `-1`, `0`, or `1`, so the agent has nine actions. Position is continuous, but velocity is discrete.

The agent chooses a new velocity every `0.02` seconds:

```text
X_new = X + Vx * speed * 0.02
Y_new = Y + Vy * speed * 0.02
```

### Final state

The agent must enter the goal radius around `PAD`. When it succeeds, the environment returns the exact goal position with `Vx=0` and `Vy=0`.

### Rewards

| Event | Reward |
|---|---:|
| Every time step | `-0.015` |
| Progress toward PAD | `5 * (old_distance - new_distance)` |
| Reach PAD | `+55` |
| Wall collision | `-0.6` |
| Asteroid collision | `-2.5` and velocity reset |

The progress reward helps the agent learn before it reaches the final state. The step penalty still encourages a shorter solution.

### Parameters that worked well

```text
episodes = 450
max_steps = 850
alpha = 0.08
gamma = 0.985
epsilon = 0.40
epsilon_min = 0.03
epsilon_decay = 0.993
```

With all five asteroid fields enabled, `37/50` of the final episodes succeeded. The average was about 714.36 time steps.

## Room 5: Portal Hazard Run

### Task

This is the optional and hardest room. The agent must reach the exit while avoiding moving portal hazards. The number and position of the obstacles change, and the agent can only observe a limited distance in front of it. Unlike the Pac-Man ghosts, a portal does not chase the player or send it back to the start. Contact gives a penalty and teleports the player to another random safe position.

### Algorithm

I used Approximate Q-Learning again, but this time the feature vector also includes information about the nearest visible obstacle.

### State

```text
(X, Y, Vx, Vy, obstacle_dx, obstacle_dy, visible)
```

`obstacle_dx` and `obstacle_dy` describe the nearest obstacle in front of the agent, normalized by the observation range. `visible` is `1` when an obstacle is visible and `0` otherwise.

### Obstacles

| Property | Value |
|---|---|
| Width | `0.5` meters |
| Default count | `7`, configurable from 2 to 15 |
| Default observation range | `3` meters, configurable from 1 to 6 |
| Position | Randomized when the room is reset |
| Movement | Horizontal or vertical |
| Collision | Penalty and teleport to a random safe position |

The distance is measured from the center of the player to the center of the obstacle. Only the nearest obstacle in front of the current heading is added to the observation.

### Final state

The agent must reach `EXIT` without being sent back by a moving portal.

### Rewards

| Event | Reward |
|---|---:|
| Every time step | `-0.015` |
| Progress toward EXIT | `4.5 * (old_distance - new_distance)` |
| Reach EXIT | `+45` |
| Moving portal collision | `-8`, random safe teleport, and velocity reset |
| Static hazard collision | `-2.5` |
| Wall collision | `-0.6` |

### Parameters that worked well

```text
episodes = 450
max_steps = 1400
alpha = 0.08
gamma = 0.985
epsilon = 0.40
epsilon_min = 0.03
epsilon_decay = 0.993
obstacle_count = 7
observation_range = 3.0
```

With the random-teleport mechanic enabled, `49/50` of the final episodes succeeded in my verification. The successful episodes averaged about 659.65 steps, and the final 50 episodes had a mean reward of 83.95. The room is still challenging because the obstacle layout, portal motion, teleport destination, and observation are changing.

After training, the Replay tab can create a new random room using a different seed and test the learned policy without additional training.

## Important hyperparameters

- `seed`: controls randomness and makes an experiment reproducible.
- `episodes`: number of complete training attempts.
- `max_steps`: maximum actions allowed in one episode.
- `alpha`: learning rate and update strength.
- `gamma`: importance of future rewards.
- `epsilon`: initial probability of choosing a random action.
- `epsilon_min`: minimum exploration probability.
- `epsilon_decay`: how quickly exploration decreases.
- `slip_probability`: movement noise on slippery tiles.
- `theta`: Value Iteration convergence threshold.
- `observation_range`: how far Room 5 can observe obstacles.

## Hyperparameter optimization

Every room has two training options:

- `Start training` uses the values selected in the interface.
- `Optimize and train` compares four candidate configurations and then performs a full training run with the best candidate.

To keep optimization practical, the continuous-room candidates use short trial runs and do not record Replay frames. Room 5 evaluates up to 70 episodes per candidate. After a candidate is selected, the application runs the complete requested training and records every episode for Replay.

The candidate score is based mainly on success rate, followed by total reward and solution length:

```text
score = success_rate * 10000 + mean_reward - mean_steps * 0.05
```

The selected values are experimental results that worked well in this project. I do not claim that they are a mathematical global optimum.

The tuning comparison is saved under `runs/tuning/`.

## Episode Replay

SARSA, Q-Learning, and Approximate Q-Learning save every training episode. Value Iteration saves 12 stochastic policy rollouts because it converges through repeated sweeps rather than through training episodes.

For every attempt, the project records:

- Episode number.
- Complete state sequence.
- Actions.
- Rewards at every step.
- Total reward.
- Number of steps.
- Success or failure.
- Epsilon value.
- Moving-obstacle frames in Room 5.

The Replay tab can filter successful or failed attempts, sort them, select any episode, play or pause it, move one step at a time, scrub the timeline, and change the playback speed.

## Analytics and saved files

The application shows:

- Reward per episode.
- Reward moving average over 25 episodes.
- Steps per episode.
- Moving success rate.
- Epsilon during training.
- TD error for Approximate Q-Learning.
- Convergence delta and start-state value for Value Iteration.

Every completed run creates a directory under `runs/` containing:

```text
metrics.csv
attempts.csv
parameters.json
summary.json
learning_report.png
```

## Project structure

```text
app.py                         Streamlit interface, game Canvas, training, replay, and analytics
escape_room/envs.py            Environment states, transitions, rewards, and obstacles
escape_room/algorithms.py      Value Iteration, SARSA, Q-Learning, and Approximate Q-Learning
escape_room/replay.py          Episode filtering and replay-library data
static/game_art/               Game backgrounds, thumbnails, and banners
tests/test_requirements.py     Automated requirement tests
.streamlit/config.toml         Streamlit theme and static-file settings
requirements.txt               Python dependencies
runtime.txt                    Python version for Streamlit Cloud
```

## Running the project locally

Create or activate a Python environment, then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the application at:

```text
http://localhost:8501
```

## Running the tests

```bash
python -m unittest discover -s tests -v
```

The tests check the main project requirements, including the grid sizes, slippery cells, known transition model, Sokoban behavior, final states, continuous dynamics, obstacle width, observation state, replay library, and packaged game artwork.

## Publishing with Streamlit Community Cloud

1. Push the project to a public GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select the repository and the `main` branch.
4. Set the main file path to `app.py`.
5. Deploy the application.

The project does not need API keys or other secrets.

## Final summary

This project starts with planning in a known model, continues with tabular model-free learning, and then moves to continuous state spaces and function approximation. Building the five rooms helped me understand that choosing the state, reward function, exploration strategy, and hyperparameters is just as important as choosing the reinforcement-learning algorithm itself.
