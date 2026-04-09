# MiniGrid Agent Mini-Project

## Overview
This mini-project involves developing an agent to navigate a family of [MiniGrid](https://minigrid.farama.org/) environments. MiniGrid is a lightweight grid-world benchmark widely used in reinforcement learning research. The agent must observe the environment and select actions to reach the goal as efficiently as possible.

## Setup
The project dependencies are specified in `environment.yml`. To set up the Conda environment, run:

```bash
conda env create -f environment.yml
conda activate csxx46-ay2526s2-mini-project
```

## Environments
The agent needs to solve the following 6 tasks (one warm-up and five main tasks):
1. **Empty 8x8** (`MiniGrid-Empty-8x8-v0`): A warm-up open grid with no obstacles.
2. **Door Key 8x8** (`MiniGrid-DoorKey-8x8-v0`): Pick up a key, unlock a door, and reach the goal.
3. **Four Rooms** (`MiniGrid-FourRooms-v0`): Four interconnected rooms with varying goal positions.
4. **Dynamic Obstacles 6x6** (`MiniGrid-Dynamic-Obstacles-6x6-v0`): Compact grid with moving obstacles. Avoid collisions.
5. **Lava Gap** (`MiniGrid-LavaGapS7-v0`): Navigate across a lava wall through a narrow gap.
6. **Memory** (`MiniGrid-MemoryS13Random-v0`): Remember a cue from earlier in the episode to choose the correct path.

## Agent Development
The agent should be implemented in the `Agent` class via the following public interface:
- `__init__(self, obs_space, action_space)`: Initialize the agent and load any required models.
- `reset(self)`: Called at the start of each episode to reset internal states.
- `act(self, obs) -> int`: Returns an integer action `{0..6}` given an observation.
  - 0 = Turn Left
  - 1 = Turn Right
  - 2 = Move Forward
  - 3 = Pick Up
  - 4 = Drop
  - 5 = Toggle
  - 6 = Done

The observation dictionary contains:
- `image`: shape `(7, 7, 3)`, dtype `uint8` representing the agent's 7x7 partial field of view (encoded as `object, colour, state`).
- `direction`: agent's facing direction (`0=right`, `1=down`, `2=left`, `3=up`).
- `mission`: task description in string format.

## Submission
Submit the final `Agent` code on Coursemology. Upload all training scripts, generation scripts, and model checkpoints used to build your agent as supplementary materials.
