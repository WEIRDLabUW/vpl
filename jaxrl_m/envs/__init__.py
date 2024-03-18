import os
from gym.envs.registration import register

register(
    id="maze2d-twogoals-mode0-v0",
    entry_point="jaxrl_m.envs.maze:MazeEnv",
    max_episode_steps=800,
    kwargs={"mode": 0},
)

register(
    id="maze2d-twogoals-mode1-v0",
    entry_point="jaxrl_m.envs.maze:MazeEnv",
    max_episode_steps=800,
    kwargs={"mode": 1},
)

register(
    id="maze2d-twogoals-multimodal-v0",
    entry_point="jaxrl_m.envs.maze:MazeEnv",
    max_episode_steps=800,
)

register(
    id="sort-mode0-v0",
    entry_point="jaxrl_m.envs.sort:SortEnv",
    max_episode_steps=300,
    kwargs={"mode": 0},
)

register(
    id="sort-mode1-v0",
    entry_point="jaxrl_m.envs.sort:SortEnv",
    max_episode_steps=300,
    kwargs={"mode": 1},
)

register(
    id="sort-multimodal-v0",
    entry_point="jaxrl_m.envs.sort:SortEnv",
    max_episode_steps=300,
)