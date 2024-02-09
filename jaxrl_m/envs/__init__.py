import os
from gym.envs.registration import register

register(
    id="maze2d-twogoals-mode0-v0",
    entry_point="jaxrl_m.envs.maze:MazeEnv",
    max_episode_steps=600,
    kwargs={"mode": 0},
)

register(
    id="maze2d-twogoals-mode1-v0",
    entry_point="jaxrl_m.envs.maze:MazeEnv",
    max_episode_steps=600,
    kwargs={"mode": 1},
)

register(
    id="maze2d-twogoals-multimodal-v0",
    entry_point="jaxrl_m.envs.maze:MazeEnv",
    max_episode_steps=600,
)
