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
    id="maze2d-twogoals-multimodal-bonus-v0",
    entry_point="jaxrl_m.envs.maze:MazeEnv",
    max_episode_steps=800,
    kwargs={"bonus": True},
)

# register(
#     id="maze2d-hidden-v0",
#     entry_point="jaxrl_m.envs.maze:MazeEnv",
#     max_episode_steps=800,
#     kwargs={"hidden_eval": True},
# )

register(
    id="kitchen-mode0-v0",
    entry_point="jaxrl_m.envs.kitchen:KitchenEnv",
    max_episode_steps=280,
    kwargs={"mode": 0, "task_penalty": False},
)

register(
    id="kitchen-mode1-v0",
    entry_point="jaxrl_m.envs.kitchen:KitchenEnv",
    max_episode_steps=280,
    kwargs={"mode": 1, "task_penalty": False},
)

register(
    id="kitchen-multimodal-v0",
    entry_point="jaxrl_m.envs.kitchen:KitchenEnv",
    max_episode_steps=280,
    kwargs={"mode": -1, "task_penalty": False},
)

register(
    id="kitchen-taskpenalty-v0",
    entry_point="jaxrl_m.envs.kitchen:KitchenEnv",
    max_episode_steps=280,
    kwargs={"mode": 0, "task_penalty": True},
)

register(
    id="sort-easy-v0",
    entry_point="jaxrl_m.envs.sort_easy:SortEnv",
    max_episode_steps=25,
)

register(
    id="sort-easy-fixed-v0",
    entry_point="jaxrl_m.envs.sort_easy:SortEnv",
    max_episode_steps=25,
    kwargs={"mode": 0},
)

