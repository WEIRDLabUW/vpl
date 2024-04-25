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
    id="maze2d-active-multimodal-v0",
    entry_point="jaxrl_m.envs.maze:MazeEnv",
    max_episode_steps=800,
    kwargs={"close_goals": True},
)

register(
    id="sort-mode0-v0",
    entry_point="jaxrl_m.envs.sort:SortEnv",
    max_episode_steps=30,
    kwargs={"mode": 0},
)

register(
    id="sort-mode1-v0",
    entry_point="jaxrl_m.envs.sort:SortEnv",
    max_episode_steps=30,
    kwargs={"mode": 1},
)

register(
    id="sort-multimodal-v0",
    entry_point="jaxrl_m.envs.sort:SortEnv",
    max_episode_steps=30,
)

register(
    id="sort-debug-v0",
    entry_point="jaxrl_m.envs.sort:SortEnv",
    max_episode_steps=30,
    kwargs={"mode": 0, "debug": True},
)

register(
    id="sort-debug2-v0",
    entry_point="jaxrl_m.envs.sort:SortEnv",
    max_episode_steps=30,
    kwargs={"mode": 0, "debug": True, "disp": True},
)

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

# register(
#     id="kitchen-multimodal-v0",
#     entry_point="jaxrl_m.envs.kitchen:KitchenEnv",
#     max_episode_steps=280,
#     kwargs={"mode": -1, "task_penalty": False},
# )

register(
    id="kitchen-taskpenalty-v0",
    entry_point="jaxrl_m.envs.kitchen:KitchenEnv",
    max_episode_steps=280,
    kwargs={"mode": 0, "task_penalty": True},
)