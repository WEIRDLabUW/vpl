import os
from gym.envs.registration import register

register(
    id="maze2d-pointmass-fixed-v0",
    entry_point="jaxrl_m.envs.pointmass:PointMassEnv",
    max_episode_steps=300,
    kwargs={
        "dataset_path": os.path.join(
            os.path.dirname(__file__), "datasets/pointmass.hdf5"
        ),
        "fixed_mode": True,
    },
)

register(
    id="maze2d-pointmass-v0",
    entry_point="jaxrl_m.envs.pointmass:PointMassEnv",
    max_episode_steps=300,
    kwargs={
        "dataset_path": os.path.join(
            os.path.dirname(__file__), "datasets/pointmass.hdf5"
        ),
    },
)

register(
    id="maze2d-fourrooms-fixed-mode0-v0",
    entry_point="jaxrl_m.envs.four_rooms:FourRoomsEnv",
    max_episode_steps=600,
    kwargs={
        "dataset_path": os.path.join(
            os.path.dirname(__file__), "datasets/four_rooms.hdf5"
        ),
        "fixed_mode": True,
        "env_mode": 0
    },
)

register(
    id="maze2d-fourrooms-fixed-mode1-v0",
    entry_point="jaxrl_m.envs.four_rooms:FourRoomsEnv",
    max_episode_steps=600,
    kwargs={
        "dataset_path": os.path.join(
            os.path.dirname(__file__), "datasets/four_rooms.hdf5"
        ),
        "fixed_mode": True,
        "env_mode": 1
    },
)

register(
    id="maze2d-fourrooms-v0",
    entry_point="jaxrl_m.envs.four_rooms:FourRoomsEnv",
    max_episode_steps=600,
    kwargs={
        "dataset_path": os.path.join(
            os.path.dirname(__file__), "datasets/four_rooms.hdf5"
        ),
    },
)


register(
    id="maze2d-fourrooms2-v0",
    entry_point="jaxrl_m.envs.four_rooms2:FourRoomsEnv",
    max_episode_steps=600,
    kwargs={
        "dataset_path": os.path.join(
            os.path.dirname(__file__), "datasets/four_rooms.hdf5"
        ),
    },
)

# register(
#     id="multi-kitchen-partial-v0",
#     entry_point="jaxrl_m.envs.franka_kitchen:FrankaKitchenEnv",
#     max_episode_steps=280,
#     kwargs={
#         "dataset_path": os.path.join(os.path.dirname(__file__), "datasets/kitchen.hdf5")
#     },
# )

# register(
#     id="multi-kitchen-partial-fixed-v0",
#     entry_point="jaxrl_m.envs.franka_kitchen:FrankaKitchenEnv",
#     max_episode_steps=280,
#     kwargs={
#         "dataset_path": os.path.join(os.path.dirname(__file__), "datasets/kitchen.hdf5"),
#         "fixed_mode": True,
#     },
# )

# register(
#     id="kitchen-dense-v0",
#     entry_point="jaxrl_m.envs.franka_kitchen_dense:FrankaKitchenDenseEnv",
#     max_episode_steps=280,
#     kwargs={
#         "dataset_path": os.path.join(os.path.dirname(__file__), "datasets/kitchen.hdf5")
#     },
# )

# register(
#     id="kitchen-dense-fixed-v0",
#     entry_point="jaxrl_m.envs.franka_kitchen_dense:FrankaKitchenDenseEnv",
#     max_episode_steps=280,
#     kwargs={
#         "dataset_path": os.path.join(os.path.dirname(__file__), "datasets/kitchen.hdf5"),
#         "fixed_mode": True,
#     },
# )


register(
    id="multi-kitchen-complete-v0",
    entry_point="jaxrl_m.envs.franka_kitchen:FrankaKitchenEnv",
    max_episode_steps=280,
    kwargs={
        "dataset_path": os.path.join(os.path.dirname(__file__), "datasets/mini_kitchen_microwave_kettle_light_slider-v0.hdf5")
    },
)

register(
    id="multi-kitchen-complete-fixed-v0",
    entry_point="jaxrl_m.envs.franka_kitchen:FrankaKitchenEnv",
    max_episode_steps=280,
    kwargs={
        "dataset_path": os.path.join(os.path.dirname(__file__), "datasets/mini_kitchen_microwave_kettle_light_slider-v0.hdf5"),
        "fixed_mode": True,
    },
)

register(
    id="multi-kitchen-mixed-v0",
    entry_point="jaxrl_m.envs.franka_kitchen:FrankaKitchenEnv",
    max_episode_steps=280,
    kwargs={
        "dataset_path": os.path.join(os.path.dirname(__file__), "datasets/kitchen_microwave_kettle_light_slider-v0.hdf5")
    },
)

register(
    id="multi-kitchen-mixed-fixed-v0",
    entry_point="jaxrl_m.envs.franka_kitchen:FrankaKitchenEnv",
    max_episode_steps=280,
    kwargs={
        "dataset_path": os.path.join(os.path.dirname(__file__), "datasets/kitchen_microwave_kettle_light_slider-v0.hdf5"),
        "fixed_mode": True,
    },
)

register(
    id="multi-kitchen-mixed-fixed-2-v0",
    entry_point="jaxrl_m.envs.franka_kitchen:FrankaKitchenEnv",
    max_episode_steps=280,
    kwargs={
        "dataset_path": os.path.join(os.path.dirname(__file__), "datasets/kitchen_microwave_kettle_light_slider-v0.hdf5"),
        "fixed_mode": True,
        "env_mode": 1
    },
)

register(
    id="twogoals-v0",
    entry_point="jaxrl_m.envs.two_goals:TwoGoalsEnv",
    max_episode_steps=600,
    kwargs={
        "dataset_path": os.path.join(os.path.dirname(__file__), "datasets/maze2d-medium-dense-v1.hdf5")
    },
)

register(
    id="twogoals-fixed-v0",
    entry_point="jaxrl_m.envs.two_goals:TwoGoalsEnv",
    max_episode_steps=600,
    kwargs={
        "dataset_path": os.path.join(os.path.dirname(__file__), "datasets/maze2d-medium-dense-v1.hdf5"),
        "fixed_mode": True,
    },
)
