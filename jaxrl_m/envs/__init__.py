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
    id="maze2d-fourrooms-fixed-v0",
    entry_point="jaxrl_m.envs.four_rooms:FourRoomsEnv",
    max_episode_steps=600,
    kwargs={
        "dataset_path": os.path.join(
            os.path.dirname(__file__), "datasets/four_rooms.hdf5"
        ),
        "fixed_mode": True,
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

register(
    id="kitchen-v0",
    entry_point="jaxrl_m.envs.franka_kitchen:FrankaKitchenEnv",
    max_episode_steps=280,
    kwargs={
        "dataset_path": os.path.join(os.path.dirname(__file__), "datasets/kitchen.hdf5")
    },
)

register(
    id="kitchen-fixed-v0",
    entry_point="jaxrl_m.envs.franka_kitchen:FrankaKitchenEnv",
    max_episode_steps=280,
    kwargs={
        "dataset_path": os.path.join(os.path.dirname(__file__), "datasets/kitchen.hdf5"),
        "fixed_mode": True,
    },
)

register(
    id="kitchen-dense-v0",
    entry_point="jaxrl_m.envs.franka_kitchen_dense:FrankaKitchenDenseEnv",
    max_episode_steps=280,
    kwargs={
        "dataset_path": os.path.join(os.path.dirname(__file__), "datasets/kitchen.hdf5")
    },
)

register(
    id="kitchen-dense-fixed-v0",
    entry_point="jaxrl_m.envs.franka_kitchen_dense:FrankaKitchenDenseEnv",
    max_episode_steps=280,
    kwargs={
        "dataset_path": os.path.join(os.path.dirname(__file__), "datasets/kitchen.hdf5"),
        "fixed_mode": True,
    },
)

