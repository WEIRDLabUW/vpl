import numpy as np
from itertools import permutations

from d4rl.kitchen.kitchen_envs import KitchenMicrowaveKettleLightSliderV0
from jaxrl_m.envs.base import MultiModalEnv

OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}
OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}

OBJECT_THRESH = {
    "slide cabinet": 0.1,
    "hinge cabinet": 1.0,
    "microwave": 0.4,
}


class FrankaKitchenEnv(MultiModalEnv):
    def __init__(
        self, dataset_path, fixed_mode=False, task_config=["slide cabinet", "microwave"]
    ):
        super().__init__(dataset_path=dataset_path, fixed_mode=fixed_mode)

        self.env = KitchenMicrowaveKettleLightSliderV0()

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        print("observation space in kitchen", self.observation_space)

        self.tasks = task_config
        self.all_task_orders = list(permutations(self.tasks))

        if self.fixed_mode:
            self.env_task = self.all_task_orders[self.env_mode]

        self.relabel_offline_reward = True

    def set_mode(self, mode):
        super().set_mode(mode)
        self.env_task = self.all_task_orders[self.env_mode]

    def get_num_modes(self):
        return len(self.all_task_orders)

    def reset(self):
        obs = self.env.reset()
        if not self.fixed_mode:
            self.set_mode(self.sample_mode())
        return obs

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        info["vel"] = obs[2:4]
        info["success"] = self.get_success(obs)
        if info["success"] == 1:
            done = True
        return obs, reward, done, info

    def get_success(self, state):
        return np.array(
            [
                np.linalg.norm(
                    state[OBS_ELEMENT_INDICES[task]] - OBS_ELEMENT_GOALS[task], axis=-1
                )
                < OBJECT_THRESH[task]
                for task in self.tasks
            ]
        ).mean()

    def get_reward(self, state, mode):
        task_order = self.all_task_orders[mode]
        r = self.get_ri(state, task_order)
        return r

    def get_preference_rewards(
        self, state1, state2, mode=None
    ):  # states are pf size B x T x STATE_DIM
        mode = mode or self.sample_mode()
        task_order = self.all_task_orders[mode]
        r0 = self.get_ri(state1, task_order)
        r1 = self.get_ri(state2, task_order)
        return r0, r1

    def get_ri(self, state, task_order):
        bonus = 5.0
        rewards = np.zeros(state.shape[:-1])
        mask = np.ones(state.shape[:-1])
        for task in task_order:
            tasks_completed = (
                np.linalg.norm(
                    state[:, :, OBS_ELEMENT_INDICES[task]] - OBS_ELEMENT_GOALS[task],
                    axis=-1,
                )
                < OBJECT_THRESH[task]
            )
            tasks_distance = np.linalg.norm(
                state[:, :, OBS_ELEMENT_INDICES[task]] - OBS_ELEMENT_GOALS[task],
                axis=-1,
            )
            rewards += (
                tasks_completed * bonus + (1 - tasks_completed) * mask * tasks_distance
            )
            mask *= tasks_completed
        return rewards

    def plot_gt(self, wandb_log=False):
        pass

    def plot_goals(self, ax, scale):
        pass
