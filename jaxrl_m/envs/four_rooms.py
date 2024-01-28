import numpy as np
import gym

from d4rl.pointmaze import MazeEnv
from jaxrl_m.envs.base import MultiModalEnv
import wandb
import matplotlib.pyplot as plt


FOUR_ROOMS_ENV = (
    "#################\\"
    + "#OOOOOOO#OOOOOOO#\\"
    + "#OOOOOOO#OOOOOOO#\\"
    + "#OOOOOOOOOOOOOOO#\\"
    + "#OOOOOOO#OOOOOOO#\\"
    + "#OOOOOOO#OOOOOOO#\\"
    + "####O#######O####\\"
    + "#OOOOOOO#OOOOOOO#\\"
    + "#OOOOOOO#OOOOOOO#\\"
    + "#OOOOOOOOOOOOOOO#\\"
    + "#OOOOOOO#OOOOOGO#\\"
    + "#OOOOOOO#OOOOOOO#\\"
    + "#################"
)


TARGET = None


def mode1_fn(x, y):
    x_, y_ = x - 6 / 12, y - 8 / 16

    if x_ < 0:
        if y_ < 0:
            d1 = -np.linalg.norm(np.array([x, y]) - np.array([3 / 12, 8 / 16]))
            d2 = -np.linalg.norm(
                np.array([3 / 12, 8 / 16]) - np.array([6 / 12, 12 / 16])
            )
            d3 = -np.linalg.norm(np.array([6 / 12, 12 / 16]) - TARGET)
            return d1 + d2 + d3
        else:
            d1 = -np.linalg.norm(np.array([x, y]) - np.array([6 / 12, 12 / 16]))
            d2 = -np.linalg.norm(np.array([6 / 12, 12 / 16]) - TARGET)
            return d1 + d2
    else:
        if y_ < 0:
            d1 = -np.linalg.norm(np.array([x, y]) - np.array([6 / 12, 4 / 16]))
            d2 = -np.linalg.norm(
                np.array([6 / 12, 4 / 16]) - np.array([3 / 12, 8 / 16])
            )
            d3 = -np.linalg.norm(
                np.array([3 / 12, 8 / 16]) - np.array([6 / 12, 12 / 16])
            )
            d4 = -np.linalg.norm(np.array([6 / 12, 12 / 16]) - TARGET)
            return d1 + d2 + d3 + d4
        else:
            return -np.linalg.norm(np.array([x, y]) - TARGET)


def mode2_fn(x, y):
    x_, y_ = x - 6 / 12, y - 8 / 16

    if x_ < 0:
        if y_ < 0:
            d1 = -np.linalg.norm(np.array([x, y]) - np.array([6 / 12, 4 / 16]))
            d2 = -np.linalg.norm(
                np.array([6 / 12, 4 / 16]) - np.array([9 / 12, 8 / 16])
            )
            d3 = -np.linalg.norm(np.array([9 / 12, 8 / 16]) - TARGET)
            return d1 + d2 + d3
        else:
            d1 = -np.linalg.norm(np.array([x, y]) - np.array([3 / 12, 8 / 16]))
            d2 = -np.linalg.norm(
                np.array([3 / 12, 8 / 16]) - np.array([6 / 12, 4 / 16])
            )
            d3 = -np.linalg.norm(
                np.array([6 / 12, 4 / 16]) - np.array([9 / 12, 8 / 16])
            )
            d4 = -np.linalg.norm(np.array([9 / 12, 8 / 16]) - TARGET)
            return d1 + d2 + d3 + d4
    else:
        if y_ < 0:
            d1 = -np.linalg.norm(np.array([x, y]) - np.array([9 / 12, 8 / 16]))
            d2 = -np.linalg.norm(np.array([9 / 12, 8 / 16]) - TARGET)
            return d1 + d2
        else:
            return -np.linalg.norm(np.array([x, y]) - TARGET)


vec_mode1 = np.vectorize(mode1_fn)
vec_mode2 = np.vectorize(mode2_fn)


class FourRoomsEnv(MultiModalEnv):
    def __init__(self, dataset_path, fixed_mode=False, **kwargs):
        super().__init__(dataset_path=dataset_path, fixed_mode=fixed_mode, **kwargs)

        self.env = MazeEnv(
            maze_spec=FOUR_ROOMS_ENV, reward_type="dense", reset_target=False, **kwargs
        )
        self.env.empty_and_goal_locations = [(1, 1)]

        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(2,),
        )  # self.env.observation_space
        self.x_range = (0, 12)
        self.y_range = (0, 16)
        self.max_x = np.array([12, 16])

        global TARGET
        TARGET = np.array(self.env._target) / self.max_x
        self._max_episode_steps = kwargs.get("max_episode_steps", 600)

        self.str_maze_spec = self.env.str_maze_spec
        self.sim = self.env.sim

    def get_preference_rewards(
        self, state1, state2, mode=None
    ):  # states are pf size B x T x STATE_DIM
        mode = mode or self.sample_mode()
        if mode == 0:
            r0 = self._mode_0_r(state1)
            r1 = self._mode_0_r(state2)
        else:
            r0 = self._mode_1_r(state1)
            r1 = self._mode_1_r(state2)
        return r0, r1

    def get_reward(self, state, mode):
        if mode == 0:
            return self._mode_0_r(state)
        else:
            return self._mode_1_r(state)

    def _mode_0_r(self, state):
        return vec_mode1(state[:, :, 0], state[:, :, 1])

    def _mode_1_r(self, state):
        return vec_mode2(state[:, :, 0], state[:, :, 1])

    def plot_gt(self, wandb_log=False):
        xv, yv = np.meshgrid(
            np.linspace(0, 1, 120), np.linspace(0, 1, 160), indexing="ij"
        )
        points = np.concatenate([xv.reshape(-1, 1), yv.reshape(-1, 1)], axis=1)[None]
        rewards = [self._mode_0_r(points), self._mode_1_r(points)]
        fig, axs = plt.subplots(1, 2, figsize=(10, 8))
        axs_flat = axs.flatten()
        for i, ax in enumerate(axs_flat):
            r = rewards[i].reshape(120, 160)
            r = (r - r.min()) / (r.max() - r.min())
            im = ax.imshow(r.T, cmap="viridis", interpolation="nearest")
            ax.scatter(self.env._target[0] * 10, self.env._target[1] * 10, c="r")
            ax.scatter(30, 120, c="g")
            ax.scatter(90, 40, c="b")
            ax.scatter(10, 10, c="black")
        plt.tight_layout()
        if wandb_log:
            return wandb.Image(fig)
        else:
            plt.savefig("reward_plot.png")
        plt.close(fig)
        return points

    def get_obs_grid(self):
        return (
            np.mgrid[0:1:120j, 0:1:160j],
            120,
            160,
            (self.env._target[0] * 10, self.env._target[1] * 10),
        )
