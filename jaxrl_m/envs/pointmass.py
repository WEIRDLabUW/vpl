import gym
import numpy as np
import matplotlib.pyplot as plt
import wandb

from d4rl.pointmaze import MazeEnv
from jaxrl_m.envs.base import MultiModalEnv

POINTMASS_ENV = (
    "#######\\"
    + "#OOOOO#\\"
    + "#OO#OO#\\"
    + "#OO#OG#\\"
    + "#OO#OO#\\"
    + "#OOOOO#\\"
    + "#######"
)


class PointMassEnv(MultiModalEnv):
    def __init__(self, dataset_path, fixed_mode=False, **kwargs):
        super().__init__(dataset_path=dataset_path, fixed_mode=fixed_mode, **kwargs)

        self.env = MazeEnv(
            maze_spec=POINTMASS_ENV, reward_type="dense", reset_target=False, **kwargs
        )
        self.env.empty_and_goal_locations = [(3, 1)]

        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(2,)
        )  # self.env.observation_space

        self.x_range = (0, 6)
        self.y_range = (0, 6)
        self.max_x = np.array([6, 6])
        self.goal1 = np.array([1, 5])
        self.goal2 = np.array([5, 5])

        self.str_maze_spec = self.env.str_maze_spec
        self.sim = self.env.sim
        self._max_episode_steps = kwargs.get("max_episode_steps", 300)
        if self.fixed_mode:
            self.env.set_target(self.goal1 if self.env_mode == 0 else self.goal2)

    def set_mode(self, mode):
        super().set_mode(mode)
        self.env.set_target(self.goal1 if mode == 0 else self.goal2)

    def get_reward(self, state, mode):
        return self.get_ri(state, self.goal1 if mode == 0 else self.goal2)

    def get_preference_rewards(
        self, state1, state2, mode=None
    ):  # states are pf size B x T x STATE_DIM
        mode = mode or self.sample_mode()
        if mode == 0:
            r0 = self.get_ri(state1, self.goal1)
            r1 = self.get_ri(state2, self.goal1)
        else:
            r0 = self.get_ri(state1, self.goal2)
            r1 = self.get_ri(state2, self.goal2)
        return r0, r1

    def get_ri(self, state, target):
        dist_goal = -np.linalg.norm(state[:, :, 0:2] - target / self.max_x, axis=2)
        return dist_goal

    def plot_gt(self, wandb_log=False):
        xv, yv = np.meshgrid(
            np.linspace(*(0, 1), 100), np.linspace(*(0, 1), 100), indexing="ij"
        )
        points = np.concatenate([xv.reshape(-1, 1), yv.reshape(-1, 1)], axis=1)[None]
        r0 = self.get_ri(points, self.goal1)
        r1 = self.get_ri(points, self.goal2)
        r = [r0, r1]
        # target_p = 100 * (np.array(self.env._target) - self.x_range[0]) / (self.x_range[1] - self.x_range[0])

        fig, axs = plt.subplots(1, 2, figsize=(10, 8))
        axs_flat = axs.flatten()
        for i, ax in enumerate(axs_flat):
            ax.imshow(
                (r[i].reshape(100, 100)).T, cmap="viridis", interpolation="nearest"
            )
            # ax.scatter(target_p[0], target_p[1], c='r')
            self.plot_goals(ax, 100)
        plt.tight_layout()
        if wandb_log:
            if wandb_log:
                return wandb.Image(fig)
        else:
            plt.savefig("reward_plot")
        plt.close(fig)
        return points

    def plot_goals(self, ax, scale):
        for g in [self.goal1, self.goal2]:
            target_p = (
                scale
                * (np.array(g) - self.x_range[0])
                / (self.x_range[1] - self.x_range[0])
            )
            ax.scatter(target_p[0], target_p[1], s=20, c="red", marker="*")
            if self.fixed_mode:
                break
    
    def get_biased_data(self, set_len):
        w, l, _ = self.factor_int(set_len*2)
        obs = np.mgrid[0:1:w*1j, 0:1:l*1j]
        obs = obs.reshape(obs.shape[0], -1).T
        # idxs = np.random.permutation(np.arange(obs.shape[0]))
        return obs[:set_len], np.copy((obs[set_len:])[::-1])
