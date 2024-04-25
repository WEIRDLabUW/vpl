import math
import gym
import numpy as np
import d4rl
from jaxrl_m.envs.maze_utils import get_qmatrix, get_reward, apply_walls, plot_walls, factor_int


class MazeEnv(gym.Env):
    def __init__(self, mode=-1, close_goals=False, bonus=False):
        super().__init__()
        self.env = gym.make("maze2d-large-v1")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,)
        )
        self._max_episode_steps = self.env._max_episode_steps

        self.mode = mode
        if close_goals:
            self.goals = np.array([(7, 2), (7, 4)])
        else:
            self.goals = np.array([(7, 1), (7, 10)])
        self.relabel_offline_reward = True
        self.is_multimodal = mode < 0
        self.biased_mode = None
        if not self.is_multimodal:
            self.env.set_target(self.goals[mode])

        self.qmatrixes = [get_qmatrix(self.env, goal, self.get_obs_grid()) for goal in self.goals]

    @property
    def target(self):
        return self.env.unwrapped._target

    def get_dataset(self):
        return self.env.get_dataset()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        # Compute shaped reward
        obs, reward, done, info = self.env.step(action)
        # Override environment termination
        success = np.linalg.norm(obs[:2] - self.target) < 0.5
        if success:
            done = True
        info["actual_reward"] = reward
        reward = success
        return obs, reward, done, info

    def compute_reward(self, obs, mode):
        # Setting mode to random if not provided
        if self.mode < 0:
            if mode < 0:
                mode = np.random.randint(2)
            mode = mode
        else:
            mode = self.mode
        qmatrix, r_min, r_max = self.qmatrixes[mode]
        obs_xy = obs[:, :, :2]
        reward = get_reward(self.env, qmatrix, obs_xy)
        reward = (reward - r_min) / (r_max - r_min)
        return reward

    def plot_gt(self, wandb_log=False):
        import matplotlib.pyplot as plt
        import wandb

        xv, yv = np.meshgrid(
            np.linspace(*(0, 8), 100), np.linspace(*(0, 11), 100), indexing="ij"
        )
        points = np.concatenate([xv.reshape(-1, 1), yv.reshape(-1, 1)], axis=1)[None]
        r = [self.compute_reward(points, mode) for mode in range(self.get_num_modes())]
        fig, axs = plt.subplots(1, self.get_num_modes(), figsize=(self.get_num_modes()*10, 8))
        if self.get_num_modes() == 1:
            axs_flat = [axs]
        else:
            axs_flat = axs.flatten()
        for i, ax in enumerate(axs_flat):
            r_hat = apply_walls(self.env, r[i][0], points[0])
            sc = ax.scatter(points[0, :, 0], points[0, :, 1], c=r_hat)
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label("r(s)")
            self.plot_goals(ax)
        plt.tight_layout()
        if wandb_log:
            return wandb.Image(fig)
        else:
            print("Saving reward plot")
            plt.savefig("reward_plot")
        plt.close(fig)

    def plot_goals(self, ax):
        plot_walls(self.env, ax, self.get_obs_grid())
        for g in self.goals:
            ax.scatter(g[0], g[1], s=100, c="black", marker="o")

    def set_biased_mode(self, mode):
        self.biased_mode = mode

    def get_biased_data(self, set_len):
        if self.biased_mode == "grid":
            w, l, _ = factor_int(set_len * 2)
            obs = np.mgrid[0 : 8 : w * 1j, 0 : 11 : l * 1j]
            obs = obs.reshape(obs.shape[0], -1).T
        elif self.biased_mode == "random":
            obs = np.random.uniform(0, 1, (set_len * 2, 2)) * np.array([8, 11])
        elif self.biased_mode == "equal":
            obs_y = np.random.uniform(5, 7, size=(2 * set_len,))
            obs_x = np.random.uniform(1, 7, size=(2 * set_len,))
            obs = np.stack([obs_x, obs_y], axis=1)
        else:
            raise ValueError("Invalid biased mode")
        # idxs = np.random.permutation(np.arange(obs.shape[0]))
        return obs[:set_len], np.copy((obs[set_len:])[::-1])

    def get_obs_grid(self):
        obs = np.mgrid[0:8:50j, 0:11:50j]
        obs = obs.reshape(obs.shape[0], -1).T
        return obs

    def get_goals(self):
        return (self.goals / np.array([8, 11])) * 50

    def render(self, mode="rgb_array"):
        return self.env.render(mode)

    ## Functions to handle multimodality
    def get_num_modes(self):
        if self.is_multimodal:
            return len(self.goals)
        return 1

    def sample_mode(self):
        if self.is_multimodal:
            return np.random.randint(len(self.goals))
        return self.mode

    def reset_mode(self):
        self.set_mode(self.sample_mode())

    def set_mode(self, mode):
        if self.is_multimodal:
            self.mode = mode
            self.env.set_target(self.goals[mode])
