import gym
import numpy as np
import d4rl


class MazeEnv(gym.Env):
    def __init__(self, mode=-1):
        super().__init__()
        self.env = gym.make("maze2d-large-v1")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._max_episode_steps = self.env._max_episode_steps

        self.mode = mode
        self.goals = np.array([(7, 1), (7, 10)])
        self.relabel_offline_reward = True
        self.is_multimodal = mode < 0

    @property
    def target(self):
        return self.env._target
    
    def get_dataset(self):
        return self.env.get_dataset()
    
    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()

    def step(self, action):
        # Compute shaped reward
        obs, reward, _, info = self.env.step(action)
        # Override environment termination
        done = self._elapsed_steps >= self.env._max_episode_steps
        self._elapsed_steps += 1
        return obs, reward, done, info

    def compute_reward(self, obs, mode):
        # Setting mode to random if not provided
        if self.mode < 0:
            if mode < 0:
                mode = np.random.randint(2)
            mode = mode
        else:
            mode = self.mode
        goal = self.goals[mode]
        obs_xy = obs[:, :, :2]
        dist_to_goal = np.linalg.norm(goal[None, None] - obs_xy, axis=-1)
        rewards = -dist_to_goal
        bonus = 1 * (dist_to_goal < 0.5)
        # Reward is negative distance to goal + bonus
        return rewards + bonus

    def plot_gt(self, wandb_log=False):
        import matplotlib.pyplot as plt
        import wandb

        xv, yv = np.meshgrid(
            np.linspace(*(0, 8), 100), np.linspace(*(0, 11), 100), indexing="ij"
        )
        points = np.concatenate([xv.reshape(-1, 1), yv.reshape(-1, 1)], axis=1)[None]
        r = [self.compute_reward(points, g) for g in self.goals]
        fig, axs = plt.subplots(1, 2, figsize=(10, 8))
        axs_flat = axs.flatten()
        for i, ax in enumerate(axs_flat):
            ax.imshow(
                (r[i].reshape(100, 100)).T,
                cmap="viridis",
                interpolation="nearest",
                origin="lower",
            )
            self.plot_goals(ax, 100)
        plt.tight_layout()
        if wandb_log:
            return wandb.Image(fig)
        else:
            plt.savefig("reward_plot")
        plt.close(fig)

    def plot_goals(self, ax):
        for g in self.get_goals():
            ax.scatter(g[0], g[1], s=20, c="red", marker="*")

    def get_biased_data(self, set_len):
        w, l, _ = self.factor_int(set_len * 2)
        obs = np.mgrid[0 : 8 : w * 1j, 0 : 11 : l * 1j]
        obs = obs.reshape(obs.shape[0], -1).T
        # idxs = np.random.permutation(np.arange(obs.shape[0]))
        return obs[:set_len], np.copy((obs[set_len:])[::-1])

    def get_obs_grid(self):
        obs = np.mgrid[0:8:50j, 0:11:50j]
        obs = obs.reshape(obs.shape[0], -1).T
        return obs
    
    def get_goals(self):
        return (self.goals/np.array([8, 11]))*50

    def render(self, mode="rgb_array"):
        return self.env.render(mode)

    ## Functions to handle multimodality
    def get_num_modes(self):
        return 2

    def sample_mode(self):
        if self.is_multimodal:
            return np.random.randint(2)
        return self.mode

    def reset_mode(self):
        self.set_mode(self.sample_mode())

    def set_mode(self, mode):
        self.mode = mode
        self.env.set_target(self.goals[mode])