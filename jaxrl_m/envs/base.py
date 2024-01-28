import gym
import numpy as np
import h5py
from tqdm import tqdm


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


class MultiModalEnv(gym.Env):
    """
    A MultiModalEnv is a modified Gym environment designed for multi-modal reward tasks.
    """

    def __init__(self, dataset_path=None, fixed_mode=False, **kwargs):
        super(MultiModalEnv, self).__init__(**kwargs)
        self.dataset_path = dataset_path
        self.fixed_mode = fixed_mode
        self.env_mode = kwargs.get("env_mode", 0)

    @property
    def target(self):
        return self.env._target

    @property
    def velocity(self):
        return self.env.data.qvel[:2]
    
    def get_env_mode(self):
        return self.env_mode

    def set_mode(self, mode):
        assert not self.fixed_mode
        self.env_mode = mode
    
    def is_multimodal(self):
        return not self.fixed_mode

    def sample_mode(self):
        if self.fixed_mode:
            return self.env_mode
        return np.random.randint(self.get_num_modes())

    def reset(self):
        obs = self.env.reset()
        if not self.fixed_mode:
            self.set_mode(self.sample_mode())
        return np.array(obs[:2]) / self.max_x

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        info["vel"] = obs[2:4]
        info["success"] = self.get_success(obs[:2])
        if self.fixed_mode:
            reward = self.get_reward(obs[:2][None, None], self.env_mode)[0,0]
        return np.array(obs[:2]) / self.max_x, reward, done, info

    def get_success(self, state):
        return np.linalg.norm(state - self.env._target) < 0.5

    def get_preferences(self, state1, state2):
        raise NotImplementedError()

    def render(self, mode="rgb_array"):
        return self.env.render(mode)

    def get_num_modes(self):
        return 2

    def get_obs_grid(self):
        return (
            np.mgrid[0:1:50j, 0:1:50j],
            50,
            50,
            50
            * (np.array(self.target) - self.x_range[0])
            / (self.x_range[1] - self.x_range[0]),
        )

    def plot_reward_model(self, model, step):
        raise NotImplementedError

    def plot_reward_model_with_z(self, model, z, mode_n, step):
        raise NotImplementedError

    def get_dataset(self):
        if self.dataset_path is None:
            raise ValueError("Offline env not configured with a dataset path.")

        data_dict = {}
        with h5py.File(self.dataset_path, "r") as dataset_file:
            for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]

        # Run a few quick sanity checks
        for key in ["observations", "actions", "rewards", "terminals"]:
            assert key in data_dict, "Dataset is missing key %s" % key
        N_samples = data_dict["observations"].shape[0]
        if self.observation_space.shape is not None:
            assert (
                data_dict["observations"].shape[1:] == self.observation_space.shape
            ), "Observation shape does not match env: %s vs %s" % (
                str(data_dict["observations"].shape[1:]),
                str(self.observation_space.shape),
            )
        assert (
            data_dict["actions"].shape[1:] == self.action_space.shape
        ), "Action shape does not match env: %s vs %s" % (
            str(data_dict["actions"].shape[1:]),
            str(self.action_space.shape),
        )
        if data_dict["rewards"].shape == (N_samples, 1):
            data_dict["rewards"] = data_dict["rewards"][:, 0]
        assert data_dict["rewards"].shape == (
            N_samples,
        ), "Reward has wrong shape: %s" % (str(data_dict["rewards"].shape))
        if data_dict["terminals"].shape == (N_samples, 1):
            data_dict["terminals"] = data_dict["terminals"][:, 0]
        assert data_dict["terminals"].shape == (
            N_samples,
        ), "Terminals has wrong shape: %s" % (str(data_dict["rewards"].shape))
        return data_dict