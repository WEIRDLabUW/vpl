import d4rl
import gym
import numpy as np
from tqdm import tqdm

from jaxrl_m.dataset import Dataset
from jaxrl_m.evaluation import EpisodeMonitor


def make_env(env_name: str):
    env = gym.make(env_name)
    env = EpisodeMonitor(env)
    return env


def get_dataset(
    env: gym.Env,
    clip_to_eps: bool = True,
    eps: float = 1e-5,
    add_mode: bool = False,
):
    dataset = d4rl.qlearning_dataset(env, terminate_on_end=True)
    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    dones_float = dataset["terminals"].astype(np.float32)
    dataset = {
        "observations": dataset["observations"],
        "actions": dataset["actions"],
        "rewards": dataset["rewards"],
        "masks": 1.0 - dataset["terminals"],
        "dones_float": dones_float,
        "next_observations": dataset["next_observations"],
    }

    if hasattr(env, "relabel_offline_reward") and env.relabel_offline_reward:
        print("Relabelling rewards and appending mode to obs is:", add_mode)
        dataset = relabel_rewards(env, dataset, add_mode)
    dataset = {k: v.astype(np.float32) for k, v in dataset.items()}
    return Dataset(dataset)


def new_get_trj_idx(dataset):
    N = dataset["rewards"].shape[0]
    episode_step = 0
    start_idx, data_idx = 0, 0
    trj_idx_list = []
    for i in range(N - 1):
        done_bool = dataset["dones_float"][i]
        if done_bool:
            episode_step = 0
            trj_idx_list.append([start_idx, data_idx])
            start_idx = data_idx + 1

        episode_step += 1
        data_idx += 1

    trj_idx_list.append([start_idx, data_idx])
    return trj_idx_list

def relabel_rewards(env, dataset, add_mode):
    new_rewards = []
    new_observations = []
    new_next_observations = []
    traj_list = new_get_trj_idx(dataset)
    modes = []
    for (start, end) in traj_list:
        obs = dataset["observations"][start:end]
        next_obs = dataset["next_observations"][start:end]
        mode = env.sample_mode()
        new_rewards.append(env.compute_reward(obs[None], mode)[0])
        modes.append(mode)
        if add_mode:
            new_observations.append(
                np.concatenate([obs, np.ones_like(obs[:, :1]) * mode], axis=-1)
            )
            new_next_observations.append(
                np.concatenate([next_obs, np.ones_like(next_obs[:, :1]) * mode], axis=-1)
            )
    print("Mean mode:", np.mean(modes))
    new_rewards = np.concatenate(new_rewards)
    normalised_rewards = (new_rewards - new_rewards.min()) / (new_rewards.max() - new_rewards.min())
    dataset["rewards"] = normalised_rewards
    if add_mode:
        dataset["observations"] = np.concatenate(new_observations, axis=0)
        dataset["next_observations"] = np.concatenate(new_next_observations, axis=0)
    return dataset
