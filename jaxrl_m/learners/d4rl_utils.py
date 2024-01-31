import d4rl
import gym
import numpy as np
from tqdm import tqdm

from jaxrl_m.dataset import Dataset
from jaxrl_m.evaluation import EpisodeMonitor
from jaxrl_m.envs.base import MultiModalEnv


def make_env(env_name: str):
    env = gym.make(env_name)
    env = EpisodeMonitor(env)
    return env


def get_dataset(
    env: gym.Env,
    clip_to_eps: bool = True,
    eps: float = 1e-5,
):
    dataset = d4rl.qlearning_dataset(env)

    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    imputed_next_observations = np.roll(dataset["observations"], -1, axis=0)
    same_obs = np.all(
        np.isclose(imputed_next_observations, dataset["next_observations"], atol=1e-5),
        axis=-1,
    )
    dones_float = 1.0 - same_obs.astype(np.float32)
    dones_float[-1] = 1

    dataset = {
        "observations": dataset["observations"],
        "actions": dataset["actions"],
        "rewards": dataset["rewards"],
        "masks": 1.0 - dataset["terminals"],
        "dones_float": dones_float,
        "next_observations": dataset["next_observations"],
    }

    if hasattr(env, "relabel_offline_reward") and env.relabel_offline_reward:
        print("RELABELING REWARDS")
        dataset["rewards"] = relabel_rewards(
            env, dataset["observations"], dataset["dones_float"]
        )

    dataset = {k: v.astype(np.float32) for k, v in dataset.items()}
    return Dataset(dataset)


def split_into_trajectories(observations, dones_float):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], dones_float[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def relabel_rewards(env, observations, dones_float):
    new_rewards = []
    trajs = split_into_trajectories(observations, dones_float)
    for traj in trajs:
        obs = np.array([t[0] for t in traj])
        new_rewards.extend(env.get_reward(obs[None], env.sample_mode())[0])
    new_rewards = np.array(new_rewards)
    normalised_rewards = (new_rewards - new_rewards.min()) / (new_rewards.max() - new_rewards.min())
    return normalised_rewards
