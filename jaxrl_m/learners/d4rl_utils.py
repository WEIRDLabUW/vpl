import d4rl
import gym
import numpy as np

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
        dataset["rewards"] = relabel_rewards(
            env, dataset["observations"], dataset["rewards"]
        )

    dataset = {k: v.astype(np.float32) for k, v in dataset.items()}
    return Dataset(dataset)


def relabel_rewards(env, observations, rewards, batch_size=256):
    new_rewards = np.zeros_like(rewards)
    for i in range(0, len(observations), batch_size):
        obs = observations[i : i + batch_size][None]
        new_rewards[i : i + batch_size] = env.get_reward(obs)[0]
    return new_rewards
