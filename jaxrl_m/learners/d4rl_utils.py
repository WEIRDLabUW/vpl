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
    add_mode: bool = False,
):
    dataset = d4rl.qlearning_dataset(env)
    terminals = dataset["terminals"]
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
        print("Relabelling rewards and add mode is:", add_mode)
        dataset["rewards"], dataset["observations"], dataset["next_observations"] = relabel_rewards(
            env, dataset["observations"], dataset["next_observations"], terminals, add_mode
        )

    dataset = {k: v.astype(np.float32) for k, v in dataset.items()}
    return Dataset(dataset)


def split_into_trajectories(observations, next_observations, dones_float):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], next_observations[i], dones_float[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def relabel_rewards(env, observations, next_observations, dones_float, add_mode):
    new_rewards = []
    new_observations = []
    new_next_observations = []
    trajs = split_into_trajectories(observations, next_observations, dones_float)
    modes = []
    for traj in trajs:
        obs = np.array([t[0] for t in traj])
        next_obs = np.array([t[1] for t in traj])
        mode = env.sample_mode()
        new_rewards.extend(env.get_reward(obs[None], mode)[0])
        modes.append(mode)
        # import pdb; pdb.set_trace()
        if add_mode:
            new_observations.append(
                np.concatenate([obs, np.ones_like(obs[:, :4]) * mode], axis=-1)
            )
            new_next_observations.append(
                np.concatenate([next_obs, np.ones_like(next_obs[:, :4]) * mode], axis=-1)
            )
    print("Mean mode:", np.mean(modes))
    # import pdb; pdb.set_trace()
    new_rewards = np.array(new_rewards)
    normalised_rewards = (new_rewards - new_rewards.min()) / (new_rewards.max() - new_rewards.min())
    if add_mode:
        new_observations = np.concatenate(new_observations, axis=0)
        new_next_observations = np.concatenate(new_next_observations, axis=0)
    else:
        new_observations = observations
        new_next_observations = next_observations
    # print(observations.shape, new_observations.shape)
    # import pdb; pdb.set_trace()
    assert observations.shape[0] == new_observations.shape[0]
    return normalised_rewards, new_observations, new_next_observations
