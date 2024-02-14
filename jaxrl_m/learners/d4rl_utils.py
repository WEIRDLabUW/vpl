import d4rl
import gym
import numpy as np
from tqdm import tqdm
import torch

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
    append_goal: bool = False,
    use_reward_model: bool = False,
    model_type: str = "",
    reward_model: torch.nn.Module = None,
    fixed_mode: int = -1
):
    dataset = d4rl.qlearning_dataset(env)
    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    imputed_next_observations = np.roll(dataset['observations'], -1, axis=0)
    same_obs = np.all(np.isclose(imputed_next_observations, dataset['next_observations'], atol=1e-5), axis=-1)
    dones_float = 1.0 - same_obs.astype(np.float32)
    dones_float[-1] = 1
    
    dones_float = dataset["terminals"].astype(np.float32)
    dataset = {
        "observations": dataset["observations"],
        "actions": dataset["actions"],
        "rewards": dataset["rewards"],
        "masks": 1.0 - dataset["terminals"],
        "dones_float": dones_float,
        "next_observations": dataset["next_observations"],
    }

    if use_reward_model:
        assert reward_model is not None
        dataset = relabel_rewards_with_model(env, dataset, model_type, reward_model, append_goal, fixed_mode)
    elif hasattr(env, "relabel_offline_reward") and env.relabel_offline_reward:
        dataset = relabel_rewards_with_env(env, dataset, append_goal)

    dataset = {k: v.astype(np.float32) for k, v in dataset.items()}
    return Dataset(dataset)

def new_get_trj_idx(dataset):
    dones_float = dataset["dones_float"]
    # If the dones are dropped just split uniformly to relabel the trajectories
    if len(np.argwhere(dones_float)) < 2:
        idx = np.arange(0, len(dones_float), 1000)
        dones_float[idx] = 1.0
        dones_float[-1] = 1.0

    N = dataset["rewards"].shape[0]
    episode_step = 0
    start_idx, data_idx = 0, 0
    trj_idx_list = []
    for i in range(N - 1):
        done_bool = dones_float[i]
        if done_bool:
            episode_step = 0
            trj_idx_list.append([start_idx, data_idx])
            start_idx = data_idx + 1

        episode_step += 1
        data_idx += 1

    trj_idx_list.append([start_idx, data_idx])
    return trj_idx_list

def relabel_rewards_with_model(env, dataset, model_type, reward_model, append_goal, fixed_mode):
    observation_dim = env.observation_space.shape[-1]
    if hasattr(env, "reward_observation_space"):
        observation_dim = env.reward_observation_space.shape[-1]

    obs_list = []
    next_obs_list = []
    new_rewards = np.zeros_like(dataset["rewards"])

    traj_idx = new_get_trj_idx(dataset)

    for start, end in traj_idx:
        obs = dataset["observations"][start : end + 1]
        next_obs = dataset["next_observations"][start : end + 1]

        input_obs = (
            torch.from_numpy(obs[:, :observation_dim])
            .float()
            .to(next(reward_model.parameters()).device)
        )
        if model_type == "MLP":
            with torch.no_grad():
                rewards = reward_model.get_reward(input_obs)
        elif model_type == "Categorical" or model_type == "MeanVar":
            with torch.no_grad():
                rewards = reward_model.sample_reward(input_obs)
        else:
            with torch.no_grad():
                idx = fixed_mode
                if idx < 0:
                    idx = env.sample_mode()
                z = reward_model.biased_latents[idx]
                z = (
                    torch.tensor(z)
                    .repeat(obs.shape[0], 1)
                    .float()
                    .to(next(reward_model.parameters()).device)
                )
                rewards = reward_model.get_reward(torch.cat([input_obs, z], dim=-1))

                if append_goal:
                    # Changing the latent 
                    z = torch.ones_like(z[:, :1]) * idx
                
                # Appending the task vector to the observation
                obs_list.append(np.concatenate([obs, z.cpu().numpy()], axis=-1))
                next_obs_list.append(
                    np.concatenate([next_obs, z.cpu().numpy()], axis=-1)
                )

        new_rewards[start : end + 1] = rewards.cpu().numpy()[:, 0]

    if len(obs_list) > 0:
        dataset["observations"] = np.concatenate(obs_list, axis=0)
        dataset["next_observations"] = np.concatenate(next_obs_list, axis=0)
    dataset["rewards"] = (new_rewards - new_rewards.min()) / (
        new_rewards.max() - new_rewards.min()
    )
    return dataset

def relabel_rewards_with_env(env, dataset, append_goal):
    new_rewards = []
    obs_list = []
    next_obs_list = []
    traj_idx = new_get_trj_idx(dataset)
    modes = []
    for (start, end) in traj_idx:
        obs = dataset["observations"][start:end+1]
        next_obs = dataset["next_observations"][start:end+1]
        mode = env.sample_mode()
        new_rewards.append(env.compute_reward(obs[None], mode)[0])
        modes.append(mode)
        if append_goal:
            obs_list.append(
                np.concatenate([obs, np.ones_like(obs[:, :1]) * mode], axis=-1)
            )
            next_obs_list.append(
                np.concatenate([next_obs, np.ones_like(next_obs[:, :1]) * mode], axis=-1)
            )
    print("Mean mode:", np.mean(modes))
    new_rewards = np.concatenate(new_rewards)
    normalised_rewards = (new_rewards - new_rewards.min()) / (new_rewards.max() - new_rewards.min())
    dataset["rewards"] = normalised_rewards
    if append_goal:
        dataset["observations"] = np.concatenate(obs_list, axis=0)
        dataset["next_observations"] = np.concatenate(next_obs_list, axis=0)
    return dataset
