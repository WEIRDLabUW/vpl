import torch
import jaxrl_m.envs
import gym
from pref_learn.models.utils import get_datasets
from pref_learn.utils.data_utils import get_labels
import os
import scipy
import matplotlib.pyplot as plt
import numpy as np
from d4rl.pointmaze.gridcraft import grid_env
from d4rl.pointmaze.gridcraft import grid_spec
import tqdm
from functools import partial
from pref_learn.models.utils import get_datasets

def gridify_state(state):
    return (int(round(state[0])), int(round(state[1])))


def reward_norm(maze_env, obs, r):
    obs = obs.detach().cpu().numpy()
    obs_grid = np.array([gridify_state(o) for o in obs])
    r = np.array(
        [
            r[i] if maze_env.gs[obs_grid[i]] != grid_spec.WALL else r.min()
            for i in range(len(r))
        ]
    )
    return r


def plot_z(obs, env, reward_model, latent, ax, mode, maze_env):
    obs_copy = np.copy(obs)
    obs = torch.from_numpy(obs).float().to(next(reward_model.parameters()).device)
    latent = np.repeat(latent[None], obs.shape[0], axis=0)
    latent = torch.from_numpy(latent).float().to(next(reward_model.parameters()).device)
    r = reward_model.decode(obs, latent).detach().cpu().numpy()  # .reshape((NX, NY))
    r = reward_norm(maze_env, obs, r)
    r_gt = env.compute_reward(obs_copy[None], mode)[0]
    r_gt = reward_norm(maze_env, obs, r_gt)
    corr_coef, _ = scipy.stats.pearsonr(r, r_gt)
    sc = ax.scatter(obs_copy[:, 0], obs_copy[:, 1], c=r)
    plt.colorbar(sc, ax=ax)
    ax.set_title(f"Goal {mode}, corr: {corr_coef:.3f}")
    ax.set_axis_off()
    env.plot_goals(ax)


def get_latent(obs1, obs2, env, reward_model, mode):
    obs_dim = obs1.shape[-1]
    obs1 = obs1.reshape(
        -1, reward_model.annotation_size, reward_model.size_segment, obs_dim
    )
    obs2 = obs2.reshape(
        -1, reward_model.annotation_size, reward_model.size_segment, obs_dim
    )
    seg_reward_1 = env.compute_reward(obs1, mode)
    seg_reward_2 = env.compute_reward(obs2, mode)

    seg_reward_1 = seg_reward_1.reshape(
        1, reward_model.annotation_size, reward_model.size_segment, -1
    )
    seg_reward_2 = seg_reward_2.reshape(
        1, reward_model.annotation_size, reward_model.size_segment, -1
    )

    labels = get_labels(seg_reward_1, seg_reward_2)
    device = next(reward_model.parameters()).device
    obs1 = torch.from_numpy(obs1).float().to(device)
    obs2 = torch.from_numpy(obs2).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)
    with torch.no_grad():
        mean, logvar = reward_model.encode(obs1, obs2, labels)
    return mean.squeeze().cpu().numpy(), logvar.squeeze().cpu().numpy()


def get_active_learning_dataset(env, samples, reward_model):
    env.set_biased_mode("random")
    pairs = []
    for _ in range(samples):
        obs1, obs2 = env.get_biased_data(reward_model.annotation_size)
        pairs.append((obs1, obs2))
    return pairs


def binary(x, bits):
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def get_information_gain(obs1, obs2, reward_model, num_samples, new_objective=False): # S x B
    obs1 = torch.from_numpy(obs1).float().to(next(reward_model.parameters()).device)
    obs2 = torch.from_numpy(obs2).float().to(next(reward_model.parameters()).device)
    num_labels = obs1.shape[0]
    num_samples = min(num_samples, 2**num_labels)
    labels = np.random.choice(np.arange(2**num_labels), num_samples, replace=False)
    labels = (
        binary(torch.tensor(labels), num_labels)
        .float()
        .to(next(reward_model.parameters()).device)
    )
    # labels = (
    #     torch.random.choice([0, 1], (num_samples, num_labels))
    #     .float()
    #     .to(next(reward_model.parameters()).device)
    # )

    obs1 = torch.repeat_interleave(obs1[None], num_samples, 0) # NxSxB
    obs2 = torch.repeat_interleave(obs2[None], num_samples, 0)
    # encoder_input = torch.cat([obs1, obs2, labels[:, :, None]], -1)
    with torch.no_grad():
        _, logvar = reward_model.encode(obs1, obs2, labels[:, :, None])
        var = torch.exp(logvar)   
        term2 = torch.log(2 * np.pi * var).sum(-1).mean()
        term1=torch.log(2 * np.pi * var.sum(-1).mean())
        inf=term1-term2

    return -inf.cpu()

def get_posterior_entropy(obs1, obs2, reward_model, num_samples, new_objective=False): # S x B
    obs1 = torch.from_numpy(obs1).float().to(next(reward_model.parameters()).device)
    obs2 = torch.from_numpy(obs2).float().to(next(reward_model.parameters()).device)
    num_labels = obs1.shape[0]
    num_samples = min(num_samples, 2**num_labels)
    labels = np.random.choice(np.arange(2**num_labels), num_samples, replace=False)
    labels = (
        binary(torch.tensor(labels), num_labels)
        .float()
        .to(next(reward_model.parameters()).device)
    )
    # labels = (
    #     torch.random.choice([0, 1], (num_samples, num_labels))
    #     .float()
    #     .to(next(reward_model.parameters()).device)
    # )

    obs1 = torch.repeat_interleave(obs1[None], num_samples, 0) # NxSxB
    obs2 = torch.repeat_interleave(obs2[None], num_samples, 0)
    with torch.no_grad():
        _, logvar = reward_model.encode(obs1, obs2, labels[:, :, None])
        var = torch.exp(logvar)
        entropy = 0.5 * (1 + torch.log(2 * np.pi * var)).sum(-1)
    return entropy.min().cpu()

def get_min_entropy_dataset(
    preference_dataset,
    sample_size,
    num_of_sets,
    entropy_fn,
):
    obs1, obs2 = sample_dataset(preference_dataset, sample_size)
    pairs = zip(obs1, obs2)

    entropy = [
        (obs1, obs2, entropy_fn(obs1, obs2))
        for obs1, obs2 in tqdm.tqdm(pairs)
    ]
    entropy = sorted(entropy, key=lambda x: x[2])
    obs1 = np.array([x[0] for x in entropy[:num_of_sets]])
    obs2 = np.array([x[1] for x in entropy[:num_of_sets]])
    if True:
        plt.figure()
        plt.hist([e[2] for e in entropy])
        plt.savefig("inf_gain_hist")
    return (obs1, obs2)

def sample_dataset(preference_dataset, num_of_sets):
    dataset, _ = preference_dataset.get_mode_data(num_of_sets)
    obs1 = dataset['observations'][:, :, 0]
    obs2 = dataset['observations_2'][:, :, 0]
    return (obs1, obs2)

def get_queries(
    env,
    reward_model,
    preference_dataset,
    sampling_method,
    num_of_sets,
    sample_size=1000,
    num_samples=50,
):
    if sampling_method == "random":
        return sample_dataset(preference_dataset, num_of_sets)
    # elif sampling_method == "grid":
    #     return sample_dataset(env, reward_model, "grid", num_of_sets=num_of_sets)
    elif sampling_method == "posterior":
        return get_min_entropy_dataset(
            preference_dataset,
            sample_size,
            num_of_sets,
            entropy_fn=partial(get_posterior_entropy, reward_model=reward_model, num_samples=num_samples)
        )
    elif sampling_method == "information_gain":
        return get_min_entropy_dataset(
            preference_dataset,
            sample_size,
            num_of_sets,
            entropy_fn=partial(get_information_gain, reward_model=reward_model, num_samples=num_samples)
        )
    else:
        raise ValueError(f"Invalid sampling method: {sampling_method}")

def plot_obs(obs1, obs2, title):
    plt.figure()
    for i in range(obs1.shape[0]):
        plt.plot([obs1[i, 0], obs2[i, 0]], [obs1[i, 1], obs2[i, 1]], color=f"C{i}")
    plt.savefig(title)
    
if __name__ == "__main__":
    env_id = "maze2d-twogoals-multimodal-v0"
    preference_dataset_path = "data/maze_queries_num5000_q1_s16"
    reward_model_path = "data/test_reward_model"
    gym_env = gym.make(env_id)
    if hasattr(gym_env, "reward_observation_space"):
        observation_dim = gym_env.reward_observation_space.shape[0]
    else:
        observation_dim = gym_env.observation_space.shape[0]
    action_dim = gym_env.action_space.shape[0]

    def load_reward_model(ckpt):
        with open(os.path.join(ckpt, f"best_model.pt"), "rb") as f:
            reward_model = torch.load(f)
        return reward_model

    env = gym_env
    reward_model = load_reward_model(reward_model_path)
    
    (
        _,
            _,
            _,
            preference_dataset,
            set_len,
            _,
            _,
    ) = get_datasets(
        preference_dataset_path,
        observation_dim,
        env.action_space.shape[0],
        256,
        reward_model.annotation_size
    )
    
    obs1, obs2 = get_queries(env, reward_model, preference_dataset, "random", 1)
    plot_obs(obs1, obs2, "random_sampling")
    
    obs1, obs2 = get_queries(env, reward_model, preference_dataset, "information_gain", 1, 5000)
    print(obs1.shape, obs2.shape)