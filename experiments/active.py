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


def gridify_state(state):
    return (int(round(state[0])), int(round(state[1])))


def reward_norm(maze_env, obs, r):
    r = r
    # r = np.exp(r / 0.1)
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
    # import pdb

    # pdb.set_trace()
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


def get_posterior_entropy(reward_model, env, obs1, obs2, num_samples):
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

    obs1 = torch.repeat_interleave(obs1[None], num_samples, 0)
    obs2 = torch.repeat_interleave(obs2[None], num_samples, 0)
    # encoder_input = torch.cat([obs1, obs2, labels[:, :, None]], -1)

    with torch.no_grad():
        mean, logvar = reward_model.encode(obs1, obs2, labels[:, :, None])
        var = torch.exp(logvar)
        entropy = 0.5 * (1 + torch.log(2 * np.pi * var)).sum(-1)
    return entropy.mean().item()


def get_reward_model_entropy(reward_model, env, obs1, obs2, num_samples):
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
    # import pdb; pdb.set_trace()
    obs1 = torch.repeat_interleave(obs1[None], num_samples, 0)
    obs2 = torch.repeat_interleave(obs2[None], num_samples, 0)
    # encoder_input = torch.cat([obs1, obs2, labels[:, :, None]], -1)
    with torch.no_grad():
        means, logvars = reward_model.encode(obs1, obs2, labels[:, :, None])

    obs = env.get_obs_grid()
    obs = obs.reshape(-1, obs.shape[-1])
    obs = torch.from_numpy(obs).float().to(next(reward_model.parameters()).device)
    rewards = []
    for i in range(num_samples):
        mean = means[None, i].repeat(obs.shape[0], 1)  # O x L
        decoder_input = torch.cat([obs, mean], -1)  # O x (L + D)
        with torch.no_grad():
            reward = reward_model.get_reward(decoder_input)
        # reward = reward_model.get_reward(decoder_input)  # O
        # reward = reward_model(obs1, obs2, label)
        rewards.append(reward.cpu())

    entropy = []
    rewards = torch.stack(rewards, 0)
    for o in range(rewards.shape[1]):
        # import pdb; pdb.set_trace()
        reward = rewards[:, o]
        histogram = torch.histogram(reward, bins=100, density=False)[0]
        histogram = histogram / histogram.sum()
        entropy.append(scipy.stats.entropy(histogram))
    return np.mean(entropy)


# def get_reward_model_entropy_cov(reward_model, env, obs1, obs2, num_samples=1000, num_samples_per_latent=100):
#     obs1 = torch.from_numpy(obs1).float().to(next(reward_model.parameters()).device)
#     obs2 = torch.from_numpy(obs2).float().to(next(reward_model.parameters()).device)
#     num_labels = obs1.shape[0]
#     labels = torch.random.choice([0, 1], (num_samples, num_labels)).float().to(next(reward_model.parameters()).device)
#     encoder_input = torch.cat([obs1, obs2, labels[:, :, None]], -1)
#     with torch.no_grad():
#         means, logvars = reward_model.encode(encoder_input)


#     obs = env.get_obs_grid()
#     obs = obs.reshape(-1, obs.shape[-1])
#     obs = torch.from_numpy(obs).float().to(next(reward_model.parameters()).device)
#     rewards = []
#     for i in range(num_samples):
#         mean = means[None, i].repeat(obs.shape[0], 1) # O x L
#         decoder_input = torch.cat([obs, mean], -1) # O x (L + D)
#         reward = reward_model.decode(decoder_input) # O
#         # reward = reward_model(obs1, obs2, label)
#         rewards.append(reward)


#     entropy = []
#     rewards = torch.stack(rewards, 0)
#     for o in range(rewards.shape[1]):
#         reward = rewards[:, o]
#         histogram = torch.histogram(reward, bins=100, density=True)
#         entropy.append(scipy.stats.entropy(histogram))
#     return np.mean(entropy)
def get_min_entropy_dataset(
    env,
    entr_samples,
    reward_model,
    num_samples,
    dataset_size=1,
    entropy_fn=get_reward_model_entropy,
    **kwargs,
):
    pairs = get_active_learning_dataset(env, entr_samples, reward_model)
    entropy = []
    entropy = [
        (obs1, obs2, entropy_fn(reward_model, env, obs1, obs2, num_samples))
        for obs1, obs2 in tqdm.tqdm(pairs)
    ]
    entropy = sorted(entropy, key=lambda x: x[2])
    obs1 = np.array([x[0] for x in entropy[:dataset_size]])
    obs2 = np.array([x[1] for x in entropy[:dataset_size]])
    return (obs1, obs2)
    # min_entropy = float("inf")
    # for i in tqdm.tqdm(range(entr_samples)):
    #     obs1, obs2 = pairs[i]
    #     entropy = entropy_fn(reward_model, env, obs1, obs2, num_samples)
    #     if entropy < min_entropy:
    #         min_entropy = entropy
    #         min_entropy_pair = (obs1, obs2)
    # return min_entropy_pair


def sample_dataset(env, reward_model, biased_mode, dataset_size=1, **kwargs):
    env.set_biased_mode(biased_mode)
    pairs = [
        env.get_biased_data(reward_model.annotation_size) for _ in range(dataset_size)
    ]
    obs1 = np.array([x[0] for x in pairs])
    obs2 = np.array([x[1] for x in pairs])
    # obs1, obs2 = env.get_biased_data(reward_model.annotation_size)
    return (obs1, obs2)


def plot_vae(env, maze_env, reward_model, sampling_mode, env_id, **kwargs):
    obs = env.get_obs_grid()
    obs = obs.reshape(-1, obs.shape[-1])

    if sampling_mode == "random":
        obs1, obs2 = sample_dataset(env, reward_model, "random")
    elif sampling_mode == "grid":
        obs1, obs2 = sample_dataset(env, reward_model, "grid")
    elif "active" in sampling_mode:
        obs1, obs2 = get_min_entropy_dataset(
            env, kwargs["entr_samples"], reward_model, kwargs["num_samples"]
        )
    else:
        obs1, obs2 = sample_dataset(env, reward_model, "equal")

    fig, axs = plt.subplots(9, 1, figsize=(5, 32))
    axs = axs.flatten()

    for i in range(obs1.shape[0]):
        axs[0].plot([obs1[i, 0], obs2[i, 0]], [obs1[i, 1], obs2[i, 1]], color=f"C{i}")
        axs[0].scatter(obs1[i, 0], obs1[i, 1], color="red", alpha=0.2)
        axs[0].scatter(obs2[i, 0], obs2[i, 1], color="blue", alpha=0.2)
    axs[0].set_title("Comparison Set")
    env.plot_goals(axs[0])

    for i in range(env.get_num_modes()):
        mean, logvar = get_latent(obs1, obs2, env, reward_model, i)
        for j in range(4):
            latent = np.random.normal(mean, np.exp(logvar / 2))
            plot_z(obs, env, reward_model, latent, axs[4 * i + j + 1], i, maze_env)
    fig.suptitle(f"Sampling Mode: {sampling_mode}")
    plt.tight_layout()
    plt.savefig(f"active_plots2/{env_id}-{sampling_mode}.png")


def get_queries(
    env,
    reward_model,
    sampling_method,
    dataset_size=1,
    entr_samples=1000,
    num_samples=50,
):
    if sampling_method == "random":
        return sample_dataset(env, reward_model, "random", dataset_size=dataset_size)
    elif sampling_method == "grid":
        return sample_dataset(env, reward_model, "grid", dataset_size=dataset_size)
    elif sampling_method == "active_posterior":
        return get_min_entropy_dataset(
            env,
            entr_samples,
            reward_model,
            num_samples,
            entropy_fn=get_posterior_entropy,
            dataset_size=dataset_size,
        )
    elif sampling_method == "active_reward":
        return get_min_entropy_dataset(
            env,
            entr_samples,
            reward_model,
            num_samples,
            entropy_fn=get_reward_model_entropy,
            dataset_size=dataset_size,
        )
    elif sampling_method == "equal":
        return sample_dataset(env, reward_model, "equal")
    else:
        raise ValueError(f"Invalid sampling method: {sampling_method}")


if __name__ == "__main__":
    env_id = "maze2d-twogoals-multimodal-v0"
    # env_id="maze2d-active-multimodal-v0"
    # dataset_path = f"/home/max/Distributional-Preference-Learning/vpl/pref_datasets/f{env_id}/relabelled_queries_num5000_q1_s4"
    gym_env = gym.make(env_id)
    if hasattr(gym_env, "reward_observation_space"):
        observation_dim = gym_env.reward_observation_space.shape[0]
    else:
        observation_dim = gym_env.observation_space.shape[0]
    action_dim = gym_env.action_space.shape[0]

    def load_reward_model(ckpt, seed=42):
        with open(os.path.join(ckpt, f"s{seed}/best_model.pt"), "rb") as f:
            reward_model = torch.load(f)
        return reward_model

    env = gym_env
    maze_env = grid_env.GridEnv(
        grid_spec.spec_from_string(env.unwrapped.env.str_maze_spec)
    )

    reward_model = load_reward_model(f"logs/{env_id}/VAE/ac_active")
    plot_vae(env, maze_env, reward_model, "random", env_id=env_id)
    plot_vae(env, maze_env, reward_model, "grid", env_id=env_id)
    plot_vae(env, maze_env, reward_model, "equal", env_id=env_id)

    plot_vae(
        env,
        maze_env,
        reward_model,
        "active_posterior",
        entr_samples=1000,
        num_samples=16,
        entropy_fn=get_posterior_entropy,
        env_id=env_id,
    )
    plot_vae(
        env,
        maze_env,
        reward_model,
        "active_reward",
        entr_samples=1000,
        num_samples=16,
        entropy_fn=get_reward_model_entropy,
        env_id=env_id,
    )
