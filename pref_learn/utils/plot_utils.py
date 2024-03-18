import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
import torch
import wandb

from pref_learn.models.utils import get_all_posterior, get_biased


def plot_reward(ax, obs, r, no_norm=False):
    if not no_norm:
        r = (r - r.min()) / (r.max() - r.min())
        r = np.exp(r / 0.1)
        r = (r - r.min()) / (r.max() - r.min())
    sc = ax.scatter(obs[:, 0], obs[:, 1], c=r)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("r(s)")


def plot_observations(observations, base_path):
    if observations.shape[1] > 2:
        observations = PCA(n_components=2).fit_transform(observations)
    plt.figure()
    plt.scatter(observations[:, 0], observations[:, 1], s=1)
    plt.savefig(base_path.replace("queries", "observation_plot"))


def plot_goals(env, ax, target_p, scale):
    if hasattr(env, "plot_goals"):
        env.plot_goals(ax, scale)
    else:
        ax.scatter(x=target_p[0], y=target_p[1], color="red", s=100)


def plot_observation_rewards(obs, r, no_norm=False):
    fig, ax = plt.subplots()
    plot_reward(ax, obs, r)
    plt.close(fig)
    return fig


def plot_mlp(env, reward_model):
    obs = env.get_obs_grid()
    obs_copy = np.copy(obs)
    obs = torch.from_numpy(obs).float().to(next(reward_model.parameters()).device)
    r = reward_model.get_reward(obs).detach().cpu().numpy()

    fig, ax = plt.subplots()
    plot_reward(ax, obs_copy, r)
    env.plot_goals(ax)
    plt.title(f"reward model")

    plot_dict = dict(reward_plot=wandb.Image(fig))
    plt.close(fig)
    if hasattr(env, "plot_gt"):
        plot_dict["gt"] = env.plot_gt(wandb_log=True)
    return plot_dict


def plot_classifier(env, reward_model, dataset):
    obs = env.get_obs_grid()
    obs_copy = np.copy(obs)
    obs = torch.from_numpy(obs).float().to(next(reward_model.parameters()).device)

    comparison_obs = dataset.get_mode_data(batch_size=100)["observations"]
    comparison_obs = comparison_obs.reshape(-1, comparison_obs.shape[-1])
    comparison_obs = (
        torch.from_numpy(comparison_obs)
        .float()
        .to(next(reward_model.parameters()).device)
    )

    r = reward_model.get_reward(obs, comparison_obs).detach().cpu().numpy()

    fig, ax = plt.subplots()
    plot_reward(ax, obs_copy, r, no_norm=True)
    env.plot_goals(ax)
    plt.title(f"reward model")

    plot_dict = dict(reward_plot=wandb.Image(fig))
    plt.close(fig)
    if hasattr(env, "plot_gt"):
        plot_dict["gt"] = env.plot_gt(wandb_log=True)
    return plot_dict

def plot_mlp_samples(env, reward_model, samples=4):
    obs = env.get_obs_grid()
    obs_copy = np.copy(obs)
    obs = torch.from_numpy(obs).float().to(next(reward_model.parameters()).device)

    plot_dict = plot_mlp(env, reward_model)
    fig, axs = plt.subplots(samples // 2, 2, figsize=(20, 16))
    for i, ax in enumerate(axs.flatten()):
        plot_reward(
            ax, obs_copy, reward_model.sample_reward(obs).detach().cpu().numpy()
        )
        ax.set_title(f"reward model sample: {i}")

    plot_dict["reward_samples"] = wandb.Image(fig)
    plt.close(fig)

    r = reward_model.get_variance(obs).detach().cpu().numpy()
    fig, ax = plt.subplots()
    # r = reward_model.sample_reward(obs).detach().cpu().numpy()
    plot_reward(ax, obs_copy, r)
    env.plot_goals(ax)
    plt.title(f"reward model variance")
    plot_dict["reward_variance"] = wandb.Image(fig)
    plt.close(fig)
    return plot_dict


def plot_prior(reward_model):
    with torch.no_grad():
        fig = plt.figure()
        x = np.linspace(-10, 10, 1000)
        for i in range(reward_model.latent_dim):
            y = stats.norm(
                reward_model.mean[i].cpu(), np.exp(reward_model.log_var[i].cpu() * 0.5)
            ).pdf(x)
            plt.plot(x, y)
        plt.title("Learned prior")
    plt.close(fig)
    return dict(prior_plot=wandb.Image(fig))


def plot_latents(env, reward_model, dataset):
    if reward_model.flow_prior:
        fig, axs = plt.subplots(1, 2, figsize=(10, 8))
        ax1 = axs[0]
        ax2 = axs[1]
    else:
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))
        ax1 = axs

    modes_n = env.get_num_modes()
    latents = get_all_posterior(env, reward_model, dataset, 128)
    for mode_n in range(modes_n):
        z = latents[mode_n]
        X_embedded = PCA(n_components=2).fit_transform(z)
        ax1.scatter(X_embedded[:, 0], X_embedded[:, 1], c=f"C{mode_n}")

        if reward_model.flow_prior:
            transformed_z = (
                reward_model.flow(
                    torch.from_numpy(z)
                    .float()
                    .to(next(reward_model.parameters()).device)
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
            X_embedded = PCA(n_components=2).fit_transform(transformed_z)
            ax2.scatter(X_embedded[:, 0], X_embedded[:, 1], c=f"C{mode_n}")
    ax1.set_title("Latent embeddings")
    if reward_model.flow_prior:
        ax2.set_title("Transformed latent embeddings")
    plt.close(fig)
    return dict(latent_plot=wandb.Image(fig))

def plot_z(obs, env, reward_model, latents, comp_obs=None):
    assert latents.shape[0] == env.get_num_modes()
    assert latents.shape[2] == reward_model.latent_dim
    num_samples = latents.shape[1]
    modes_n = env.get_num_modes()
    fig, axs = plt.subplots(modes_n, num_samples, figsize=(5, 8))
    obs_copy = np.copy(obs)
    obs = torch.from_numpy(obs).float().to(next(reward_model.parameters()).device)

    axs = axs.flatten()
    for mode_n in range(modes_n):
        for i in range(num_samples):
            ax = axs[mode_n * num_samples + i]
            z = latents[None, mode_n, i]
            z = np.repeat(z, obs.shape[0], axis=0)
            z = torch.from_numpy(z).float().to(next(reward_model.parameters()).device)
            if comp_obs is not None:
                r = reward_model.decode(obs, comp_obs, z).detach().cpu().numpy()
                plot_reward(ax, obs_copy, r, no_norm=True)
            else:
                r = (
                    reward_model.decode(obs, z).detach().cpu().numpy()
                )  # .reshape((NX, NY))
                plot_reward(ax, obs_copy, r)
            ax.set_title(f"Mode {mode_n}")
            env.plot_goals(ax)

    plt.tight_layout()
    plt.close(fig)
    return wandb.Image(fig)

def plot_vae(env, reward_model, dataset, classifier=False, num_samples=4):
    obs = env.get_obs_grid()
    comp_obs = None
    if classifier:
        comp_obs = dataset.get_mode_data(batch_size=50)["observations"]
        comp_obs = comp_obs.reshape(-1, obs.shape[-1])
        comp_obs = (
            torch.from_numpy(comp_obs)
            .float()
            .to(next(reward_model.parameters()).device)
        )
    plot_dict = plot_prior(reward_model)
    plot_dict.update(plot_latents(env, reward_model, dataset))

    prior_latents = (
        reward_model.sample_prior(size=num_samples)
        .view(num_samples // 2, -1, reward_model.latent_dim)
        .detach()
        .cpu()
        .numpy()
    )

    posterior_latents = get_all_posterior(env, reward_model, dataset, num_samples)
    biased_latents = get_biased(env, reward_model)

    reward_model.update_posteriors(posterior_latents, biased_latents)

    plot_dict["prior"] = plot_z(
        obs, env, reward_model, prior_latents, comp_obs=comp_obs
    )
    plot_dict["posterior"] = plot_z(
        obs, env, reward_model, posterior_latents, comp_obs=comp_obs
    )
    plot_dict["biased"] = plot_z(
        obs, env, reward_model, biased_latents, comp_obs=comp_obs
    )
    if hasattr(env, "plot_gt"):
        plot_dict["gt"] = env.plot_gt(wandb_log=True)
    return plot_dict
