import os

import numpy as np
import torch

from jaxrl_m.learners.d4rl_utils import get_dataset, split_into_trajectories


def get_reward_model(model_type, ckpt):
    if os.path.isdir(ckpt):
        models = [
            f
            for f in os.listdir(ckpt)
            if os.path.isfile(os.path.join(ckpt, f)) and f.startswith("model_")
        ]
        max_epoch = -1
        max_epoch_model = None

        for model in models:
            try:
                epoch = int(model.split("_")[1].split(".")[0])
                if epoch > max_epoch:
                    max_epoch = epoch
                    max_epoch_model = model
            except ValueError:
                pass
        ckpt = os.path.join(ckpt, max_epoch_model)

    print(f"Loading {model_type} reward model")
    reward_model = torch.load(ckpt)
    return reward_model.to("cuda")


def relabel_rewards(
    env, dataset, model_type, reward_model, fix_latent=False, debug=False, label_freq=10
):
    observations = dataset["observations"]
    dones_float = dataset["dones_float"]
    new_rewards = []
    new_obs = []
    trajs = split_into_trajectories(observations, dones_float)
    if fix_latent:
        assert NotImplementedError
    else:
        z = reward_model.sample_prior(size=1)

    for i, traj in enumerate(trajs):
        obs = np.array([t[0] for t in traj])
        obs = torch.from_numpy(obs).float().to(next(reward_model.parameters()).device)
        if model_type == "MLP":
            rewards = reward_model.get_reward(obs)
        elif model_type == "Categorical" or model_type == "MeanVar":
            rewards = reward_model.sample_reward(obs)
        else:
            if i % label_freq == 0:
                if fix_latent:
                    assert NotImplementedError
                else:
                    z = reward_model.sample_prior(size=1)
            with torch.no_grad():
                z = (
                    torch.tensor(z)
                    .repeat(obs.shape[0], 1)
                    .float()
                    .to(next(reward_model.parameters()).device)
                )
                obs = torch.cat([obs, z], dim=-1)
                rewards = reward_model.get_reward(obs)

            # if debug:
            #     n = env.get_num_modes()
            #     temp_z = z[0, None].repeat(n, 0).view(n, -1, z.shape[-1]).cpu().numpy()
            #     fig1 = plot_z(env, reward_model, temp_z)
            #     fig2 = plot_train_values(
            #         obs[:, 0:2].cpu().numpy(), rewards.cpu().numpy()
            #     )
            #     wandb.log(
            #         dict(z_plot=wandb.Image(fig1), train_values=wandb.Image(fig2))
            #     )
            # plt.close(fig1)
            # plt.close(fig2)
        new_rewards.append(rewards.flatten().cpu().numpy())
        new_obs.append(obs.cpu().numpy())
    new_rewards = np.concatenate(new_rewards)
    new_obs = np.concatenate(new_obs)
    new_dataset = dict(
        observations=new_obs,
        actions=dataset["actions"],
        rewards=new_rewards,
        dones=dataset["dones"],
        infos=dataset["infos"],
    )
    return new_dataset
