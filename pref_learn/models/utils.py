import math
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pref_learn.utils.data_utils import get_labels


def get_datasets(query_path, observation_dim, action_dim, batch_size):
    with open(query_path, "rb") as fp:
        batch = pickle.load(fp)

    batch["observations"] = batch["observations"][..., :observation_dim]
    batch["observations_2"] = batch["observations_2"][..., :observation_dim]
    assert batch["actions"].shape[-1] == action_dim
    
    eval_data_size = int(0.1 * len(batch["observations"]))
    train_data_size = len(batch["observations"]) - eval_data_size

    train_batch = {
        "observations": batch["observations"][:train_data_size],
        "actions": batch["actions"][:train_data_size],
        "observations_2": batch["observations_2"][:train_data_size],
        "actions_2": batch["actions_2"][:train_data_size],
        "labels": batch["labels"][:train_data_size],
    }

    eval_batch = {
        "observations": batch["observations"][train_data_size:],
        "actions": batch["actions"][train_data_size:],
        "observations_2": batch["observations_2"][train_data_size:],
        "actions_2": batch["actions_2"][train_data_size:],
        "labels": batch["labels"][train_data_size:],
    }

    train_dataset = PreferenceDataset(train_batch)
    eval_dataset = PreferenceDataset(eval_batch)
    kwargs = {"num_workers": 1, "pin_memory": True}
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = DataLoader(
        dataset=eval_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )

    _, len_set, len_query, obs_dim = batch["observations"].shape
    return (
        train_loader,
        test_loader,
        train_dataset,
        eval_dataset,
        len_set,
        len_query,
        obs_dim,
    )


class PreferenceDataset(Dataset):
    def __init__(self, pref_dataset):
        self.pref_dataset = pref_dataset

    def __len__(self):
        return len(self.pref_dataset["observations"])

    def __getitem__(self, idx):
        observations = self.pref_dataset["observations"][idx]
        observations_2 = self.pref_dataset["observations_2"][idx]
        labels = self.pref_dataset["labels"][idx]
        return dict(
            observations=observations, observations_2=observations_2, labels=labels
        )

    def get_mode_data(self, batch_size):
        idxs = np.random.choice(range(len(self)), size=batch_size, replace=False)
        return dict(
            observations=self.pref_dataset["observations"][idxs],
            observations_2=self.pref_dataset["observations_2"][idxs],
        )


class Annealer:
    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = "none"
            self.baseline = 0.0

    def __call__(self, kld):
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == "linear":
            y = self.current_step / self.total_steps
        elif self.shape == "cosine":
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == "logistic":
            exponent = (self.total_steps / 2) - self.current_step
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == "none":
            y = 1.0
        else:
            raise ValueError(
                "Invalid shape for annealing function. Must be linear, cosine, or logistic."
            )
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError(
                "Cyclical_setter method requires boolean argument (True/False)"
            )
        else:
            self.cyclical = value
        return


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def get_latent(batch, env, reward_model, mode, num_samples):
    # obs_dim = env.reward_observation_space.shape[0]
    obs1 = batch["observations"]
    obs2 = batch["observations_2"]
    obs_dim = obs1.shape[-1]
    seg_reward_1 = env.compute_reward(obs1.reshape(-1, reward_model.size_segment, obs_dim), mode)
    seg_reward_2 = env.compute_reward(obs2.reshape(-1, reward_model.size_segment, obs_dim), mode)

    seg_reward_1 = seg_reward_1.reshape(
        num_samples, reward_model.annotation_size, reward_model.size_segment, -1
    )
    seg_reward_2 = seg_reward_2.reshape(
        num_samples, reward_model.annotation_size, reward_model.size_segment, -1
    )

    labels = get_labels(seg_reward_1, seg_reward_2)
    device = next(reward_model.parameters()).device
    obs1 = torch.from_numpy(obs1).float().to(device)
    obs2 = torch.from_numpy(obs2).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)
    with torch.no_grad():
        mean, _ = reward_model.encode(obs1, obs2, labels)
    return mean.cpu().numpy()


def get_posterior(env, reward_model, dataset, mode, num_samples):
    batch = dataset.get_mode_data(num_samples)
    return get_latent(batch, env, reward_model, mode, num_samples)


def get_all_posterior(env, reward_model, dataset, num_samples):
    means = []
    for mode in range(env.get_num_modes()):
        means.append(get_posterior(env, reward_model, dataset, mode, num_samples))
    return np.stack(means, axis=0)


def get_biased(env, reward_model):
    assert hasattr(env, "get_biased_data")
    assert reward_model.size_segment == 1
    means = []
    obs1, obs2 = env.get_biased_data(reward_model.annotation_size)
    batch = dict(
        observations=obs1[None, :, None],
        observations_2=obs2[None, :, None],
    )
    # import pdb; pdb.set_trace()
    for mode in range(env.get_num_modes()):
        means.append(get_latent(batch, env, reward_model, mode, 1))
    return np.stack(means, axis=0)
