import os
import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPModel, self).__init__()
        self.reward_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def get_reward(self, r):
        r = self.reward_model(r)
        return r

    def reconstruction_loss(self, x, x_hat):
        return nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")

    def accuracy(self, x, x_hat):
        predicted_class = (x_hat > 0.5).float()
        return torch.mean((predicted_class == x).float())

    def forward(self, s1, s2, y):
        r_hat1 = self.reward_model(s1).sum(dim=2)
        r_hat2 = self.reward_model(s2).sum(dim=2)

        p_hat = torch.nn.functional.sigmoid(r_hat1 - r_hat2).view(-1, 1)
        labels = y.view(-1, 1)

        loss = self.reconstruction_loss(labels, p_hat)
        accuracy = self.accuracy(labels, p_hat)

        metrics = {
            "loss": loss.item(),
            "reconstruction_loss": loss.item(),
            "accuracy": accuracy.item(),
        }

        return loss, metrics


class CategoricalModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_atoms=51,
        r_min=0,
        r_max=1,
        entropy_coeff=0.1,
        device="cuda",
    ):
        super(CategoricalModel, self).__init__()
        self.r_min = r_min
        self.r_max = r_max
        self.r_bins = torch.linspace(r_min, r_max, n_atoms).to(device)
        self.n_atoms = n_atoms
        self.entropy_coeff = entropy_coeff
        self.reward_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, n_atoms),
        )

    def get_reward(self, r):
        # import pdb; pdb.set_trace()
        probs = torch.nn.functional.softmax(self.reward_model(r), dim=-1)
        return torch.sum(probs * self.r_bins, dim=-1)

    def get_variance(self, r):
        # import pdb; pdb.set_trace()
        probs = torch.nn.functional.softmax(self.reward_model(r), dim=-1)
        mean = torch.sum(probs * self.r_bins, dim=-1)
        return torch.sum(probs * (self.r_bins - mean[:, None]) ** 2, dim=-1).sqrt()

    def sample_reward(self, r):
        probs = torch.nn.functional.softmax(self.reward_model(r), dim=-1)
        idxs = torch.multinomial(probs, 1)
        return self.r_bins[idxs]

    def reconstruction_loss(self, rewards_chosen, rewards_rejected):
        num_atoms = self.n_atoms
        device = rewards_chosen.device

        comparison_matrix = torch.empty(
            (num_atoms, num_atoms),
            device=device,
            dtype=rewards_chosen.dtype,
        )
        atom_values = torch.linspace(0, 1, num_atoms, device=device)
        comparison_matrix[:] = atom_values[None, :] > atom_values[:, None]
        comparison_matrix[atom_values[None, :] == atom_values[:, None]] = 0.5

        dist_rejected = rewards_rejected.softmax(1)
        dist_chosen = rewards_chosen.softmax(1)
        prob_chosen = ((dist_rejected @ comparison_matrix) * dist_chosen).sum(dim=1)
        loss = -prob_chosen.log()
        return loss.mean(), torch.mean((loss < np.log(2)).float())

    def forward(self, s1, s2, y):
        assert s1.shape[2] == 1 and len(s1.shape) == 4
        assert s2.shape[2] == 1 and len(s2.shape) == 4
        s1 = s1.squeeze(2)
        s2 = s2.squeeze(2)

        r_hat1 = self.reward_model(s1)
        r_hat2 = self.reward_model(s2)

        rewards_chosen = y * r_hat1 + (1 - y) * r_hat2
        rewards_rejected = (1 - y) * r_hat1 + y * r_hat2

        dist_rejected = rewards_rejected.softmax(1)
        dist_chosen = rewards_chosen.softmax(1)
        mean_dist = torch.concatenate(
            [dist_chosen, dist_rejected],
            dim=0,
        ).mean(dim=[0, 1])

        entropy_loss = torch.sum(mean_dist * mean_dist.log())
        reconstruction_loss, accuracy = self.reconstruction_loss(rewards_chosen, rewards_rejected)
        loss = reconstruction_loss + self.entropy_coeff * entropy_loss

        metrics = {
            "loss": loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "accuracy": accuracy.item(),
        }

        return loss, metrics


class MeanVarianceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, variance_penalty=0.1):
        super(MeanVarianceModel, self).__init__()
        self.variance_penalty = variance_penalty
        self.reward_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def get_reward(self, r):
        return self.reward_model(r)[:, 0]
    
    def get_variance(self, r):
        return F.softplus(r[:, 1])

    def sample_reward(self, r):
        r = self.reward_model(r)
        mean, std = r[:, 0], F.softplus(r[:, 1])
        return torch.normal(mean, std)

    def reconstruction_loss(self, rewards_chosen, rewards_rejected):
        mean_chosen = rewards_chosen[:, 0]
        std_chosen = F.softplus(rewards_chosen[:, 1])
        mean_rejected = rewards_rejected[:, 0]
        std_rejected = F.softplus(rewards_rejected[:, 1])

        diff_mean = mean_chosen - mean_rejected
        var_combined = std_chosen**2 + std_rejected**2
        z = diff_mean / torch.sqrt(var_combined)
        loss = F.softplus(-z * np.sqrt(2 * np.pi))
        return loss.mean(), torch.mean((loss < np.log(2)).float())

    def forward(self, s1, s2, y):
        assert s1.shape[2] == 1 and len(s1.shape) == 4
        assert s2.shape[2] == 1 and len(s2.shape) == 4

        s1 = s1.squeeze(2)
        s2 = s2.squeeze(2)

        r_hat1 = self.reward_model(s1)
        r_hat2 = self.reward_model(s2)

        rewards_chosen = y * r_hat1 + (1 - y) * r_hat2
        rewards_rejected = (1 - y) * r_hat1 + y * r_hat2

        std_chosen = F.softplus(rewards_chosen[:, 1])
        std_rejected = F.softplus(rewards_rejected[:, 1])
        variance_loss = (std_chosen**2 + std_rejected**2).mean()

        reconstruction_loss, accuracy = self.reconstruction_loss(
            rewards_chosen, rewards_rejected
        )
        loss = reconstruction_loss + self.variance_penalty * variance_loss

        metrics = {
            "loss": loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "variance_loss": variance_loss.item(),
            "accuracy": accuracy.item(),
        }

        return loss, metrics
