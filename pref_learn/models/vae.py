import os
import math
import pickle

import numpy as np
import torch
import torch.nn as nn
from pref_learn.models.flow import Flow

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h_ = self.model(x)
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x_hat = self.model(x)
        return x_hat


class VAEModel(nn.Module):
    def __init__(
        self,
        encoder_input,
        decoder_input,
        latent_dim,
        hidden_dim,
        annotation_size,
        size_segment,
        kl_weight=1.0,
        learned_prior=False,
        flow_prior=False,
        annealer=None
    ):
        super(VAEModel, self).__init__()
        self.Encoder = Encoder(encoder_input, hidden_dim, latent_dim)
        self.Decoder = Decoder(decoder_input, hidden_dim, 1)
        self.latent_dim = latent_dim
        self.mean = torch.nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learned_prior
        )
        self.log_var = torch.nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learned_prior
        )
        self.annotation_size = annotation_size
        self.size_segment = size_segment
        self.learned_prior = learned_prior

        self.flow_prior = flow_prior
        if flow_prior:
            self.flow = Flow(latent_dim, "radial", 4)
        
        self.kl_weight = kl_weight
        self.annealer = annealer

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(mean.device)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def encode(self, s1, s2, y):
        s1_ = s1.view(s1.shape[0], s1.shape[1], -1)
        s2_ = s2.view(s2.shape[0], s2.shape[1], -1)
        y = y.reshape(s1.shape[0], s1.shape[1], -1)

        encoder_input = torch.cat([s1_, s2_, y], dim=-1).view(
            s1.shape[0], -1
        )  # Batch x Ann x (2*T*State + 1)
        mean, log_var = self.Encoder(encoder_input)
        return mean, log_var

    def decode(self, obs, z):
        r = torch.cat([obs, z], dim=-1)  # Batch x Ann x T x (State + Z)
        r = self.Decoder(r)  # Batch x Ann x T x 1
        return r

    def get_reward(self, r):
        r = self.Decoder(r)  # Batch x Ann x T x 1
        return r

    def transform(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)

        return self.flow(z)

    def reconstruction_loss(self, x, x_hat):
        return nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")

    def accuracy(self, x, x_hat):
        predicted_class = (x_hat > 0.5).float()
        return torch.mean((predicted_class == x).float())

    def latent_loss(self, mean, log_var):
        if self.learned_prior:
            kl = -torch.sum(
                1
                + (log_var - self.log_var)
                - (log_var - self.log_var).exp()
                - (mean.pow(2) - self.mean.pow(2)) / (self.log_var.exp())
            )
        else:
            kl = -torch.sum(1.0 + log_var - mean.pow(2) - log_var.exp())
        return kl

    def forward(self, s1, s2, y):  # Batch x Ann x T x State, Batch x Ann x 1
        # import pdb; pdb.set_trace()
        mean, log_var = self.encode(s1, s2, y)

        if self.flow_prior:
            z, log_det = self.transform(mean, log_var)
        else:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # Batch x Z
            log_det = None
        z = z.repeat((1, self.annotation_size * self.size_segment)).view(
            -1, self.annotation_size, self.size_segment, z.shape[1]
        )

        r0 = self.decode(s1, z)
        r1 = self.decode(s2, z)

        r_hat1 = r0.sum(axis=2)
        r_hat2 = r1.sum(axis=2)

        p_hat = torch.nn.functional.sigmoid(r_hat1 - r_hat2).view(-1, 1)
        labels = y.view(-1, 1)

        reconstruction_loss = self.reconstruction_loss(labels, p_hat)
        accuracy = self.accuracy(labels, p_hat)
        latent_loss = self.latent_loss(mean, log_var)

        kl_weight = self.annealer.slope() if self.annealer else self.kl_weight
        loss = reconstruction_loss + kl_weight * latent_loss

        if self.flow_prior:
            loss = loss - torch.sum(log_det)

        metrics = {
            "loss": loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "kld_loss": latent_loss.item(),
            "accuracy": accuracy.item(),
            "kl_weight": kl_weight,
        }

        return loss, metrics

    def sample_prior(self, size):
        z = torch.randn(size, self.latent_dim).cuda()
        if self.learned_prior:
            z = z * torch.exp(0.5*self.log_var) + self.mean
        elif self.flow_prior:
            z, _ = self.flow(z)
        return z
    
    def sample_posterior(self, s1, s2, y):
        mean, log_var = self.encode(s1, s2, y)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        return mean, log_var, z