import os
from functools import partial
import pickle

from absl import app, flags
import d4rl
from flax.training import checkpoints
from ml_collections import config_flags
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tqdm
import wandb
import torch
from gym.wrappers import RecordEpisodeStatistics
from jaxrl_m.wandb import WANDBVideo
from collections import defaultdict
import gym

from jaxrl_m.evaluation import supply_rng, evaluate
import jaxrl_m.envs
from jaxrl_m.dataset import Dataset
import jaxrl_m.learners.iql as learner
import jaxrl_m.learners.d4rl_utils as d4rl_utils
from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict
from pref_learn.utils.plot_utils import plot_observation_rewards

from active_utils import get_queries
from pref_learn.models.utils import get_datasets

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "maze2d-twogoals-multimodal-v0", "Environment name.")
flags.DEFINE_string("wandb_project", "eval", "Wandb project name.")
flags.DEFINE_integer("seed", np.random.choice(1000000), "Random seed.")
flags.DEFINE_integer("samples", 100, "Number of episodes used for evaluation.")
flags.DEFINE_string("sampling_method", "random", "sampling method for queries")
flags.DEFINE_string("preference_dataset_path", "", "path to dataset")
flags.DEFINE_string("reward_model_path", "", "path to dataset")
flags.DEFINE_string("policy_path", "", "path to policy")

from typing import Dict
from pref_learn.utils.data_utils import get_labels

def get_latents(reward_model, env, obs1, obs2, mode):
    # obs1, obs2 = queries #[:, 0], queries[:, 1]
    obs_dim = obs1.shape[-1]
    seg_reward_1 = env.compute_reward(
        obs1.reshape(-1, reward_model.size_segment, obs_dim), mode
    )
    seg_reward_2 = env.compute_reward(
        obs2.reshape(-1, reward_model.size_segment, obs_dim), mode
    )
    seg_reward_1 = seg_reward_1.reshape(
        -1, reward_model.annotation_size, reward_model.size_segment, 1
    )
    seg_reward_2 = seg_reward_2.reshape(
        -1, reward_model.annotation_size, reward_model.size_segment, 1
    )

    labels = get_labels(seg_reward_1, seg_reward_2)
    device = next(reward_model.parameters()).device
    obs1 = torch.from_numpy(obs1).float().to(device)
    obs2 = torch.from_numpy(obs2).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)
    with torch.no_grad():
        mean, logvar = reward_model.encode(obs1, obs2, labels)
    # print(mean.shape)
    return mean.cpu().numpy(), logvar.cpu().numpy()

def main(_):
    setup_wandb(FLAGS, project=FLAGS.wandb_project)
    np.random.seed(FLAGS.seed)
    torch.random.manual_seed(FLAGS.seed)
    env_id = FLAGS.env_name
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
    reward_model = load_reward_model(FLAGS.reward_model_path)
    (
        _,
            _,
            _,
            preference_dataset,
            set_len,
            _,
            _,
    ) = get_datasets(
        FLAGS.preference_dataset_path,
        observation_dim,
        env.action_space.shape[0],
        256,
        reward_model.annotation_size
    )
    
    if FLAGS.sampling_method == "random":
        obs1, obs2 = get_queries(env, reward_model, preference_dataset, "random", 100)
        FLAGS.samples = 100
    else:
        obs1, obs2 = get_queries(env, reward_model, preference_dataset, FLAGS.sampling_method, 100, 5000)
        FLAGS.samples = 1
    hidden_dims = kwargs.get("hidden_dims")
    action_dim = kwargs.get("action_dim")
    agent = learner.load_learner(
        seed=FLAGS.seed,
        model_path=FLAGS.policy_path,
        discount=0.99,
        temperature=10.0,
        expectile=0.7,
        tau=0.1,
        hidden_dims=(256, 256),
        action_dim=env.action_space.shape[0],

    )
    policy_fn = partial(supply_rng(agent.sample_actions), temperature=0.0)

    def obs_fn(observation, mean, logvar):
        latent = mean #np.random.normal(mean, np.exp(logvar / 2))
        observation = np.concatenate([observation, latent], axis=-1)
        return observation

    # stats = {}
    rewards = []
    for ep in range(100):
        mode_n = np.random.randint(env.get_num_modes())
        env.set_mode(mode_n)
        if FLAGS.samples > 1:
        # if FLAGS.sampling_method == "random":
            ep_obs1 = obs1[ep, None]
            ep_obs2 = obs2[ep, None]
        else:
            ep_obs1 = obs1[0, None]
            ep_obs2 = obs2[0, None]
        mean, logvar = get_latents(reward_model, env, ep_obs1, ep_obs2, mode_n)
        # for ep in range(FLAGS.eval_episodes):
        eval_info = evaluate(
            policy_fn,
            env,
            num_episodes=1,
            save_video=False,
            name="video",
            obs_fn=partial(obs_fn, mean=mean[0], logvar=logvar[0]),
        )
        rewards.append(eval_info["episode.r"])
    wandb.log({"rewards": np.array(rewards).flatten().mean()})

if __name__ == "__main__":
    app.run(main)