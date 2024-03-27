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

from active import get_queries

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_integer("seed", np.random.choice(1000000), "Random seed.")
flags.DEFINE_integer("eval_episodes", 100, "Number of episodes used for evaluation.")
flags.DEFINE_bool("save_video", False, "Save video of the agent.")
flags.DEFINE_string("model_path", "none", "path to policy model")
flags.DEFINE_string("reward_path","none", "path to reward model")
flags.DEFINE_string("sampling_method", "random", "sampling method for queries")

from typing import Dict
from pref_learn.utils.data_utils import get_labels

# def load_reward_model(ckpt):
#     with open(os.path.join(ckpt, "best_model.pt"), "rb") as f:
#         reward_model = torch.load(f)
#     return reward_model


# def get_queries(env, annotation_size, num_queries):
#     env.set_biased_mode("random")
#     queries = [env.get_biased_data(annotation_size) for _ in range(num_queries)]
#     return np.array(queries)  # (num_queries, 2, annotation_size, obs_dim)

def get_latents(reward_model, env, queries, mode):
    obs1, obs2 = queries #[:, 0], queries[:, 1]
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

def load_reward_model(ckpt):
    with open(os.path.join(ckpt, "best_model.pt"), "rb") as f:
        reward_model = torch.load(f)
    return reward_model


def main(_):
    setup_wandb(FLAGS)
    env = d4rl_utils.make_env(FLAGS.env_name)
    reward_model = load_reward_model(FLAGS.reward_path)
    # with open(os.path.join(FLAGS.model_path, "config.pkl"), "rb") as f:
    #     config = pickle.load(f)
    # print(config)
    agent = learner.load_learner(
        seed=FLAGS.seed,
        model_path=FLAGS.model_path,
        discount=0.99,
        temperature=10.0,
        expectile=0.7,
        tau=0.1,
    )
    policy_fn = partial(supply_rng(agent.sample_actions), temperature=0.0)
    queries = get_queries(env, reward_model, FLAGS.sampling_method)#get_queries(env, reward_model.annotation_size, FLAGS.eval_episodes)

    def obs_fn(observation, mean, logvar):
        latent = mean#np.random.normal(mean, np.exp(logvar / 2))
        observation = np.concatenate([observation, latent], axis=-1)
        return observation

    stats = {}
    for mode_n in range(env.get_num_modes()):
        env.set_mode(mode_n)
        mean, logvar = get_latents(reward_model, env, queries, mode_n)
        rewards = []
        # for ep in range(FLAGS.eval_episodes):
        eval_info = evaluate(
            policy_fn,
            env,
            num_episodes=FLAGS.eval_episodes,
            save_video=FLAGS.save_video,
            name="video",
            obs_fn=partial(obs_fn, mean=mean[0], logvar=logvar[0]),
        )
        stats[mode_n] = eval_info["episode.r"]
            # print(eval_info)
    print(FLAGS.sampling_method)
    print(stats)
    print("\n\n")

if __name__ == "__main__":
    app.run(main)