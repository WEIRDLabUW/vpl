from typing import Dict
import jax
import numpy as np
from collections import defaultdict

import gym
from gym.wrappers import RecordEpisodeStatistics
import numpy as np

from jaxrl_m.wandb import WANDBVideo


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
    Wrapper that supplies a jax random key to a function (using keyword `seed`).
    Useful for stochastic policies that require randomness.

    Similar to functools.partial(f, seed=seed), but makes sure to use a different
    key for each new call (to avoid stale rng keys).

    """

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    """
    Helper function that flattens a dictionary of dictionaries
    into a single dictionary.
    E.g: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def step(env, action):
    try:
        obs, reward, done, info = env.step(action)
        return (obs, reward, done, info), False
    except Exception as e:
        print(e)
        return None, True


def sample_evaluate(
    policy_fn,
    env: gym.Env,
    reward_model,
    dataset,
    num_episodes: int,
    model_type: str,
    obs_fn=lambda x: x,
):
    from pref_learn.models.utils import get_posterior

    stats = defaultdict(list)
    env = RecordEpisodeStatistics(env)

    episode_count = 0
    while episode_count < num_episodes:
        # for i in range(num_episodes):
        mode = np.random.randint(env.get_num_modes())
        env.set_mode(mode)

        latent = None
        if "VAE" in model_type:
            latent = get_posterior(env, reward_model, dataset, mode, 1)[0]
        observation = env.reset()
        done = False
        infos = []
        fail_flag = False
        while not done:
            observation = obs_fn(observation, mode=mode, latent=latent)
            action = policy_fn(observation)
            step_data, fail_flag = step(env, action)
            if fail_flag:
                break
            observation, _, done, info = step_data

            if done:
                info["episode.mode"] = mode
            infos.append(info)
        if not fail_flag:
            episode_count += 1
            for info in infos:
                add_to(stats, flatten(info))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats


def evaluate(
    policy_fn,
    env: gym.Env,
    num_episodes: int,
    save_video: bool = False,
    render_frame: bool = True,
    name="eval_video",
    reset_kwargs={},
    obs_fn=lambda x: x,
    reward_fn=None,
    critic_fn=None,
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(
            env,
            name=name,
            max_videos=1,
            render_frame=render_frame,
            obs_fn=obs_fn,
            agent=critic_fn,
        )
    env = RecordEpisodeStatistics(env)

    stats = defaultdict(list)
    episode_count = 0
    while episode_count < num_episodes:
        observation = env.reset(**reset_kwargs)
        done = False
        learned_reward = []
        fail_flag = False
        infos = []
        while not done:
            observation = obs_fn(observation)
            if reward_fn is not None:
                learned_reward.append(reward_fn(observation))
            action = policy_fn(observation)
            step_data, fail_flag = step(env, action)
            if fail_flag:
                break
            observation, rew, done, info = step_data

            done = done
            if done and reward_fn is not None:
                info["episode.learned_reward"] = np.array(learned_reward).mean()
            infos.append(info)
        if not fail_flag:
            episode_count += 1
            for info in infos:
                add_to(stats, flatten(info))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats


# class EpisodeMonitor(gym.ActionWrapper):
#     """A class that computes episode returns and lengths."""

#     def __init__(self, env: gym.Env):
#         super().__init__(env)
#         self._reset_stats()
#         self.total_timesteps = 0

#     def _reset_stats(self):
#         self.reward_sum = 0.0
#         self.episode_length = 0
#         self.start_time = time.time()
#         # self.success = 0.0
#         self.success = 0.0

#     def step(self, action: np.ndarray):
#         observation, reward, done, info = self.env.step(action)

#         self.reward_sum += reward
#         self.success = max(self.success, info.get("success", 0.0))
#         self.success = max(self.success, info.get("success", 0.0))
#         self.episode_length += 1
#         self.total_timesteps += 1
#         info["total"] = {"timesteps": self.total_timesteps}
#         info = {}
#         if done:
#             info["episode"] = {}
#             info["episode"]["return"] = self.reward_sum
#             info["episode"]["length"] = self.episode_length
#             info["episode"]["success"] = self.success
#             info["episode"]["duration"] = time.time() - self.start_time
#             info["episode"]["success"] = self.success
#             info["episode"]["actual_success"] = info.get("success", 0.0)
#             if hasattr(self, "get_normalized_score"):
#                 info["episode"]["normalized_return"] = (
#                     self.get_normalized_score(info["episode"]["return"]) * 100.0
#                 )

#         return observation, reward, done, info

#     def reset(self) -> np.ndarray:
#         self._reset_stats()
#         return self.env.reset()
