from typing import Dict
import jax
import numpy as np
from collections import defaultdict
import time

from typing import Optional, Sequence
from collections import OrderedDict
import gym
import numpy as np
import sys
import wandb
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Optional, Any

import gym
import numpy as np


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


def evaluate(
    policy_fn,
    env: gym.Env,
    num_episodes: int,
    save_video: bool = False,
    render_frame: bool = True,
    name="eval_video",
    reset_kwargs={},
    latent=None,
    mode=None
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name=name, max_videos=1, render_frame=render_frame)
    env = EpisodeMonitor(env)

    stats = defaultdict(list)
    for i in range(num_episodes):
        observation = env.reset(**reset_kwargs)
        if mode is not None:
            env.set_mode(mode)
        elif hasattr(env, "reset_mode"):
            env.reset_mode()
        done = False
        while not done:
            if latent is not None:
                observation = np.concatenate([observation, latent], axis=-1)
            action = policy_fn(observation)
            observation, rew, done, info = env.step(action)
            done = done
            add_to(stats, flatten(info))
        # add_to(stats, flatten(info, parent_key="final"))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats


class WANDBVideo(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        name: str = "video",
        render_kwargs={},
        max_videos: Optional[int] = None,
        pixel_keys: Sequence[str] = ("pixels",),
        render_frame: bool = False,
        agent=None,
    ):
        super().__init__(env)
        self._name = name
        self._render_kwargs = render_kwargs
        self._max_videos = max_videos
        self._video = OrderedDict()
        self._rendered_video = []
        self._rewards = []
        self._pixel_keys = pixel_keys
        self._render_frame = render_frame
        if self._render_frame:
            assert len(self._pixel_keys) == 1 and self._pixel_keys[0] == "pixels"
        self._agent = agent
        if self._agent:
            self._curr_obs = None

    def get_rendered_video(self):
        rendered_video = [np.array(v) for v in self._video.values()]
        rendered_video = np.concatenate(rendered_video, axis=1)
        if rendered_video.ndim == 5:
            rendered_video = rendered_video[..., -1]
        return rendered_video

    def get_video(self):
        video = {k: np.array(v) for k, v in self._video.items()}
        return video

    def get_rewards(self):
        return self._rewards

    def _add_frame(self, obs, action=None):
        if self._max_videos is not None and self._max_videos <= 0:
            return
        if self._render_frame:
            frame = self.render(mode="rgb_array")
            if self._agent:
                if self._curr_obs is not None:
                    value = self._agent.eval_critic(self._curr_obs, action)
                    img = Image.fromarray(frame)
                    l, w, d = frame.shape
                    i1 = ImageDraw.Draw(img)
                    i1.text(
                        (l // 20, w // 20),
                        "critic: " + str(np.round(value, 3)),
                        font=ImageFont.truetype("FreeMonoBold.ttf", min(l, w) // 10),
                        fill=(255, 255, 255),
                    )
                    frame = np.asarray(img)
                self._curr_obs = obs
            if "pixels" in self._video:
                self._video["pixels"].append(frame)
            else:
                self._video["pixels"] = [frame]
        elif isinstance(obs, dict):
            img = []
            for k in self._pixel_keys:
                if k in obs:
                    if k in self._video:
                        self._video[k].append(obs[k])
                    else:
                        self._video[k] = [obs[k]]
        else:
            raise Exception("bad obs")

    def _add_rewards(self, rew):
        self._rewards.append(rew)

    def reset(self, **kwargs):
        self._video.clear()
        self._rendered_video.clear()
        self._rewards.clear()
        if self._agent:
            self._curr_obs = None
        obs = super().reset(**kwargs)
        self._add_frame(obs)
        return obs

    def step(self, action: np.ndarray):
        obs, reward, done, info = super().step(action)
        # done = done
        self._add_frame(obs, action)
        self._add_rewards(reward)

        if done and len(self._video) > 0:
            if self._max_videos is not None:
                self._max_videos -= 1
            video = self.get_rendered_video().transpose(0, 3, 1, 2)
            if video.shape[1] == 1:
                video = np.repeat(video, 3, 1)
            video = wandb.Video(video, fps=30, format="mp4")
            wandb.log({self._name: video}, commit=False)

        return obs, reward, done, info


class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()
        self.success = 0.0

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.success = max(self.success, info.get("success", 0.0))
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time
            info["episode"]["success"] = self.success
            info["episode"]["actual_success"] = info.get("success", 0.0)
            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()
