# coding=utf-8
# Copyright 2023 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data collection script."""

import os

from absl import app
from absl import flags

import numpy as np

from ravens import tasks
from ravens.dataset import Dataset
from ravens.environments.environment import ContinuousEnvironment
from ravens.environments.environment import Environment

flags.DEFINE_string('assets_root', '.', '')
flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'towers-of-hanoi', '')
flags.DEFINE_string('mode', 'train', '')
flags.DEFINE_integer('n', 1000, '')
flags.DEFINE_bool('continuous', True, '')
flags.DEFINE_integer('steps_per_seg', 3, '')

FLAGS = flags.FLAGS

import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

def step(env, agent, obs, info, timeout_seconds=5):
    # Set the signal handler
    signal.signal(signal.SIGALRM, timeout_handler)

    # Set the timeout for the function
    signal.alarm(timeout_seconds)
    try:
      act = agent.act(obs, info)
      obs, reward, done, info = env.step(act)
      return obs, reward, done, info, act, False
    except TimeoutError as e:
      print("Function execution timed out")
      return None, None, None, None, None, True
    else:
      # Reset the alarm
      signal.alarm(0)

def main(unused_argv):

  # Initialize environment and task.
  env_cls = ContinuousEnvironment if FLAGS.continuous else Environment
  env = env_cls(
      FLAGS.assets_root,
      disp=FLAGS.disp,
      shared_memory=FLAGS.shared_memory,
      hz=480)
  task = tasks.names[FLAGS.task](continuous=FLAGS.continuous)
  task.mode = FLAGS.mode

  # Initialize scripted oracle agent and dataset.
  agent = task.oracle(env, steps_per_seg=FLAGS.steps_per_seg)
  dataset = Dataset(os.path.join(FLAGS.data_dir, f'{FLAGS.task}-{task.mode}'))

  # Train seeds are even and test seeds are odd.
  seed = dataset.max_seed
  if seed < 0:
    seed = -1 if (task.mode == 'test') else -2

  # Determine max steps per episode.
  max_steps = task.max_steps
  if FLAGS.continuous:
    max_steps *= (FLAGS.steps_per_seg * agent.num_poses)

  # Collect training data from oracle demonstrations.
  while dataset.n_episodes < FLAGS.n:
    print(f'Oracle demonstration: {dataset.n_episodes + 1}/{FLAGS.n}')
    episode, total_reward = [], 0
    seed += 2
    np.random.seed(seed)
    env.set_task(task)
    obs = env.reset()
    info = None
    reward = 0
    for _ in range(max_steps):
      og_obs = obs
      og_obs_info = env.info
      obs, reward, done, info, act, timeout = step(env, agent, obs, info)
      if timeout:
        episode[-1][3]["done"] = True
        break
      # obs, reward, done, info = env.step(act)
      info["obs_info"] = og_obs_info
      info["next_obs_info"] = env.info
      info["done"] = done
      episode.append((og_obs, act, reward, info))
      total_reward += reward
      # print(f'Total Reward: {total_reward} Done: {done}')
      if done:
        break
    episode.append((obs, None, reward, info))

    # Only save completed demonstrations.
    # TODO(andyzeng): add back deformable logic.
    if total_reward > 0.99:
      dataset.add(seed, episode)

if __name__ == '__main__':
  app.run(main)
