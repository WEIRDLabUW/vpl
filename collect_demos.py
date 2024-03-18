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

import jaxrl_m.envs
import gym
from ravens.dataset import Dataset

flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_string('env_name', 'sort-mode0-v0', '')
flags.DEFINE_integer('n', 1000, '')
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
        oracle_actions = agent.act(obs, info)
        if oracle_actions is None:
            return None, True
        oracle_action = np.concatenate([env.scale_actions(oracle_actions["pose0"][0]), env.scale_actions(oracle_actions["pose1"][0])])
        obs_arr = env.get_obs()
        obs, reward, done, info = env.step(oracle_action)
        obs_ = np.copy(obs_arr)[7:]
        # print(obs_[:5])
        # print(obs_[5:10])
        # print(obs_[10:15])
        # print(obs_[15:20])
        # print("\n\n")

        return (obs_arr, oracle_action, reward, done, info), False
    #   return obs, reward, done, info, act, False
    except TimeoutError as e:
        print("Function execution timed out")
        return None, True
    except Exception as e:
        print(e)
        return None, True
    finally:
      # Reset the alarm
      signal.alarm(0)

import gym
def main(unused_argv):

    env = gym.make(FLAGS.env_name, disp=FLAGS.disp, debug=True)
    task = env.unwrapped.task
    agent = task.oracle(env.unwrapped.env, steps_per_seg=FLAGS.steps_per_seg)
    dataset = Dataset(os.path.join(FLAGS.data_dir, f'{FLAGS.env_name}'))

    # Train seeds are even and test seeds are odd.
    seed = dataset.max_seed
    if seed < 0:
        seed = -1 if (task.mode == 'test') else -2

    # Determine max steps per episode.
    max_steps = task.max_steps

    # Collect training data from oracle demonstrations.
    while dataset.n_episodes < FLAGS.n:
        print(f'Oracle demonstration: {dataset.n_episodes + 1}/{FLAGS.n}')
        episode, total_reward = [], 0
        seed += 2
        np.random.seed(seed)
        # env.set_task(task)
        obs = env.reset()
        info = None
        reward = 0
        for _ in range(max_steps):
            traj_data, timeout = step(env, agent, obs, info)
            if timeout:
                if len(episode) > 0:
                    episode[-1][3]["done"] = True
                break
            # obs, reward, done, info = env.step(act)

            obs, action, reward, done, info = traj_data
            info["done"] = done
            episode.append((obs, action, reward, info))
            total_reward += reward
            
            # print(f'Total Reward: {total_reward} Done: {done}')
            if done:
                # import pdb; pdb.set_trace()
                break
        episode.append((obs, None, reward, info))

        # Only save completed demonstrations.
        # TODO(andyzeng): add back deformable logic.
        if total_reward > 0.99:
            dataset.add(seed, episode)

if __name__ == '__main__':
    app.run(main)
