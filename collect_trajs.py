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

np.set_printoptions(precision=3, suppress=True)
import jaxrl_m.envs
import gym
from ravens.dataset import Dataset
from collections import defaultdict
import pickle

flags.DEFINE_string("data_dir", ".", "")
flags.DEFINE_bool("disp", False, "")
flags.DEFINE_string("env_name", "sort-mode0-v0", "")
flags.DEFINE_integer("n", 1000, "")
flags.DEFINE_integer("steps_per_seg", 3, "")
flags.DEFINE_bool("multimodal", True, "")
FLAGS = flags.FLAGS


def rollout_trajectory(env, agent, max_steps):
    obs = env.reset()
    init_pose = obs
    done = False
    traj = defaultdict(list)
    for i in range(max_steps):
        action = agent.act(obs, {})

        pos = action["move_cmd"][0]
        rot = action["move_cmd"][1]
        pos = pos + np.random.normal(0, 0.01, 3)
        action["move_cmd"] = (pos, rot)
        next_obs, reward, done, info = env.step_oracle(action)

        traj["obs"].append(obs)
        traj["action_dict"].append(action)
        traj["reward"].append(reward)
        traj["info"].append(info)
        traj["next_obs"].append(next_obs)
        traj["done"].append(done)
        obs = next_obs

        if done:
            action = agent.act(obs, {})
            next_obs, reward, done, info = env.step_oracle(action)
            traj["obs"].append(obs)
            traj["action_dict"].append(action)
            traj["reward"].append(reward)
            traj["info"].append(info)
            traj["next_obs"].append(next_obs)
            traj["done"].extend([False, True])
            break

    traj["obs"] = np.stack(traj["obs"])
    traj["next_obs"] = np.stack(traj["next_obs"])
    traj["action_dict"] = traj["action_dict"]
    traj["reward"] = np.array(traj["reward"])
    traj["info"] = traj["info"]
    traj["done"] = np.array(traj["done"])

    return traj, init_pose


def convert_traj(env, traj):
    ## quat = obs[9:13]
    #delta_pos = traj["next_obs"][:, 0:3] - traj["obs"][:, 0:3]
    ## delta_pos = np.clip(delta_pos, -1, 1)
    #suction = np.array([step["suction_cmd"] for step in traj["action_dict"]])[:, None]
    ## import pdb; pdb.set_trace()
    #action = np.concatenate([delta_pos, suction], axis=1)
    #assert len(action) == len(traj["obs"])
    #traj["action"] = action
    pose = np.stack([np.array(step["move_cmd"][0]) for step in traj["action_dict"]])
    suction = np.array([step["suction_cmd"] for step in traj["action_dict"]])[:, None]
    action = np.concatenate([pose, suction], axis=1)
    assert len(action) == len(traj["obs"])
    traj["action"] = action
    return traj


def test_traj(env, traj, init_pose):
    env.unwrapped.task.debug = True
    env.unwrapped.task.initial_pose = (init_pose[7:10], init_pose[10:14])
    obs = env.reset()
    done = False
    for i in range(len(traj["action"])):
        action = traj["action"][i]
        next_obs, reward, done, info = env.step(action)
        # if done:
        #     break
    env.unwrapped.task.debug = False

def plot_traj(traj):
    import matplotlib.pyplot as plt
    #plt.plot(traj["obs"][:, 0], traj["obs"][:, 1], label="obs")
    #plt.plot(traj["next_obs"][:, 0], traj["next_obs"][:, 1], label="next_obs")
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()
    #import pdb; pdb.set_trace()
    axs[0].plot(traj['obs'][:, 3], traj['obs'][:, 4], label="box_pose")
    axs[0].set_title("box_pose")
    axs[1].plot(traj['rewards'], label="rewards")
    axs[1].set_title("rewards")
    axs[2].plot(traj['done'], label="done")
    axs[2].set_title("done")
    axs[3].plot(traj['action'][:, -1], label="suction_cmd")
    axs[3].set_title("suction_cmd")
    plt.legend()
    plt.show()

def dump_traj(save_dir, traj, episode):
    with open(os.path.join(save_dir, f"traj_{episode}.pkl"), "wb") as f:
        pickle.dump(traj, f)


def main(unused_argv):

    env = gym.make(FLAGS.env_name, disp=FLAGS.disp, debug=True)
    task = env.unwrapped.task
    agent = task.oracle(env.unwrapped.env, steps_per_seg=FLAGS.steps_per_seg)

    seed = 0  # dataset.max_seed
    max_steps = task.max_steps * (FLAGS.steps_per_seg * agent.num_poses)
    episode = 0
    os.makedirs(FLAGS.data_dir, exist_ok=True)
    while episode < FLAGS.n:
        print(f"Oracle demonstration: {episode + 1}/{FLAGS.n}")
        np.random.seed(seed)
        env.set_mode(np.random.randint(len(env.goal_poses)))
        agent.reset()
        try:
            traj, init_pose = rollout_trajectory(env, agent, max_steps)
            traj = convert_traj(env, traj)

            # np.random.seed(seed)
            # test_traj(env, traj, init_pose)
            #plot_traj(traj)
        except Exception as e:
            seed += 2
            print(e)
            continue
        plot_traj(traj)
        dump_traj(FLAGS.data_dir, traj, episode)
        episode += 1
        seed += 2


if __name__ == "__main__":
    app.run(main)
