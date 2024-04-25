import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import wandb

from ravens import tasks
from ravens.environments.environment import Environment

from d4rl.offline_env import OfflineEnv
import pybullet as p
import copy

import timeout_decorator


class SortEnv(OfflineEnv, gym.Env):
    def __init__(
        self, mode=-1, disp=False, debug=False, ee=False, oracle=False, **kwargs
    ):
        super().__init__()
        env_cls = Environment
        self.env = env_cls(
            "dependencies/ravens/ravens/environments/assets",  # TODO: fix path to assets
            disp=disp,  # or debug,
            shared_memory=False,
            hz=480,
            use_egl=not disp,  # and not debug
        )
        self.mode = mode
        self.is_multimodal = mode < 0

        if mode == 0:
            self.task = tasks.names["sort-easy-fixed"](continuous=True)
            self.dataset_path = os.path.join(
                os.path.dirname(__file__), f"dataset/sort_debug2-v0.hdf5"
            )
            self.goal_poses = [
                ((0.5, 0.2, 0), (0, 0, 0, 1)),
            ]
            self.task.goal_pose = self.goal_poses[0]
        else:
            self.task = tasks.names["sort-easy"](continuous=True)
            self.dataset_path = os.path.join(
                os.path.dirname(__file__), f"dataset/sort_debug2-v0.hdf5"
            )
            self.goal_poses = [
                ((0.5, 0.2, 0), (0, 0, 0, 1)),
                ((0.5, -0.2, 0), (0, 0, 0, 1)),
            ]
        self.task.mode = "train"
        self.env.set_task(self.task)

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(4,)  # 8,)
        )  # TODO: setup action space, self.env.action_space

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,)  # 15,)
        )  # TODO: setup observation space self.env.observation_space

        self._max_episode_steps = 25  # kwargs.get("max_episode_steps", 300)
        self.relabel_offline_reward = True
        self.debug = debug

        dataset = self.get_dataset()
        self.action_stats = {
            "min": dataset["action_min"],
            "max": dataset["action_max"],
        }

    @property
    def target(self):
        return self.goal_poses[self.mode]

    def get_dataset(self, path=None):
        path = path or self.dataset_path
        return super().get_dataset(h5path=path)

    def get_obs(self):
        ee_obs = self.env.get_ee_pose()
        grasp = self.env.ee.check_grasp()
        obs = [np.array(ee_obs[0])]
        pos, rot = p.getBasePositionAndOrientation(7)
        obs.append(np.array(pos))
        # obs.append(np.array(rot))
        obs.append(np.array([grasp]))
        obs = np.concatenate(
            obs
        ).flatten()  # ee_x, ee_y, ee_z, obj_x, obj_y, obj_z, grasp
        return obs

    def array2dict(self, action):
        action = np.clip(action, -1, 1)
        action = (action + 1) / 2
        action = (
            action * (self.action_stats["max"] - self.action_stats["min"])
            + self.action_stats["min"]
        )
        delta_pos = action[:3]
        suction = action[3]
        # current_pos = np.array(self.env.get_ee_pose()[0])
        # pose = current_pos + delta_pos
        pose = delta_pos
        suction = int(suction > 0.2)

        return {
            "move_cmd": (pose, (0, 0, 0, 1)),
            "suction_cmd": suction,
        }

    def compute_reward(self, state, mode):
        goal = self.goal_poses[mode]
        grip_pose = state[:, :, 0:3]
        obj_pose = state[:, :, 3:6]
        target_pose = np.array(goal[0])

        gripper_dist = np.linalg.norm(grip_pose - obj_pose, axis=-1)
        
        goal_dist = np.linalg.norm(obj_pose[:, :, :2] - target_pose[:2], axis=-1)
        dist = goal_dist < 0.05
        
        grasped = state[:, :, -1]
        
        reward = (
            dist * (1 - grasped) * 100.0  # if object is within goal dist, and ungrasped
            + dist * grasped * 5.0  # if object is within goal dist and grasped
            + (1 - dist) * grasped * (2 + np.exp(-goal_dist))  # if object is grasped and not within goal dist
            + (1 - dist) * (1 - grasped) * np.exp(-gripper_dist)  # if object is outside goal dist and ungrasped
        )
        return reward / 100.0

    def compute_sparse_reward(self, obs):
        goal_pos = self.goal_poses[self.mode][0][:2]
        obs_pos = obs[3:5]  # obs[7:9]
        dist = np.linalg.norm(goal_pos - obs_pos)
        height = obs[5]
        grasp = self.env.ee.check_grasp()
        return float(dist < 0.05 and height < 0.03 and not grasp)

    def reset(self):
        self.env.reset()
        self.elasped_steps = 0
        return self.get_obs()

    @timeout_decorator.timeout(2)
    def step(self, action):
        action_dict = self.array2dict(action)
        obs, reward, done, info = self.env.step(action_dict)
        obs = self.get_obs()
        
        reward = self.compute_sparse_reward(obs)
        done = done or reward > 0.99

        info = {}
        goal_pos = self.goal_poses[self.mode][0][:2]
        dist = np.linalg.norm(goal_pos - obs[3:5])
        info["debug_string"] = (
           f"{dist:.3f}, {action[-1]:.3f}, is grasped: {obs[-1]}"
        )
        info["dense_reward"] = self.compute_reward(obs[None, None], self.mode)[0, 0]
        if done:
           info["distance_to_goal"] = dist
        self.elasped_steps += 1
        done = done or (self.elasped_steps >= 50) # self.max_episode_steps
        #info["is_success"] = (reward > 0.99)
        return obs, reward, done, info

    @timeout_decorator.timeout(2)
    def step_oracle(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.get_obs()
        reward = self.compute_reward(obs[None, None], self.mode)[0, 0]
        sparse_reward = self.compute_sparse_reward(obs)
        done = done or sparse_reward > 0.99
        info["sparse_reward"] = sparse_reward
        info["distance_to_goal"] = np.linalg.norm(
            # obs[7:9] - self.goal_poses[self.mode][0][:2]
            obs[3:5]
            - self.goal_poses[self.mode][0][:2]
        )
        self.elasped_steps += 1
        done = self.elasped_steps >= self._max_episode_steps
        reward = self.compute_sparse_reward(obs)
        info["dense_reward"] = self.compute_reward(obs[None, None], self.mode)[0, 0]
        # print(reward)
        return obs, reward, done, info

    ## Functions to handle multimodality
    def get_num_modes(self):
        if self.is_multimodal:
            return len(self.goal_poses)
        return 1

    def sample_mode(self):
        if self.is_multimodal:
            return np.random.randint(2)
        return self.mode

    def reset_mode(self):
        self.set_mode(self.sample_mode())

    def set_mode(self, mode):
        if self.is_multimodal:
            self.mode = mode
            self.task.goal_pose = self.goal_poses[mode]

    def render(self, mode="rgb_array"):
        return self.env.render(mode)
