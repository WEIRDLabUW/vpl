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

class SortEnv(OfflineEnv, gym.Env):
    def __init__(self, mode=-1, disp=False, debug=False, ee=False, **kwargs):
        super().__init__()

        env_cls = Environment
        self.env = env_cls(
            "dependencies/ravens/ravens/environments/assets",  # TODO: fix path to assets
            disp=disp, # or debug,
            shared_memory=False,
            hz=480,
            use_egl=not disp# and not debug
        )
        self.mode = mode
        self.is_multimodal = mode < 0
        
        self.task = tasks.names["sort"](continuous=False)
        self.task.mode = "train"
        self.env.set_task(self.task)

        if not self.is_multimodal:
            self.task.sort_by_color = True if mode == 0 else False

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(6,)
        )  # TODO: setup action space, self.env.action_space
        self.ee = ee
        if ee:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(19,)
            )  # TODO: setup observation space self.env.observation_space
        else:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(12,))
        self._max_episode_steps = kwargs.get("max_episode_steps", 50)

        self.relabel_offline_reward = True
        self.debug = debug
        
        # if self.debug:
        #     self.dataset_path = os.path.join(
        #         os.path.dirname(__file__), f"dataset/sort_mode_0_debug.hdf5"
        #     )
        # else:
        #     self.dataset_path = os.path.join(
        #         os.path.dirname(__file__), f"dataset/sort_mode_0.hdf5"
        #     )
        self.dataset_path =  os.path.join(
            os.path.dirname(__file__), f"dataset/sort_mode_0_debug.hdf5"
        )
        self.init_obs = self.get_dataset()["init_poses"]
        #     if ee:
        #         self.dataset_path = os.path.join(
        #             os.path.dirname(__file__), f"dataset/sort_{mode}_ee.hdf5"
        #         )
        #     else:
        #         if debug:
        #             self.dataset_path = os.path.join(
        #                 os.path.dirname(__file__), f"dataset/sort_{mode}_no_ee_debug2.hdf5"
        #             )
        #             self.init_obs = self.get_dataset()["init_poses"]
        #         else:
        #             self.dataset_path = os.path.join(
        #                 os.path.dirname(__file__), f"dataset/sort_{mode}_no_ee.hdf5"
        #             )
        # else:
        #     self.dataset_path = os.path.join(
        #         os.path.dirname(__file__), f"dataset/sort.hdf5"
        #     )
        self.goals = None

    @property
    def target(self):
        return self.task.sort_by_color
    
    # @property
    # def _max_episode_steps(self):
    #     return 100

    def get_dataset(self, path=None):
        path = path or self.dataset_path
        return super().get_dataset(h5path=path)

    def get_obs(self):
        ee_obs = self.env.get_ee_pose()
        if self.ee:
            obs = [np.array(ee_obs[0]), np.array(ee_obs[1])]
        else:
            obs = []
        for obj_id in [7, 8, 9, 10]:
            pos, rot = p.getBasePositionAndOrientation(obj_id)
            # dim = p.getVisualShapeData(obj_id)[0][3]
            # color = p.getVisualShapeData(obj_id)[0][7]
            # # print(obj_id, dim, color)
            # if dim[0] > 0.04:
            #     dim = np.array([1.0])
            # else:
            #     dim = np.array([0.0])

            # if color[0] > 0.5:
            #     color = np.array([1.0])
            # else:
            #     color = np.array([0.0])

            obs.append(pos)
            # obs.append(dim)
            # obs.append(color)
        return np.concatenate(obs).flatten()
    
    def compute_reward(self, state, mode): # 
        # Setting mode to random if not provided
        if mode == 0:
            goals = self.env.task.mode_0_goals
        elif mode == 1:
            goals = self.env.task.mode_1_goals
        else:
            raise ValueError("Invalid mode")
        
        rewards = np.zeros(state.shape[:-1])
        for i, obj_id in enumerate([7, 8, 9, 10]):
            obj_pose = state[:, :, 3*i:3*(i+1)]
            target_pose = np.array(goals[i][0])
            dist = np.linalg.norm(obj_pose[:, :, :2] - target_pose[:2], axis=-1)

            mask = dist < 0.01
            rewards = rewards + 5*mask + (1-mask)*-dist
        return rewards
    
    def scale_actions(self, action):
        # x,y,z = action[:3]
        # x = 2*x-1
        # y = 2*y
        # z = 4*z-1
        return action#np.array([x,y,z])
    
    def unscale_actions(self, action):
        # action = np.clip(action, -1, 1)
        # x,y,z = action[:3]
        # x = (x+1)/2
        # y = y/2
        # z = (z+1)/4
        # return np.array([x,y,z])
        return action
    
    def get_action(self, action):
        pose0 = self.unscale_actions(action[:3])
        pose1 = self.unscale_actions(action[3:])
        # suction = not self.env.ee.check_grasp()
        # act_dict = {
        #     "move_cmd": (pose, np.array([0,0,0,1])),
        #     "suction_cmd": suction,
        #     "acts_left": 0
        # }
        # act_dict = {
        #     "pose0": (pose, np.array([0,0,0,1])),
        #     "pick_action": suction,
        # }
        act_dict = {
            "pose0": (pose0, np.array([0,0,0,1])),
            "pose1": (pose1, np.array([0,0,0,1])),
        }
        return act_dict

    def reset_end_effector(self, pose):
        pose = (pose[:3], pose[3:])
        act_dict = {
            "pose0": pose,
            "pick_action": 1,
        }
        for _ in range(10):
            self.env.step(act_dict)

    def reset_scene(self):
        init_obs = self.init_obs[np.random.randint(len(self.init_obs))]
        if init_obs.shape[0] == 19:
            init_obs = init_obs[7:]
        else:
            init_obs = init_obs
        init_pose_obs = []
        for i in range(4):
            o = init_obs[i*3: (i+1)*3]
            init_pose_obs.append(o[:3])
        self.env.task.init_pose(init_pose_obs)

    def reset(self):
        if self.debug:
            self.reset_scene()
        self.env.reset()
        self.goals = copy.deepcopy(self.env.task.goals)
        return self.get_obs()

    def reset_mode(self):
        pass  # TODO: setup reset_mode
        # self.env.set_target(self.goal1 if self.env_mode == 0 else self.goal2)

    def step(self, action):
        obs, reward, done, info = self.env.step(self.get_action(action))
        # if self.debug:
        #     return obs, reward, done, info
        obs = self.get_obs()
        # reward = self.compute_reward(obs[None, None], self.mode)[0,0]
        return obs, reward, done, info

    ## Functions to handle multimodality
    def get_num_modes(self):
        if self.is_multimodal:
            return 2
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
            self.task.sort_by_color = True if mode == 0 else False

    def render(self, mode="rgb_array"):
        return self.env.render(mode)
