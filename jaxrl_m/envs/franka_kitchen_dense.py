import gym

import numpy as np
from itertools import permutations

from d4rl.kitchen.kitchen_envs import KitchenMicrowaveKettleLightSliderV0
from jaxrl_m.envs.base import MultiModalEnv

OBJECT_GOAL_VALS = { 
                          'slide cabinet':  [0.37],
                          'hinge cabinet':   [1.45],#[1.45],
                          'microwave'    :   [-0.75],
                        }

OBJECT_THRESH = { 
                          'slide cabinet':  0.1,
                          'hinge cabinet':   1,#[1.45],
                          'microwave'    :   0.4,
                        }
OBJECT_KEY_POS = {  
                    'slide cabinet':  [-0.12, 0.65, 2.6],
                    'hinge cabinet':  [-0.53, 0.65, 2.6],
                    'microwave'    :  [-0.63, 0.48, 1.8],
                    }
FINAL_KEY_POS = { 
                    'slide cabinet':  [0.2, 0.65, 2.6],
                    'hinge cabinet':  [-0.45, 0.53, 2.6],
                    'microwave'    :  [-0.7, 0.38, 1.8],
                    }
OBJECT_GOAL_IDXS = {
                    'slide cabinet':  [0],
                    'hinge cabinet':  [1],
                    'microwave'    :  [2],
                    }

INITIAL_STATE = np.array([ 4.79267505e-02,  3.71350919e-02, -4.65501369e-04, -1.77048263e-03,
        1.08009684e-03, -6.54363909e-01,  6.41530225e-01,  2.50198809e-01,
        3.12485842e-01, -4.31878959e-01,  1.18886426e-01,  2.02456874e+00])


BASE_TASK_NAMES = [   'bottom_burner', 
                        'light_switch', 
                        'slide cabinet', 
                        'hinge cabinet', 
                        'microwave', 
                        #'kettle' 
                  ]

class FrankaKitchenDenseEnv(MultiModalEnv):
    def __init__(
        self, dataset_path, fixed_mode=False, task_config=["slide cabinet", "microwave"]
    ):
        super().__init__(dataset_path=dataset_path, fixed_mode=fixed_mode)

        self.env = KitchenMicrowaveKettleLightSliderV0()

        obs_upper = 1 * np.ones(6)
        obs_lower = -obs_upper
        obs_upper_pose = 3 * np.ones(7)
        obs_lower_pose = -obs_upper_pose
        self.observation_space = gym.spaces.Box(np.concatenate([obs_lower, obs_lower_pose]),np.concatenate([obs_upper, obs_upper_pose]), dtype=np.float32)
        self.action_space = self.env.action_space
        print("observation space in kitchen", self.observation_space)

        self.tasks = task_config
        self.all_task_orders = list(permutations(self.tasks))

        if self.fixed_mode:
            self.env_task = self.all_task_orders[self.env_mode]

        self.relabel_offline_reward = True

    def set_mode(self, mode):
        super().set_mode(mode)
        self.env_task = self.all_task_orders[self.env_mode]

    def get_ee_quat(self):
        return self.env.sim.data.body_xquat[10]

    def get_ee_pose(self):
        return self.env.sim.data.site_xpos[self.env.sim.model.site_name2id("end_effector")]

    def get_num_modes(self):
        return len(self.all_task_orders)

    def reset(self):
        obs = self.env.reset()
        if not self.fixed_mode:
            self.set_mode(self.sample_mode())
        return obs

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        world_obs = self.internal_extract_state(self.env._get_obs())
        task_success = []
        for idx, task in enumerate(['slide cabinet', "hinge cabinet", "microwave"]):
            task_success.append(int(self.task_succeeded(task, world_obs)))
        task_success = np.array(task_success)
        ee_quat = self.get_ee_quat()
        ee_obs = self.get_ee_pose()
        obs = np.concatenate([world_obs, task_success, ee_quat,  ee_obs])

        info["success"] = task_success.mean()
        if info["success"] == 1:
            done = True
        reward = self.get_reward(obs[None, None], self.env_mode)[0, 0]
        return obs, reward, done, info
    
    def task_succeeded(self, task_name, achieved_state):
        per_obj_success = {
        #'bottom_burner' : ((achieved_state[2]<-0.38) and (goal[2]<-0.38)) or ((achieved_state[2]>-0.38) and (goal[2]>-0.38)),
        #'top_burner':    ((achieved_state[15]<-0.38) and (goal[6]<-0.38)) or ((achieved_state[6]>-0.38) and (goal[6]>-0.38)),
        #'light_switch':  ((achieved_state[10]<-0.25) and (goal[10]<-0.25)) or ((achieved_state[10]>-0.25) and (goal[10]>-0.25)),
        'slide cabinet' :  abs(achieved_state[OBJECT_GOAL_IDXS['slide cabinet']] - OBJECT_GOAL_VALS['slide cabinet'])<OBJECT_THRESH['slide cabinet'],
        'hinge cabinet' :  abs(achieved_state[OBJECT_GOAL_IDXS['hinge cabinet']] - OBJECT_GOAL_VALS['hinge cabinet'])<OBJECT_THRESH['hinge cabinet'],#0.6,#0.2,
        'microwave' :      abs(achieved_state[OBJECT_GOAL_IDXS['microwave']] - OBJECT_GOAL_VALS['microwave'])<OBJECT_THRESH['microwave'], #0.4,#0.2,
        #'kettle' : np.linalg.norm(achieved_state[16:18] - goal[16:18]) < 0.2
        }

        return per_obj_success[task_name]

    def batch_task_succeeded(self, achieved_state):
        per_obj_success = {
            'slide cabinet' :  abs(achieved_state[:, :, OBJECT_GOAL_IDXS['slide cabinet']] - OBJECT_GOAL_VALS['slide cabinet'])<OBJECT_THRESH['slide cabinet'],
            'hinge cabinet' :  abs(achieved_state[:, :, OBJECT_GOAL_IDXS['hinge cabinet']] - OBJECT_GOAL_VALS['hinge cabinet'])<OBJECT_THRESH['hinge cabinet'],#0.6,#0.2,
            'microwave' :      abs(achieved_state[:, :, OBJECT_GOAL_IDXS['microwave']] - OBJECT_GOAL_VALS['microwave'])<OBJECT_THRESH['microwave'], #0.4,#0.2,
        }

        return per_obj_success

    def internal_extract_state(self, obs):
        #gripper_pos = obs[7:9]
        slide_cabinet_joint = [obs[19]]
        hinge_cabinet_joint = [obs[21]]
        microwave_joint = [obs[22]]
        return np.concatenate([slide_cabinet_joint, hinge_cabinet_joint, microwave_joint])

    def get_task_distance(self, achieved_state, goal_name):
        goal_idxs = OBJECT_GOAL_IDXS[goal_name][0]
        achieved_joint = achieved_state[:, :, goal_idxs]
        goal_joint = OBJECT_GOAL_VALS[goal_name]
        original_joint = INITIAL_STATE[goal_idxs]

        distance_from_original = abs(original_joint -  achieved_joint)

        dist_slide = abs(achieved_joint-goal_joint)
        key_position = OBJECT_KEY_POS[goal_name]
  
        distance_to_key_pos = np.linalg.norm(achieved_state[:, : ,-3:]-key_position, axis=-1)
        # import pdb; pdb.set_trace()
        return distance_to_key_pos + dist_slide
    
    # def get_success(self, state):
    #     return np.array(
    #         [
    #            abs(state[OBS_ELEMENT_INDICES[task]] - OBJECT_GOAL_VALS[task]) < OBJECT_THRESH[task]
    #             for task in self.tasks
    #         ]
    #     ).mean()

    def get_reward(self, state, mode):
        task_order = self.all_task_orders[mode]
        r = self.get_ri(state, task_order)
        return r

    def get_preference_rewards(
        self, state1, state2, mode=None
    ):  # states are pf size B x T x STATE_DIM
        mode = mode or self.sample_mode()
        task_order = self.all_task_orders[mode]
        r0 = self.get_ri(state1, task_order)
        r1 = self.get_ri(state2, task_order)
        return r0, r1

    def get_ri(self, state, task_order):
        bonus = 5.0
        rewards = np.zeros(state.shape[:-1])
        mask = np.ones(state.shape[:-1])
        tasks_completed_dict = self.batch_task_succeeded(state)
        for task in task_order:
            tasks_distance = self.get_task_distance(state, task)
            tasks_completed = tasks_completed_dict[task][:, :, 0]
            rewards += (
                tasks_completed * bonus + (1 - tasks_completed) * mask * tasks_distance
            )
            mask *= tasks_completed
        return rewards

    def plot_gt(self, wandb_log=False):
        pass

    def plot_goals(self, ax, scale):
        pass
