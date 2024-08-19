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

"""Stacking task."""
import os

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils

import pybullet as p



class SortingEasy(Task):
    """Stacking task."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = 4
        # self.poses = None

        # self.goal_positions = [
        #     ((0.5, 0.2, 0), (0, 0, 0, 1)),
        #     ((0.5, -0.2, 0), (0, 0, 0, 1)),
        # ]

        self.no_rot = False
        # self.goal_idx = None
        self.goal_pose = None
        self.debug = False
        self.initial_pose = None

    def init_pose(self, poses):
        # self.poses = poses
        pass

    def reset(self, env):
        super().reset(env)

        zone_pose1 = ((0.5, 0.2, 0), (0, 0, 0, 1))
        env.add_object("zone/zone.urdf", zone_pose1, "fixed")

        zone_pose2 = ((0.5, -0.2, 0), (0, 0, 0, 1))
        env.add_object("zone/zone.urdf", zone_pose2, "fixed")

        num_blocks = 1
        colors = [
            utils.COLORS["blue"],
            # utils.COLORS["blue"],
            # utils.COLORS["red"],
            # utils.COLORS["red"],
        ]
        block_sizes = [
            (0.05, 0.05, 0.05),
            # (0.05, 0.05, 0.05),
            # (0.025, 0.025, 0.025),
            # (0.05, 0.05, 0.05),
        ]
        block_urdfs = [
            "sort/large_block.urdf",
            # "sort/block.urdf",
            # "sort/large_block.urdf",
            # "sort/block.urdf",
        ]
        # import pdb; pdb.set_trace()
        # goal_idx = self.goal_idx #np.random.randint(len(self.goal_positions))
        # goal_poses = [zone_pose1 if self.goal_idx == 0 else zone_pose2]
        # self.goal_pose = goal_poses[0]
        # goal_poses = [np.random.choice([zone_pose1, zone_pose2]) for _ in range(num_blocks)]
        # if self.sort_by_color:
        #     goal_poses = [zone_pose1]#[zone_pose1, zone_pose1, zone_pose2, zone_pose2]
        # else:
        #     goal_poses = [zone_pose2]#[zone_pose1, zone_pose2, zone_pose1, zone_pose2]

        # Add blocks.
        objs = []
        for i in range(num_blocks):
            block_pose = self.get_random_pose(env, block_sizes[i])
            # (x, y, z), _ = block_pose
            # if self.poses is not None:
            #     (x, y, z) = self.poses[i]
            if self.debug:
                block_pose = self.initial_pose#((0.703125, 0.065625, 0.025), (0.0, 0.0, 0.97192322, 0.23529821))
            # if self.no_rot:
            block_pose = (block_pose[0], (0, 0, 0, 1))
            block_id = env.add_object(block_urdfs[i], block_pose) #((x, y, z), (0, 0, 0, 1)))
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))

        self.goals = [
            (
                [objs[0]],
                np.ones((1, 1)),
                [self.goal_pose],
                False,
                True,
                "pose",
                None,
                1 / 4,
            ),
        ]
        # self.task_info = {
        #     "colors": colors,
        #     "goal_poses": goal_poses,
        #     "sort_by_color": self.sort_by_color,
        # }
        # self.poses = None

    def get_world_obs(
        self,
    ):
        obs = self._env._get_obs()
        return (
            obs["state"],
            obs["goal"],
            int(obs["object_grabbed"]),
            int(obs["suction_state"]),
        )

    def get_observation(self, env):
        obs = super().get_observation(env)
        obs["goal"] = np.array([0, 0, 0, 0])
        # self.poses = None
        return obs
    
class SortingEasyFixed(SortingEasy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.goal_positions = [
        #     ((0.5, 0.2, 0), (0, 0, 0, 1)),
        # ]