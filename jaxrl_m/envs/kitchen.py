import math
import gym
import numpy as np
import d4rl

INITIAL_STATE = np.array([ 4.79267505e-02,  3.71350919e-02, -4.65501369e-04, -1.77048263e-03,
        1.08009684e-03, -6.54363909e-01,  6.41530225e-01,  2.50198809e-01,
        3.12485842e-01, -4.31878959e-01,  1.18886426e-01,  2.02456874e+00])

class TruncateObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, truncate_size: int):
        super().__init__(env)
        self.truncate_size = truncate_size

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation[:self.truncate_size]
    
class KitchenEnv(gym.Env):
    def __init__(self, mode=-1, task_penalty=False):
        super().__init__()
        self.env = TruncateObservation(gym.make("kitchen-mixed-v0"), 30)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(30,)
        )
        self.action_space = self.env.action_space
        self._max_episode_steps = 280#self.env._max_episode_steps

        self.obs_element_indices = {
            "bottom left burner": np.array([11, 12]),
            "top left burner": np.array([15, 16]),
            "light switch": np.array([17, 18]),
            "slide cabinet": np.array([19]),
            "hinge cabinet": np.array([21]),
            "microwave": np.array([22]),
            "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
        }
        self.obs_element_goals = {
            "bottom left burner": np.array([-0.88, -0.01]),
            "top left burner": np.array([-0.92, -0.01]),
            "light switch": np.array([-0.69, -0.05]),
            "slide cabinet": np.array([0.37]),
            "hinge cabinet": np.array([1.45]),
            "microwave": np.array([-0.75]),
            "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
        }
        self.obs_element_thresh = {
            "bottom left burner": 0.3,
            "top left burner": 0.3,
            "light switch": 0.3,
            "slide cabinet": 0.1,
            "hinge cabinet": 1.0,
            "microwave": 0.4,
            "kettle": 0.1,
        }
        self.dist_thresh = 0.3
        self.num_tasks = len(self.obs_element_indices)

        self.goals = [['microwave', 'kettle'], ['bottom left burner', 'slide cabinet']] #["slide cabinet", "hinge cabinet"]] #[['slide cabinet','microwave'], ["kettle", "light switch"]] # #
        self.mode = mode
        self.relabel_offline_reward = True
        self.is_multimodal = mode < 0
        if not self.is_multimodal:
            self._target = self.goals[mode]
        
        self.task_penalty = task_penalty

    @property
    def target(self):
        return self._target

    def get_dataset(self):
        dataset = self.env.get_dataset()
        dataset["observations"] = dataset["observations"][:, :30]
        return dataset

    def reset(self):
        return self.env.reset()

    def step(self, action):
        # Compute shaped reward
        obs, reward, done, info = self.env.step(action)
        # Override environment termination
        reward = self.compute_sparse_reward(obs, done)
        return obs, reward, done, info

    def compute_sparse_reward(self, obs, done):
        if not done:
            return 0.0
        reward = []
        target = self.goals[self.mode]
        for task in target:
            goal = self.obs_element_goals[task]
            indices = self.obs_element_indices[task]
            reward.append(np.linalg.norm(obs[indices] - goal) < self.dist_thresh)
        return np.mean(reward)
    
    def compute_reward(self, obs, mode):
        # Setting mode to random if not provided
        if self.mode < 0:
            if mode < 0:
                mode = np.random.randint(2)
            mode = mode
        else:
            mode = self.mode
        target = self.goals[mode]
        rewards = np.zeros(obs.shape[:-1])
        mask = np.ones(obs.shape[:-1], dtype=bool)
        for task in target:
            goal = self.obs_element_goals[task]
            indices = self.obs_element_indices[task]
            dist_to_goal = np.linalg.norm(obs[:, :, indices] - goal, axis=-1)
            success_flag = np.linalg.norm(obs[:, :, indices] - goal, axis=-1) < self.obs_element_thresh[task]#self.dist_thresh
            rewards += 5.0*success_flag + (1 - success_flag) * mask * (-dist_to_goal)
            # mask = mask & success_flag
        
        if self.task_penalty:
            for task, target_indices in self.obs_element_indices.items():
                if task in target:
                    continue
                init_state = INITIAL_STATE[target_indices]
                dist_from_init = np.linalg.norm(obs[:, :, target_indices] - init_state, axis=-1)
                rewards += -dist_from_init
        return rewards
        # for task, target_indices in self.obs_element_indices.items():
        #     goal = self.obs_element_goals[task]
        #     dist_to_goal = np.linalg.norm(obs[:, :, target_indices] - goal, axis=-1)
        #     success_flag = dist_to_goal < self.dist_thresh
        #     if task in self._target:
        #         rewards += 5.0*success_flag + (1 - success_flag) * (-dist_to_goal)
        #     elif self.task_penalty:
        #         rewards += -success_flag
        # return rewards

    def render(self, mode="rgb_array"):
        return self.env.render(mode)

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
            self._target = self.goals[mode]