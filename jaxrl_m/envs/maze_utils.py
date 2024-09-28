import d4rl
import numpy as np
from d4rl.pointmaze import q_iteration
from d4rl.pointmaze.gridcraft import grid_env
from d4rl.pointmaze.gridcraft import grid_spec
import math

def factor_int(n):
    val = math.ceil(math.sqrt(n))
    val2 = int(n / val)
    while val2 * val != float(n):
        val -= 1
        val2 = int(n / val)
    return val, val2, n

def gridify_state(state):
    return (int(round(state[0])), int(round(state[1])))

def get_maze_env(env, maze_spec):
    if maze_spec is not None:
        maze_env = grid_env.GridEnv(grid_spec.spec_from_string(maze_spec))
    else:
        maze_env = grid_env.GridEnv(grid_spec.spec_from_string(env.str_maze_spec))
    return maze_env

def get_qmatrix(env, goal, obs, maze_spec=None):
    maze_env = get_maze_env(env, maze_spec)
    maze_env.gs[gridify_state(goal)] = grid_spec.REWARD
    q_values = q_iteration.q_iteration(env=maze_env, num_itrs=500, discount=0.99)
    obs_grid = np.array([gridify_state(o) for o in obs])
    obs_id = [maze_env.gs.xy_to_idx(o) for o in obs_grid]
    r = np.max(q_values[obs_id], axis=1)
    rs = np.array([r[i] for i in range(len(r)) if maze_env.gs[obs_grid[i]] != grid_spec.WALL])
    return (q_values, rs.min(), rs.max())

def get_reward(env, q_values, obs, maze_spec=None): # B x S x T x 2
    # import pdb; pdb.set_trace()
    maze_env = get_maze_env(env, maze_spec)
    obs_original_shape = obs.shape
    obs = obs.reshape(-1, 2)
    obs_grid = np.array([gridify_state(o) for o in obs])
    obs_id = [maze_env.gs.xy_to_idx(o) for o in obs_grid]
    r = np.max(q_values[obs_id], axis=1).reshape(obs_original_shape[:-1])
    return r

def apply_walls(env, reward, obs, maze_spec=None):
    maze_env = get_maze_env(env, maze_spec)
    for i in range(obs.shape[0]):
        if maze_env.gs[gridify_state(obs[i])] == grid_spec.WALL:
            reward[i] = 0
    return reward
    # obs_original_shape = obs.shape
    # obs = obs.reshape(-1, 2)
    # obs_grid = np.array([gridify_state(o) for o in obs])
    # obs_id = [maze_env.gs.xy_to_idx(o) for o in obs_grid]
    # r = np.max(reward[obs_id], axis=1).reshape(obs_original_shape[:-1])
    # return np.array([maze_env.gs[obs_grid[i]] == grid_spec.WALL for i in range(len(obs_grid))])

def plot_walls(env, ax, obs, maze_spec=None):
    maze_env = get_maze_env(env, maze_spec)
    walls = []
    for i in range(obs.shape[0]):
        if maze_env.gs[gridify_state(obs[i])] == grid_spec.WALL:
            walls.append(obs[i])
    walls = np.array(walls)
    ax.scatter(walls[:, 0], walls[:, 1], c="black")
    return ax

HARDEST_MAZE_TEST ='############\\'+\
                   '#OOOO#OOOOO#\\'+\
                   '#O##O#O#O#O#\\'+\
                   '#OOOOOO#OOO#\\'+\
                   '#O####O###O#\\'+\
                   '#OO#O#OOOOO#\\'+\
                   '##O#O#O#O###\\'+\
                   '#OO#OOO#OOO#\\'+\
                   '############'

def get_qmatrix_antmaze(env, goal, obs):
    return get_qmatrix(env, goal, obs, maze_spec=HARDEST_MAZE_TEST)