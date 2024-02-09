import os
import pickle

import numpy as np
from tqdm import tqdm
from jaxrl_m.learners.d4rl_utils import new_get_trj_idx

def sample_from_env(env, num_query, len_set, len_query, data_dir):
    assert len_query == 1
    observation_dim = env.reward_observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]
    seg_obs_1 = np.stack(
        [env.reward_observation_space.sample() for _ in range(num_query * len_set)],
        axis=1,
    ).reshape(num_query, len_set, len_query, observation_dim)
    seg_obs_2 = np.stack(
        [env.reward_observation_space.sample() for _ in range(num_query * len_set)],
        axis=1,
    ).reshape(num_query, len_set, len_query, observation_dim)
    seg_act_1 = np.stack(
        [env.action_space.sample() for _ in range(num_query * len_set)], axis=1
    ).reshape(num_query, len_set, len_query, observation_dim)
    seg_act_2 = np.stack(
        [env.action_space.sample() for _ in range(num_query * len_set)], axis=1
    ).reshape(num_query, len_set, len_query, observation_dim)
    labels = np.zeros((num_query, len_set))

    query_path = os.path.join(
        data_dir, f"queries_num{num_query}_q{len_query}_s{len_set}"
    )
    batch = {}
    batch["labels"] = labels.reshape(num_query, len_set, 1)
    batch["observations"] = seg_obs_1.reshape(
        num_query, len_set, len_query, observation_dim
    )
    batch["observations_2"] = seg_obs_2.reshape(
        num_query, len_set, len_query, observation_dim
    )
    batch["actions"] = seg_act_1.reshape(num_query, len_set, len_query, action_dim)
    batch["actions_2"] = seg_act_2.reshape(num_query, len_set, len_query, action_dim)
    with open(query_path, "wb") as fp:
        pickle.dump(batch, fp)

    return batch, query_path


def get_queries_from_multi(
    env,
    dataset,
    num_query,
    len_query,
    len_set,
    data_dir=None,
    skip_flag=1,
):
    _num_query = num_query
    num_query *= len_set

    os.makedirs(data_dir, exist_ok=True)
    trj_idx_list = new_get_trj_idx(dataset)  # get_nonmdp_trj_idx(env)
    labeler_info = np.zeros(len(trj_idx_list) - 1)

    # to-do: parallel implementation
    trj_idx_list = np.array(trj_idx_list)
    trj_len_list = trj_idx_list[:, 1] - trj_idx_list[:, 0] + 1

    assert max(trj_len_list) > len_query

    total_reward_seq_1, total_reward_seq_2 = np.zeros((num_query, len_query)), np.zeros(
        (num_query, len_query)
    )

    observation_dim = dataset["observations"].shape[-1]
    total_obs_seq_1, total_obs_seq_2 = np.zeros(
        (num_query, len_query, observation_dim)
    ), np.zeros((num_query, len_query, observation_dim))
    total_next_obs_seq_1, total_next_obs_seq_2 = np.zeros(
        (num_query, len_query, observation_dim)
    ), np.zeros((num_query, len_query, observation_dim))

    action_dim = dataset["actions"].shape[-1]
    total_act_seq_1, total_act_seq_2 = np.zeros(
        (num_query, len_query, action_dim)
    ), np.zeros((num_query, len_query, action_dim))

    total_timestep_1, total_timestep_2 = np.zeros(
        (num_query, len_query), dtype=np.int32
    ), np.zeros((num_query, len_query), dtype=np.int32)

    start_indices_1, start_indices_2 = np.zeros(num_query), np.zeros(num_query)
    time_indices_1, time_indices_2 = np.zeros(num_query), np.zeros(num_query)

    query_path = os.path.join(
        data_dir, f"queries_num{_num_query}_q{len_query}_s{len_set}"
    )
    # already_queried = []
    for query_count in tqdm(range(num_query), desc="get queries"):
        temp_count = 0
        labeler = -1
        while temp_count < 2:
            trj_idx = np.random.choice(
                np.arange(len(trj_idx_list) - 1)[np.logical_not(labeler_info)]
            )
            len_trj = trj_len_list[trj_idx]

            if len_trj > len_query and (
                temp_count == 0 or labeler_info[trj_idx] == labeler
            ):
                labeler = labeler_info[trj_idx]
                time_idx = np.random.choice(len_trj - len_query + 1)
                start_idx = trj_idx_list[trj_idx][0] + time_idx
                end_idx = start_idx + len_query

                assert end_idx <= trj_idx_list[trj_idx][1] + 1

                reward_seq = dataset["rewards"][start_idx:end_idx]
                obs_seq = dataset["observations"][start_idx:end_idx]
                next_obs_seq = dataset["next_observations"][start_idx:end_idx]
                act_seq = dataset["actions"][start_idx:end_idx]
                # timestep_seq = np.arange(time_idx + 1, time_idx + len_query + 1)
                timestep_seq = np.arange(1, len_query + 1)

                # skip flag 1: skip queries with equal rewards.
                if skip_flag == 1 and temp_count == 1:
                    if np.sum(total_reward_seq_1[-1]) == np.sum(reward_seq):
                        continue
                # skip flag 2: keep queries with equal reward until 50% of num_query.
                if (
                    skip_flag == 2
                    and temp_count == 1
                    and query_count < int(0.5 * num_query)
                ):
                    if np.sum(total_reward_seq_1[-1]) == np.sum(reward_seq):
                        continue
                # skip flag 3: keep queries with equal reward until 20% of num_query.
                if (
                    skip_flag == 3
                    and temp_count == 1
                    and query_count < int(0.2 * num_query)
                ):
                    if np.sum(total_reward_seq_1[-1]) == np.sum(reward_seq):
                        continue

                if temp_count == 0:
                    start_indices_1[query_count] = start_idx
                    time_indices_1[query_count] = time_idx
                    total_reward_seq_1[query_count] = reward_seq
                    total_obs_seq_1[query_count] = obs_seq
                    total_next_obs_seq_1[query_count] = next_obs_seq
                    total_act_seq_1[query_count] = act_seq
                    total_timestep_1[query_count] = timestep_seq
                else:
                    # if (start_idx, start_indices_1[query_count]) in already_queried:
                    #     continue
                    start_indices_2[query_count] = start_idx
                    time_indices_2[query_count] = time_idx
                    total_reward_seq_2[query_count] = reward_seq
                    total_obs_seq_2[query_count] = obs_seq
                    total_next_obs_seq_2[query_count] = next_obs_seq
                    total_act_seq_2[query_count] = act_seq
                    total_timestep_2[query_count] = timestep_seq

                temp_count += 1
                # already_queried.append(
                #     (start_indices_2[query_count], start_indices_1[query_count])
                # )

    seg_reward_1 = total_reward_seq_1.copy()
    seg_reward_2 = total_reward_seq_2.copy()

    seg_obs_1 = total_obs_seq_1.copy()
    seg_obs_2 = total_obs_seq_2.copy()

    seg_next_obs_1 = total_next_obs_seq_1.copy()
    seg_next_obs_2 = total_next_obs_seq_2.copy()

    seq_act_1 = total_act_seq_1.copy()
    seq_act_2 = total_act_seq_2.copy()

    seq_timestep_1 = total_timestep_1.copy()
    seq_timestep_2 = total_timestep_2.copy()

    rational_labels = get_labels(seg_reward_1, seg_reward_2)

    start_indices_1 = np.array(start_indices_1, dtype=np.int32)
    start_indices_2 = np.array(start_indices_2, dtype=np.int32)
    time_indices_1 = np.array(time_indices_1, dtype=np.int32)
    time_indices_2 = np.array(time_indices_2, dtype=np.int32)

    batch = {}
    batch["labels"] = rational_labels.reshape(_num_query, len_set, 1)
    batch["observations"] = seg_obs_1.reshape(
        _num_query, len_set, len_query, observation_dim
    )  # for compatibility, remove "_1"
    batch["next_observations"] = seg_next_obs_1.reshape(
        _num_query, len_set, len_query, observation_dim
    )
    batch["actions"] = seq_act_1.reshape(_num_query, len_set, len_query, action_dim)
    batch["observations_2"] = seg_obs_2.reshape(
        _num_query, len_set, len_query, observation_dim
    )
    batch["next_observations_2"] = seg_next_obs_2.reshape(
        _num_query, len_set, len_query, observation_dim
    )
    batch["actions_2"] = seq_act_2.reshape(_num_query, len_set, len_query, action_dim)
    batch["timestep_1"] = seq_timestep_1.reshape(_num_query, len_set, len_query)
    batch["timestep_2"] = seq_timestep_2.reshape(_num_query, len_set, len_query)
    batch["start_indices"] = start_indices_1.reshape(_num_query, len_set)
    batch["start_indices_2"] = start_indices_2.reshape(_num_query, len_set)

    with open(query_path, "wb") as fp:
        pickle.dump(batch, fp)

    return batch, query_path


def get_labels(seg_reward_1, seg_reward_2):
    sum_r_t_1 = np.sum(seg_reward_1, axis=-1)
    sum_r_t_2 = np.sum(seg_reward_2, axis=-1)
    binary_label = (sum_r_t_1 > sum_r_t_2).reshape(-1, 1).astype(np.float32)
    return binary_label
