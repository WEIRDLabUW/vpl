import pathlib
import numpy as np
import pickle
import jaxrl_m.envs
import gym
import gzip
import h5py


def get_data(path):
    traj_paths = sorted(list(pathlib.Path(path).rglob("traj*.pkl")))

    rewards = []
    infos = []
    observations = []
    dones = []
    traj_len = []
    init_poses = []
    last_poses = []
    actions = []
    for traj_path in traj_paths:
        with open(traj_path, "rb") as f:
            traj = pickle.load(f)
        rewards.append(traj["reward"][:25])
        infos.append(traj["info"][:25])
        observations.append(traj["obs"][:25])
        dones.append(traj["done"][:25])
        traj_len.append(len(traj["reward"][:25]))
        init_poses.append(traj["obs"][0][:25])
        last_poses.append(traj["obs"][-1][:25])
        actions.append(traj["action"][:25])

    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    observations = np.concatenate(observations, axis=0)
    dones = np.concatenate(dones, axis=0)
    traj_len = np.array(traj_len)
    init_poses = np.stack(init_poses, axis=0)
    last_poses = np.stack(last_poses, axis=0)
    return (
        actions,
        rewards,
        observations,
        dones,
        traj_len,
        init_poses,
        last_poses,
        infos,
    )


import numpy as np
import pickle
import gzip
import h5py


def reset_data():
    return {
        "observations": [],
        "actions": [],
        "terminals": [],
        "rewards": [],
        "suctions": [],
        "init_poses": [],
    }


def add_data(s, a, r, done, init_poses, infos):
    # print(s.shape, a.shape, r.shape, done.shape)
    data = reset_data()
    data["observations"] = s
    data["rewards"] = r
    data["terminals"] = done
    data["init_poses"] = init_poses
    data["action_max"] = np.max(a, axis=0)
    data["action_min"] = np.min(a, axis=0)
    normalised_action = (a - data["action_min"]) / (
        data["action_max"] - data["action_min"]
    )
    normalised_action = normalised_action * 2 - 1
    data["actions"] = normalised_action
    return data


def npify(data):
    for k in data:
        if k == "infos":
            continue
        if k == "terminals":
            dtype = np.bool_
        else:
            dtype = np.float32
        data[k] = np.array(data[k], dtype=dtype)


if __name__ == "__main__":
    path = "/home/max/Distributional-Preference-Learning/vpl/dependencies/sort_dataset"
    fname = "/home/max/Distributional-Preference-Learning/vpl/jaxrl_m/envs/dataset/sort_debug2-v0.hdf5"
    actions, rewards, observations, dones, traj_len, init_poses, last_poses, infos = (
        get_data(path)
    )
    print("actions", "rewards", "observations", "dones")
    print(actions.shape, rewards.shape, observations.shape, dones.shape)
    data = add_data(observations, actions, rewards, dones, init_poses, infos)
    with h5py.File(fname, "w") as dataset:
        npify(data)
        for k in data:
            dataset.create_dataset(k, data=data[k], compression="gzip")
    info_path = "infos.pkl"
    with open(info_path, "wb") as f:
        pickle.dump(infos, f)
