import os
import pickle

import absl.app
import absl.flags
import gym
from tqdm import tqdm

import jaxrl_m.envs
from jaxrl_m.learners.d4rl_utils import get_dataset
from pref_learn.utils.utils import (
    define_flags_with_default,
    set_random_seed,
)
from pref_learn.utils.data_utils import (
    sample_from_env,
    get_queries_from_multi,
    get_labels,
)
from pref_learn.utils.plot_utils import plot_observations

FLAGS_DEF = define_flags_with_default(
    env="maze2d-pointmass-v0",
    sample_from_env=True,
    data_dir="./pref_datasets",
    data_seed=42,
    num_query=100,
    query_len=1,
    set_len=32,
    relabel=False,
    dataset_path="",
)


def main(_):
    FLAGS = absl.flags.FLAGS
    # use fixed seed for collecting segments.
    set_random_seed(FLAGS.data_seed)
    base_path = os.path.join(FLAGS.data_dir, FLAGS.env)
    os.makedirs(base_path, exist_ok=True)

    gym_env = gym.make(FLAGS.env)
    if FLAGS.relabel:
        assert os.path.exists(FLAGS.dataset_path)
        dataset = pickle.load(open(FLAGS.dataset_path, "rb"))
    else:
        if FLAGS.sample_from_env:
            dataset, query_path, all_obs = sample_from_env(
                gym_env, FLAGS.num_query, FLAGS.set_len, FLAGS.query_len, base_path
            )
        else:
            dataset = get_dataset(gym_env)
            dataset, query_path, all_obs = get_queries_from_multi(
                gym_env,
                dataset,
                FLAGS.num_query,
                FLAGS.query_len,
                FLAGS.set_len,
                base_path,
            )
        print("Saved queries at: ", query_path)

    for i in tqdm(range(len(dataset["observations"]))):
        seg_reward_1, seg_reward_2 = gym_env.get_preference_rewards(
            dataset["observations"][i], dataset["observations_2"][i]
        )
        dataset["labels"][i] = get_labels(seg_reward_1, seg_reward_2)

    relabelled_path = str(query_path).replace("queries", "relabelled_queries")
    with open(relabelled_path, "wb") as f:
        pickle.dump(dataset, f)
    print("Saved relabelled queries at: ", query_path)

    plot_observations(all_obs, query_path)


if __name__ == "__main__":
    absl.app.run(main)
