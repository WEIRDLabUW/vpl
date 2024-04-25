import os
from functools import partial
import pickle

from absl import app, flags
import d4rl
from flax.training import checkpoints
from ml_collections import config_flags
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tqdm
import wandb
import torch
import orbax.checkpoint as ocp

from jaxrl_m.evaluation import supply_rng, evaluate, sample_evaluate
import jaxrl_m.envs
from jaxrl_m.dataset import Dataset
import jaxrl_m.learners.iql as learner
import jaxrl_m.learners.d4rl_utils as d4rl_utils
from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict
from pref_learn.utils.plot_utils import plot_observation_rewards
from pref_learn.models.utils import get_datasets

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", None, "Logging dir (if not None, save params).")
flags.DEFINE_integer("seed", np.random.choice(1000000), "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Eval interval.")
flags.DEFINE_integer("save_interval", 250000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_bool("save_video", False, "Save video of the agent.")

flags.DEFINE_bool("use_reward_model", False, "Use reward model.")
flags.DEFINE_string("model_type", "MLP", "Path to reward model.")
flags.DEFINE_string("ckpt", "", "Path to reward model.")
flags.DEFINE_integer("fix_mode", -1, "Fix mode for a multimodal environment.")
flags.DEFINE_bool("append_goal", False, "Append goal to obs.")
flags.DEFINE_bool("vae_sampling", False, "Sample from VAE.")
flags.DEFINE_integer("comp_size", 1000, "Size of comparison set.")
flags.DEFINE_string("vae_norm", "none", "Normalize VAE latent.")
flags.DEFINE_string("preference_dataset_path", "", "Path to preference dataset.")

wandb_config = default_wandb_config()
wandb_config.update(
    {
        "project": os.environ.get("WANDB_PROJECT", "iql_test"),
        "group": os.environ.get("WANDB_GROUP", "iql"),
        "name": "iql_{env_name}",
    }
)

config_flags.DEFINE_config_dict("wandb", wandb_config, lock_config=False)
config_flags.DEFINE_config_file(
    "config", "configs/default_config.py", lock_config=False
)


def plot_traj(env, dataset):
    if "maze" not in FLAGS.env_name:
        return
    traj_idx = d4rl_utils.new_get_trj_idx(dataset)
    os.makedirs("./traj_plots", exist_ok=True)

    for i, (start, end) in enumerate(traj_idx):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        obs = dataset["observations"][start : end + 1, :2]
        r = dataset["rewards"][start : end + 1]
        m = dataset["masks"][start : end + 1]
        sc = ax.scatter(obs[:, 0], obs[:, 1], c=r)
        cb = plt.colorbar(sc, ax=ax)
        for goal in env.goals:
            ax.scatter(goal[0], goal[1], s=50, c="green", marker="*")
        plt.title(f"{np.argwhere(1-m)}, {r.shape[0]}")
        wandb.log({"traj": wandb.Image(fig)})
        plt.close(fig)
        if i > 100:
            break

def load_reward_model(ckpt):
    with open(os.path.join(ckpt, "best_model.pt"), "rb") as f:
        reward_model = torch.load(f)
    return reward_model


def update_observation(observation, mode, append_goal, model_type, latent):
    if append_goal:
        observation = np.concatenate([observation, np.array([mode])], axis=-1)
    elif "VAE" in model_type:
        observation = np.concatenate([observation, latent], axis=-1)
    return observation


def get_modes_list(env):
    if FLAGS.fix_mode < 0:
        if hasattr(env, "get_num_modes"):
            n = env.get_num_modes()
            if n > 1:
                return range(n)
            else:
                return [env.mode]
        else:
            return [-1]
        # return range(env.get_num_modes())
    return [FLAGS.fix_mode]


def evaluate_fn(agent, env, reward_model, num_episodes, comp_obs=None):
    policy_fn = partial(supply_rng(agent.sample_actions), temperature=0.0)
    eval_reward_fn = None
    eval_metrics = {}
    for n in get_modes_list(env):
        if hasattr(env, "set_mode"):
            env.set_mode(n)
        latent = None
        # if FLAGS.use_reward_model and "VAE" in FLAGS.model_type:
        #     latent = reward_model.biased_latents[n, 0]
        #     eval_reward_fn = partial(
        #         reward_fn, reward_model=reward_model, comp_obs=comp_obs, latent=latent
        #     )
        eval_info = evaluate(
            policy_fn,
            env,
            num_episodes=num_episodes,
            save_video=FLAGS.save_video,
            name=f"video_mode_{n}",
            obs_fn=partial(
                update_observation,
                mode=n,
                append_goal=FLAGS.append_goal,
                model_type=FLAGS.model_type,
                latent=latent,
            ),
            # reward_fn=eval_reward_fn,
        )
        eval_metrics.update(
            {f"evaluation/mode_{n}_{k}": v for k, v in eval_info.items()}
        )
    return eval_metrics

def sample_evaluate_fn(env, agent, reward_model, preference_dataset, eval_episodes):
    policy_fn = partial(supply_rng(agent.sample_actions), temperature=0.0)
    eval_metrics = sample_evaluate(
        policy_fn,
        env,
        reward_model,
        preference_dataset,
        num_episodes=eval_episodes,
        model_type=FLAGS.model_type,
        obs_fn=partial(
            update_observation,
            append_goal=FLAGS.append_goal,
            model_type=FLAGS.model_type,
        ),
    )
    return eval_metrics

def reward_fn(obs, reward_model, latent, comp_obs):
    obs = torch.tensor(obs).float().to(next(reward_model.parameters()).device)[None, :2]
    z = torch.tensor(latent).float().to(next(reward_model.parameters()).device)[None]
    with torch.no_grad():
        if "Classifier" in FLAGS.model_type:
            rewards = reward_model.decode(obs, comp_obs, z)
        else:
            rewards = reward_model.decode(obs, z)
    return rewards.cpu().numpy()

def main(_):
    print(FLAGS.config.to_dict())
    env = d4rl_utils.make_env(FLAGS.env_name)

    reward_model = None
    if FLAGS.use_reward_model:
        reward_model = load_reward_model(FLAGS.ckpt)
    dataset, comp_obs = d4rl_utils.get_dataset(
        env,
        use_reward_model=FLAGS.use_reward_model,
        append_goal=FLAGS.append_goal,
        fix_mode=FLAGS.fix_mode,
        model_type=FLAGS.model_type,
        reward_model=reward_model,
        vae_sampling=FLAGS.vae_sampling,
        comp_size=FLAGS.comp_size,
        vae_norm=FLAGS.vae_norm,
        terminate_on_end="sort" in FLAGS.env_name,
    )
    setup_wandb(FLAGS.config.to_dict(), **FLAGS.wandb)
    if FLAGS.save_dir is not None:
        FLAGS.save_dir = os.path.join(
            FLAGS.save_dir,
            wandb.run.project,
            wandb.config.exp_prefix,
            wandb.config.experiment_id,
        )
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        print(f"Saving config to {FLAGS.save_dir}/config.pkl")
        with open(os.path.join(FLAGS.save_dir, "config.pkl"), "wb") as f:
            pickle.dump(get_flag_dict(), f)
        checkpointer = ocp.PyTreeCheckpointer()
            
    example_batch = dataset.sample(1)
    agent = learner.create_learner(
        FLAGS.seed,
        example_batch["observations"],
        example_batch["actions"],
        max_steps=FLAGS.max_steps,
        **FLAGS.config,
    )

    preference_dataset = None
    if FLAGS.use_reward_model and "VAE" in FLAGS.model_type:
        if hasattr(env, "reward_observation_space"):
            obs_dim = env.reward_observation_space.shape[0]
        else:
            obs_dim = env.observation_space.shape[0]
        (
            _,
            _,
            _,
            preference_dataset,
            set_len,
            _,
            _,
        ) = get_datasets(
            FLAGS.preference_dataset_path,
            obs_dim,
            env.action_space.shape[0],
            FLAGS.batch_size,
            reward_model.annotation_size
        )
        assert set_len == reward_model.annotation_size

    plot_traj(env, dataset)

    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):
        batch = dataset.sample(FLAGS.batch_size)
        agent, update_info = agent.update(batch)
        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0:
            # Sampling evaluation
            eval_metrics =  sample_evaluate_fn(env, agent, reward_model, preference_dataset, FLAGS.eval_episodes)
            wandb.log({f"sampling/{k}":v for k,v in eval_metrics.items()}, step=i)
            eval_metrics = evaluate_fn(
                agent, env, reward_model, FLAGS.eval_episodes, comp_obs
            )
            fig_dict = {
                "train_batch": wandb.Image(
                    plot_observation_rewards(
                        batch["observations"], batch["rewards"], no_norm=True
                    )
                )
            }
            eval_metrics.update(fig_dict)
            wandb.log(eval_metrics, step=i)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            checkpoints.save_checkpoint(FLAGS.save_dir, agent, i, orbax_checkpointer=checkpointer)

    # Final evaluation
    eval_info = sample_evaluate_fn(env, agent, reward_model, preference_dataset, 100)
    for k, v in eval_info.items():
        wandb.log({f"final-sampling-{k}": v})

    eval_metrics = evaluate_fn(agent, env, reward_model, num_episodes=100, comp_obs=comp_obs)
    for k, v in eval_metrics.items():
        wandb.log({f"final-{k}": v})


if __name__ == "__main__":
    app.run(main)
