import os
from collections import defaultdict

import absl.app
import absl.flags
import gym
import numpy as np
import torch

import jaxrl_m.envs
from pref_learn.utils.utils import (
    define_flags_with_default,
    set_random_seed,
    get_user_flags,
    WandBLogger,
    prefix_metrics,
)
from pref_learn.models.utils import get_datasets, Annealer, EarlyStopper
from pref_learn.models.vae import VAEModel
from pref_learn.models.mlp import MLPModel, CategoricalModel, MeanVarianceModel
import pref_learn.utils.plot_utils as putils

FLAGS_DEF = define_flags_with_default(
    env="maze2d-target-v0",  # can change
    comment="",
    data_seed=42,
    batch_size=256,
    early_stop=False,
    min_delta=3e-4,
    patience=10,
    lr=1e-3,
    model_type="MLP",  # can change
    # MLP
    hidden_dim=256,
    # Categorical
    num_atoms=10,
    r_min=0,
    r_max=1,
    entropy_coeff=0.1,
    # Mean Var
    variance_penalty=0.0,
    # VAE
    latent_dim=32,
    kl_weight=1.0,
    learned_prior=False,
    flow_prior=False,
    use_annealing=False,
    annealer_baseline=0.0,
    annealer_type="cosine",
    annealer_cycles=4,
    # Training
    n_epochs=500,
    eval_freq=50,
    save_freq=50,
    device="cuda",
    # Dataset
    dataset_path="",
    logging=WandBLogger.get_default_config(),
    seed=42,
    # plotting
    debug_plots=True,
    plot_observations=False,
    reward_scaling=1.0
)


def log_metrics(metrics, epoch, logger):
    for key, val in metrics.items():
        if isinstance(val, list):
            metrics[key] = np.mean(val)
    logger.log(metrics, step=epoch)


def main(_):
    FLAGS = absl.flags.FLAGS
    assert os.path.exists(FLAGS.dataset_path), "You must provide a dataset path."
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    save_dir = FLAGS.logging.output_dir + "/" + FLAGS.env
    save_dir += "/" + str(FLAGS.model_type) + "/"

    FLAGS.logging.group = f"{FLAGS.env}_{FLAGS.model_type}"
    assert FLAGS.comment, "You must leave your comment for logging experiment."
    FLAGS.logging.group += f"_{FLAGS.comment}"
    FLAGS.logging.experiment_id = FLAGS.logging.group + f"_s{FLAGS.seed}"
    save_dir += f"{FLAGS.comment}" + "/"
    save_dir += "s" + str(FLAGS.seed)
    FLAGS.logging.output_dir = save_dir
    wb_logger = WandBLogger(FLAGS.logging, variant=variant)

    gym_env = gym.make(FLAGS.env)
    gym_env.seed(FLAGS.seed)
    gym_env.action_space.seed(FLAGS.seed)
    gym_env.observation_space.seed(FLAGS.seed)
    set_random_seed(FLAGS.seed)
    if hasattr(gym_env, "reward_observation_space"):
        observation_dim = gym_env.reward_observation_space.shape[0]
    else:
        observation_dim = gym_env.observation_space.shape[0]
    action_dim = gym_env.action_space.shape[0]

    (
        train_loader,
        test_loader,
        train_dataset,
        eval_dataset,
        len_set,
        len_query,
        obs_dim,
    ) = get_datasets(FLAGS.dataset_path, observation_dim, action_dim, FLAGS.batch_size)

    if FLAGS.model_type == "MLP":
        reward_model = MLPModel(obs_dim, FLAGS.hidden_dim)
    elif FLAGS.model_type == "Categorical":
        reward_model = CategoricalModel(
            input_dim=obs_dim,
            hidden_dim=FLAGS.hidden_dim,
            n_atoms=FLAGS.num_atoms,
            r_min=FLAGS.r_min,
            r_max=FLAGS.r_max,
            entropy_coeff=FLAGS.entropy_coeff,
        )
    elif FLAGS.model_type == "MeanVar":
        reward_model = MeanVarianceModel(
            input_dim=obs_dim,
            hidden_dim=FLAGS.hidden_dim,
            variance_penalty=FLAGS.variance_penalty,
        )
    elif FLAGS.model_type == "VAE":
        annealer = None
        if FLAGS.use_annealing:
            annealer = Annealer(
                total_steps=FLAGS.n_epochs // FLAGS.annealer_cycles,
                shape=FLAGS.annealer_type,
                baseline=FLAGS.annealer_baseline,
                cyclical=FLAGS.annealer_cycles > 1,
            )
        reward_model = VAEModel(
            encoder_input=len_set * (2 * observation_dim * len_query + 1),
            decoder_input=(obs_dim + FLAGS.latent_dim),
            latent_dim=FLAGS.latent_dim,
            hidden_dim=FLAGS.hidden_dim,
            annotation_size=len_set,
            size_segment=len_query,
            kl_weight=FLAGS.kl_weight,
            learned_prior=FLAGS.learned_prior,
            flow_prior=FLAGS.flow_prior,
            annealer=annealer,
            reward_scaling=FLAGS.reward_scaling
        )
    else:
        raise NotImplementedError

    device = FLAGS.device
    reward_model = reward_model.to(device)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=FLAGS.lr)
    early_stop = EarlyStopper(FLAGS.patience, FLAGS.min_delta)
    best_criteria = None
    for epoch in range(FLAGS.n_epochs):
        metrics = defaultdict(list)
        metrics["epoch"] = epoch

        for batch in train_loader:
            optimizer.zero_grad()
            observations = batch["observations"].to(device).float()
            observations_2 = batch["observations_2"].to(device).float()
            labels = batch["labels"].to(device).float()
            loss, batch_metrics = reward_model(observations, observations_2, labels)
            loss.backward()
            optimizer.step()

            for key, val in prefix_metrics(batch_metrics, "train").items():
                metrics[key].append(val)

        if epoch % FLAGS.eval_freq == 0:
            for batch in test_loader:
                with torch.no_grad():
                    observations = batch["observations"].to(device).float()
                    observations_2 = batch["observations_2"].to(device).float()
                    labels = batch["labels"].to(device).float()
                    loss, batch_metrics = reward_model(
                        observations, observations_2, labels
                    )

                    for key, val in prefix_metrics(batch_metrics, "eval").items():
                        metrics[key].append(val)

            if FLAGS.debug_plots:
                if FLAGS.model_type == "MLP":
                    fig_dict = putils.plot_mlp(gym_env, reward_model)
                elif FLAGS.model_type == "Categorical" or FLAGS.model_type == "MeanVar":
                    fig_dict = putils.plot_mlp_samples(gym_env, reward_model)
                else:
                    fig_dict = putils.plot_vae(gym_env, reward_model, eval_dataset)
                # import pdb; pdb.set_trace()

                metrics.update(prefix_metrics(fig_dict, "debug_plots"))

            criteria = np.mean(metrics["eval/loss"])

            if best_criteria is None:
                best_criteria = criteria
                torch.save(reward_model, save_dir + f"/best_model.pt")

            if criteria < best_criteria:
                torch.save(reward_model, save_dir + f"/best_model.pt")
                best_criteria = criteria

            if FLAGS.early_stop and early_stop.early_stop(criteria):
                log_metrics(metrics, epoch, wb_logger)
                torch.save(reward_model, save_dir + f"/model_{epoch}.pt")
                break

        if epoch % FLAGS.save_freq == 0:
            torch.save(reward_model, save_dir + f"/model_{epoch}.pt")

        if FLAGS.model_type=="VAE" and FLAGS.use_annealing:
            reward_model.annealer.step()

        log_metrics(metrics, epoch, wb_logger)


if __name__ == "__main__":
    absl.app.run(main)
