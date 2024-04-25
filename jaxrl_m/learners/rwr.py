"""Implementations of algorithms for continuous control."""

from jaxrl_m.typing import *

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState
from jaxrl_m.networks import Policy
import flax


def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class RWRAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    actor: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    @jax.jit
    def update(agent, batch: Batch) -> InfoDict:
        def actor_loss_fn(actor_params):
            r = batch["next_rewards"]
            exp_a = jnp.exp(r * agent.config["temperature"])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = agent.actor(batch["observations"], params=actor_params)
            log_probs = dist.log_prob(batch["actions"])
            actor_loss = -(exp_a * log_probs).mean()

            return actor_loss, {"actor_loss": actor_loss, "r": r}

        new_actor, actor_info = agent.actor.apply_loss_fn(
            loss_fn=actor_loss_fn, has_aux=True
        )

        return agent.replace(actor=new_actor), {**actor_info}

    @jax.jit
    def sample_actions(
        agent, observations: np.ndarray, *, seed: PRNGKey, temperature: float = 1.0
    ) -> jnp.ndarray:
        actions = agent.actor(observations, temperature=temperature).sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions


def create_learner(
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    actor_lr: float = 3e-4,
    hidden_dims: Sequence[int] = (256, 256),
    discount: float = 0.99,
    tau: float = 0.005,
    expectile: float = 0.8,
    temperature: float = 0.1,
    max_steps: Optional[int] = None,
    opt_decay_schedule: str = "cosine",
    **kwargs
):

    print("Extra kwargs:", kwargs)

    rng = jax.random.PRNGKey(seed)
    rng, actor_key = jax.random.split(rng, 2)

    action_dim = actions.shape[-1]
    actor_def = Policy(
        hidden_dims,
        action_dim=action_dim,
        log_std_min=-5.0,
        state_dependent_std=False,
        tanh_squash_distribution=False,
    )

    if opt_decay_schedule == "cosine":
        schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
        actor_tx = optax.chain(
            optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)
        )
    else:
        actor_tx = optax.adam(learning_rate=actor_lr)

    actor_params = actor_def.init(actor_key, observations)["params"]
    actor = TrainState.create(actor_def, actor_params, tx=actor_tx)

    config = flax.core.FrozenDict(
        dict(
            discount=discount,
            temperature=temperature,
            expectile=expectile,
            target_update_rate=tau,
        )
    )

    return RWRAgent(
        rng,
        actor=actor,
        config=config,
    )


# def load_learner(
#     seed, model_path, discount, temperature, expectile, tau, **kwargs
# ) -> IQLAgent:
#     import orbax.checkpoint as ocp
#     from flax.training import checkpoints

#     checkpointer = ocp.PyTreeCheckpointer()
#     state = checkpoints.restore_checkpoint(
#         ckpt_dir=model_path, target=None, orbax_checkpointer=checkpointer
#     )
#     rng = jax.random.PRNGKey(seed)
#     rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)
#     # actor = state["actor"]
#     hidden_dims = (256, 256)  # kwargs.get("hidden_dims")
#     action_dim = 8  # kwargs.get("action_dim")
#     actor_def = Policy(
#         hidden_dims,
#         action_dim=action_dim,
#         log_std_min=-5.0,
#         state_dependent_std=False,
#         tanh_squash_distribution=False,
#     )
#     actor = TrainState.create(
#         actor_def, state["actor"]["params"], tx=optax.adam(learning_rate=3e-4)
#     )

#     critic_def = ensemblize(Critic, num_qs=2)(hidden_dims)
#     critic = TrainState.create(
#         critic_def, state["critic"]["params"], tx=optax.adam(learning_rate=3e-4)
#     )
#     value_def = ValueCritic(hidden_dims)
#     value = TrainState.create(
#         value_def, state["value"]["params"], tx=optax.adam(learning_rate=3e-4)
#     )
#     target_critic = TrainState.create(critic_def, state["critic"]["params"])
#     # critic = state["critic"]
#     # value = state["value"]
#     # target_critic = state["target_critic"]
#     config = flax.core.FrozenDict(
#         dict(
#             discount=discount,
#             temperature=temperature,
#             expectile=expectile,
#             target_update_rate=tau,
#         )
#     )
#     return IQLAgent(rng, critic, target_critic, value, actor, config)
