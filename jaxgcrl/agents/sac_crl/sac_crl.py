"""SAC + CRL: Soft Actor-Critic with Contrastive RL critic.

Compared to the SAC baseline:
  - Replaces the standard Q-network (Bellman backup) with CRL's contrastive
    dual encoder (sa_encoder, g_encoder) trained via InfoNCE loss.
  - Replaces sparse reward with CRL's discounted future-state goal sampling.
  - No bootstrapping — the critic is a pure representation matching objective.

Compared to the CRL baseline:
  - Adds Hindsight Experience Replay (HER) as a complementary goal-relabeling
    strategy (controlled by ``her_ratio``).  At each training step, a fraction
    ``her_ratio`` of goals is replaced by the achieved goal at the end of the
    same episode (HER) while the rest keeps CRL's discounted future-state
    sampling.  Mixing the two widens goal coverage: HER anchors the policy to
    actually-reached states while CRL provides dense intermediate supervision.
  - Exposes SAC-aligned hyper-parameter names (``policy_lr``, ``critic_lr``,
    ``alpha_lr``) so the two ablations can be compared with identical configs.

Usage (drop-in for JaxGCRL):
    Place this directory at ``jaxgcrl/agents/sac_crl/`` and run:
        jaxgcrl sac_crl --env reacher --num-envs 1024 ...
"""

import functools
import logging
import time
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

# Reuse CRL's network building-blocks and loss functions unchanged.
from jaxgcrl.agents.crl.networks import Actor, Encoder
from jaxgcrl.agents.crl.losses import (
    energy_fn as _energy_fn,
    contrastive_loss_fn as _contrastive_loss_fn,
    update_critic,
    update_actor_and_alpha,
)

import pickle

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


# ---------------------------------------------------------------------------
# Training-state container
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """All mutable state carried across gradient steps."""

    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState


# ---------------------------------------------------------------------------
# Transition tuple (same schema as CRL so the replay buffer is compatible)
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


# ---------------------------------------------------------------------------
# Goal-relabeling: CRL future-state sampling + optional HER mixing
# ---------------------------------------------------------------------------

def flatten_batch(buffer_config, her_ratio, transition, sample_key):
    """Relabel goals by mixing CRL's discounted future-state sampling with HER.

    Args:
        buffer_config: ``(gamma, state_size, goal_indices)`` tuple (static).
        her_ratio: fraction in [0, 1] of transitions that use HER instead of
            CRL goal sampling.  ``0.0`` = pure CRL, ``1.0`` = pure HER.
        transition: a single trajectory of shape ``(episode_len, ...)``.
        sample_key: JAX PRNG key.
    """
    gamma, state_size, goal_indices = buffer_config
    key_crl, key_mix = jax.random.split(sample_key)

    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)

    # ------------------------------------------------------------------
    # Build trajectory-aware probability matrix (same as CRL)
    # ------------------------------------------------------------------
    is_future_mask = jnp.array(
        arrangement[:, None] < arrangement[None], dtype=jnp.float32
    )
    discount = gamma ** jnp.array(
        arrangement[None] - arrangement[:, None], dtype=jnp.float32
    )
    probs = is_future_mask * discount

    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
        axis=0,
    )
    probs = (
        probs * jnp.equal(single_trajectories, single_trajectories.T)
        + jnp.eye(seq_len) * 1e-5
    )

    # ------------------------------------------------------------------
    # CRL goal: sample a future state proportional to discounted distance
    # ------------------------------------------------------------------
    goal_index_crl = jax.random.categorical(key_crl, jnp.log(probs))
    future_obs_crl = jnp.take(transition.observation, goal_index_crl[:-1], axis=0)
    future_action_crl = jnp.take(transition.action, goal_index_crl[:-1], axis=0)
    goal_crl = future_obs_crl[:, goal_indices]
    future_state_crl = future_obs_crl[:, :state_size]

    # ------------------------------------------------------------------
    # HER goal: use the goal that was *actually achieved* at the end of
    # the same episode (truncation step), falling back to the current
    # observation goal when no truncation is present in the window.
    # ------------------------------------------------------------------
    final_step_mask = is_future_mask * jnp.equal(
        single_trajectories, single_trajectories.T
    ) + jnp.eye(seq_len) * 1e-5
    final_step_mask = jnp.logical_and(
        final_step_mask,
        transition.extras["state_extras"]["truncation"][None, :],
    )
    non_zero_columns = jnp.nonzero(final_step_mask, size=seq_len)[1]
    # Fall back to current index when no truncation found
    new_goals_idx = jnp.where(non_zero_columns == 0, arrangement, non_zero_columns)
    binary_mask = jnp.logical_and(non_zero_columns, non_zero_columns)

    # HER relabeled goal (achieved goal at episode end)
    her_obs = transition.observation[new_goals_idx[:-1]]
    goal_her = (
        binary_mask[:-1, None] * her_obs[:, goal_indices]
        + jnp.logical_not(binary_mask[:-1])[:, None]
        * transition.observation[:-1, state_size:]
    )
    future_state_her = her_obs[:, :state_size]
    future_action_her = transition.action[new_goals_idx[:-1]]

    # ------------------------------------------------------------------
    # Mix: for each sample independently draw from CRL vs. HER
    # ------------------------------------------------------------------
    use_her = jax.random.uniform(key_mix, shape=(seq_len - 1,)) < her_ratio

    goal = jnp.where(use_her[:, None], goal_her, goal_crl)
    future_state = jnp.where(use_her[:, None], future_state_her, future_state_crl)
    future_action = jnp.where(use_her[:, None], future_action_her, future_action_crl)

    state = transition.observation[:-1, :state_size]
    new_obs = jnp.concatenate([state, goal], axis=1)

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(
                transition.extras["state_extras"]["truncation"][:-1]
            ),
            "traj_id": jnp.squeeze(
                transition.extras["state_extras"]["traj_id"][:-1]
            ),
        },
        "state": state,
        "future_state": future_state,
        "future_action": future_action,
    }

    return transition._replace(
        observation=jnp.squeeze(new_obs),
        action=jnp.squeeze(transition.action[:-1]),
        reward=jnp.squeeze(transition.reward[:-1]),
        discount=jnp.squeeze(transition.discount[:-1]),
        extras=extras,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        return pickle.loads(fin.read())


def save_params(path: str, params: Any):
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


# ---------------------------------------------------------------------------
# SAC_CRL agent dataclass
# ---------------------------------------------------------------------------

@dataclass
class SAC_CRL:
    """SAC + CRL: entropy-regularised actor with a contrastive goal-conditioned critic.

    Key hyper-parameters
    --------------------
    her_ratio : float
        Fraction of replay-buffer samples that get their goal replaced by the
        HER *achieved* goal.  The remaining ``1 - her_ratio`` fraction keeps
        CRL's discounted future-state goal.  Default ``0.5`` gives an equal
        mix; set to ``0.0`` to recover plain CRL behaviour.
    contrastive_loss_fn : str
        InfoNCE variant: ``"sym_infonce"`` (default), ``"fwd_infonce"``,
        ``"bwd_infonce"``, or ``"binary_nce"``.
    energy_fn : str
        Energy (similarity) function between representations:
        ``"l2"`` (default), ``"norm"``, ``"dot"``, or ``"cosine"``.
    """

    # Actor / alpha learning rates
    policy_lr: float = 6e-4
    alpha_lr: float = 3e-4
    # Critic learning rate
    critic_lr: float = 3e-4

    batch_size: int = 256
    discounting: float = 0.99
    logsumexp_penalty_coeff: float = 0.1
    train_step_multiplier: int = 1

    # Replay buffer
    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 62

    # Network architecture (shared actor + encoder)
    h_dim: int = 256
    n_hidden: int = 2
    skip_connections: int = 4
    use_relu: bool = False
    use_ln: bool = False

    # Contrastive critic
    repr_dim: int = 64
    contrastive_loss_fn: Literal[
        "fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"
    ] = "sym_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "l2"

    # HER mixing ratio (0 = pure CRL, 1 = pure HER)
    her_ratio: float = 0.5

    disable_entropy_actor: bool = False

    def train_fn(
        self,
        config,
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        # ---------------------------------------------------------------
        # Environment setup
        # ---------------------------------------------------------------
        unwrapped_env = train_env
        train_env = TrajectoryIdWrapper(train_env)
        train_env = envs.training.wrap(
            train_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )
        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = envs.training.wrap(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = int(np.ceil(self.min_replay_size / self.unroll_length))
        num_training_steps_per_epoch = (
            config.total_env_steps - num_prefill_env_steps
        ) // (config.num_evals * env_steps_per_actor_step) + 1

        logging.info("num_prefill_actor_steps: %d", num_prefill_actor_steps)
        logging.info(
            "num_training_steps_per_epoch: %d", num_training_steps_per_epoch
        )

        # ---------------------------------------------------------------
        # RNG setup
        # ---------------------------------------------------------------
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, actor_key, sa_key, g_key = (
            jax.random.split(key, 7)
        )

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        # ---------------------------------------------------------------
        # Dimensions
        # ---------------------------------------------------------------
        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size

        # ---------------------------------------------------------------
        # Networks
        # ---------------------------------------------------------------
        actor = Actor(
            action_size=action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, np.ones([1, obs_size])),
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        sa_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        g_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        critic_state = TrainState.create(
            apply_fn=None,
            params={
                "sa_encoder": sa_encoder.init(
                    sa_key, np.ones([1, state_size + action_size])
                ),
                "g_encoder": g_encoder.init(g_key, np.ones([1, goal_size])),
            },
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        target_entropy = -0.5 * action_size
        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
        )

        # ---------------------------------------------------------------
        # Replay buffer
        # ---------------------------------------------------------------
        dummy_obs = jnp.zeros((obs_size,))
        dummy_action = jnp.zeros((action_size,))
        dummy_transition = Transition(
            observation=dummy_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                }
            },
        )

        def jit_wrap(buf):
            buf.insert_internal = jax.jit(buf.insert_internal)
            buf.sample_internal = jax.jit(buf.sample_internal)
            return buf

        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        buffer_state = jax.jit(replay_buffer.init)(buffer_key)

        # ---------------------------------------------------------------
        # Actor step helpers
        # ---------------------------------------------------------------
        def deterministic_actor_step(training_state, env, env_state, extra_fields):
            means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
            actions = nn.tanh(means)
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        def actor_step(actor_state, env, env_state, key, extra_fields):
            means, log_stds = actor.apply(actor_state.params, env_state.obs)
            stds = jnp.exp(log_stds)
            actions = nn.tanh(
                means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            )
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        # ---------------------------------------------------------------
        # Experience collection
        # ---------------------------------------------------------------
        @jax.jit
        def get_experience(actor_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, _t):
                env_state, cur_key = carry
                cur_key, next_key = jax.random.split(cur_key)
                env_state, transition = actor_step(
                    actor_state,
                    train_env,
                    env_state,
                    cur_key,
                    extra_fields=("truncation", "traj_id"),
                )
                return (env_state, next_key), transition

            (env_state, _), data = jax.lax.scan(
                f, (env_state, key), (), length=self.unroll_length
            )
            buffer_state = replay_buffer.insert(buffer_state, data)
            return env_state, buffer_state

        def prefill_replay_buffer(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, _):
                training_state, env_state, buffer_state, key = carry
                key, new_key = jax.random.split(key)
                env_state, buffer_state = get_experience(
                    training_state.actor_state, env_state, buffer_state, key
                )
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + env_steps_per_actor_step
                )
                return (training_state, env_state, buffer_state, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

        # ---------------------------------------------------------------
        # Gradient-update step
        # ---------------------------------------------------------------
        context = dict(
            **vars(self),
            **vars(config),
            state_size=state_size,
            action_size=action_size,
            goal_size=goal_size,
            obs_size=obs_size,
            goal_indices=train_env.goal_indices,
            target_entropy=target_entropy,
        )
        networks_dict = dict(
            actor=actor,
            sa_encoder=sa_encoder,
            g_encoder=g_encoder,
        )
        buffer_config = (self.discounting, state_size, tuple(train_env.goal_indices))
        her_ratio = self.her_ratio  # captured as a static value

        @jax.jit
        def update_networks(carry, transitions):
            training_state, key = carry
            key, critic_key, actor_key = jax.random.split(key, 3)

            training_state, actor_metrics = update_actor_and_alpha(
                context, networks_dict, transitions, training_state, actor_key
            )
            training_state, critic_metrics = update_critic(
                context, networks_dict, transitions, training_state, critic_key
            )
            training_state = training_state.replace(
                gradient_steps=training_state.gradient_steps + 1
            )
            metrics = {**actor_metrics, **critic_metrics}
            return (training_state, key), metrics

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            exp_key1, exp_key2, samp_key, train_key = jax.random.split(key, 4)

            env_state, buffer_state = get_experience(
                training_state.actor_state, env_state, buffer_state, exp_key1
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step
            )

            buffer_state, transitions = replay_buffer.sample(buffer_state)

            # Apply SAC_CRL goal relabeling (HER + CRL mix) per trajectory
            batch_keys = jax.random.split(samp_key, transitions.observation.shape[0])
            transitions = jax.vmap(flatten_batch, in_axes=(None, None, 0, 0))(
                buffer_config, her_ratio, transitions, batch_keys
            )

            # Flatten trajectories → individual transitions
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
            )

            # Shuffle
            perm = jax.random.permutation(exp_key2, len(transitions.observation))
            transitions = jax.tree_util.tree_map(lambda x: x[perm], transitions)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                transitions,
            )

            (training_state, _), metrics = jax.lax.scan(
                update_networks, (training_state, train_key), transitions
            )
            return (training_state, env_state, buffer_state), metrics

        @jax.jit
        def training_epoch(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, _t):
                ts, es, bs, k = carry
                k, train_key = jax.random.split(k)
                (ts, es, bs), metrics = training_step(ts, es, bs, train_key)
                return (ts, es, bs, k), metrics

            (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_training_steps_per_epoch,
            )
            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            return training_state, env_state, buffer_state, metrics

        # ---------------------------------------------------------------
        # Pre-fill
        # ---------------------------------------------------------------
        key, prefill_key = jax.random.split(key)
        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key
        )

        # ---------------------------------------------------------------
        # Evaluator
        # ---------------------------------------------------------------
        evaluator = ActorEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        # ---------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------
        training_walltime = 0
        logging.info("starting SAC_CRL training....")
        params = None
        for ne in range(config.num_evals):
            t = time.time()
            key, epoch_key = jax.random.split(key)
            training_state, env_state, buffer_state, metrics = training_epoch(
                training_state, env_state, buffer_state, epoch_key
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(
                lambda x: x.block_until_ready(), metrics
            )

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (
                env_steps_per_actor_step * num_training_steps_per_epoch
            ) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **{f"training/{k}": v for k, v in metrics.items()},
            }
            current_step = int(training_state.env_steps.item())
            metrics = evaluator.run_evaluation(training_state, metrics)
            logging.info("step: %d", current_step)

            make_policy = lambda param: lambda obs, rng: (actor.apply(param, obs)[0], {})
            params = (
                training_state.alpha_state.params,
                training_state.actor_state.params,
                training_state.critic_state.params,
            )

            do_render = ne % config.visualization_interval == 0
            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.actor_state.params,
                unwrapped_env,
                do_render=do_render,
            )

            if config.checkpoint_logdir:
                path = f"{config.checkpoint_logdir}/step_{current_step}.pkl"
                save_params(path, params)

        assert int(training_state.env_steps.item()) >= config.total_env_steps
        logging.info("total steps: %s", int(training_state.env_steps.item()))
        return make_policy, params, metrics
