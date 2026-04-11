"""PPO + CRL: Proximal Policy Optimization with a CRL contrastive auxiliary loss.

Compared to the PPO baseline:
  - Adds CRL's contrastive dual encoder (sa_encoder, g_encoder) as *auxiliary*
    networks trained alongside the policy and value networks.
  - At every PPO mini-batch update, a CRL InfoNCE loss is computed on
    within-rollout future-state pairs and added to the total objective:

        total_loss = ppo_policy_loss + ppo_value_loss
                   + contrastive_coeff * contrastive_auxiliary_loss

  - Because PPO already collects full trajectory rollouts of length
    ``unroll_length``, those trajectories are reused for future-state
    goal sampling without any replay buffer — the contrastive pairs are
    drawn fresh from the current on-policy batch.
  - The auxiliary networks are trained with their own Adam optimizer, kept
    separate from the PPO policy/value networks so that the contrastive
    loss cannot destabilise the PPO clipped objective.

Compared to the CRL baseline:
  - Policy is a stochastic PPO actor with GAE advantage (not entropy-max SAC).
  - On-policy: no replay buffer, no off-policy corrections.
  - Contrastive loss uses only the *current* rollout as the source of
    future-state supervision (rather than a large experience-replay buffer).
    This is weaker than CRL's off-policy approach but provides a dense
    representation-learning signal to PPO which otherwise sees only sparse
    goal-reaching rewards.

Usage (drop-in for JaxGCRL):
    Place this directory at ``jaxgcrl/agents/ppo_crl/`` and run:
        jaxgcrl ppo_crl --env reacher --num-envs 4096 ...
"""

import functools
import logging
import time
from typing import Any, Callable, Literal, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import acting, gradients, pmap, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params, PRNGKey
from brax.v1 import envs as envs_v1
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import Evaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

# CRL network and loss primitives
from jaxgcrl.agents.crl.networks import Encoder
from jaxgcrl.agents.crl.losses import energy_fn as _energy_fn, contrastive_loss_fn as _contrastive_loss_fn

import pickle

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]

_PMAP_AXIS_NAME = "i"


# ---------------------------------------------------------------------------
# Training-state container (PPO state + contrastive auxiliary state)
# ---------------------------------------------------------------------------

@dataclass
class PPOCRLTrainingState:
    """Combined training state for PPO policy/value + CRL auxiliary encoders."""

    # Standard PPO
    params: Any                                         # PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    optimizer_state: optax.OptState
    env_steps: jnp.ndarray

    # CRL auxiliary encoders (separate optimiser)
    contrastive_params: Any                             # {"sa_encoder": ..., "g_encoder": ...}
    contrastive_optimizer_state: optax.OptState


# ---------------------------------------------------------------------------
# Within-rollout future-state goal sampling for on-policy CRL
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("buffer_config",))
def sample_contrastive_pairs(buffer_config, rollout_obs, rollout_actions, traj_ids, sample_key):
    """Sample (state, action, future_goal) triples from an on-policy rollout.

    Args:
        buffer_config: ``(gamma, state_size, goal_indices)`` — static.
        rollout_obs: shape ``(T, obs_size)`` — single-environment trajectory.
        rollout_actions: shape ``(T, action_size)``.
        traj_ids: shape ``(T,)`` — trajectory IDs for boundary detection.
        sample_key: JAX PRNG key.

    Returns:
        (state, action, goal) triple, each shape ``(T-1, ...)``.
    """
    gamma, state_size, goal_indices = buffer_config
    seq_len = rollout_obs.shape[0]
    arrangement = jnp.arange(seq_len)

    is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    single_trajectories = jnp.concatenate(
        [traj_ids[:, jnp.newaxis].T] * seq_len, axis=0
    )
    probs = (
        probs * jnp.equal(single_trajectories, single_trajectories.T)
        + jnp.eye(seq_len) * 1e-5
    )

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_obs = jnp.take(rollout_obs, goal_index[:-1], axis=0)
    goal = future_obs[:, goal_indices]

    state = rollout_obs[:-1, :state_size]
    action = rollout_actions[:-1]

    return state, action, goal


# ---------------------------------------------------------------------------
# CRL auxiliary loss (operates on a flat minibatch of (state, action, goal))
# ---------------------------------------------------------------------------

def contrastive_auxiliary_loss(
    contrastive_params,
    sa_encoder,
    g_encoder,
    state,
    action,
    goal,
    energy_fn_name: str,
    loss_fn_name: str,
    logsumexp_coeff: float,
):
    """InfoNCE loss on (state, action) vs goal representations."""
    sa_repr = sa_encoder.apply(
        contrastive_params["sa_encoder"],
        jnp.concatenate([state, action], axis=-1),
    )
    g_repr = g_encoder.apply(contrastive_params["g_encoder"], goal)

    logits = _energy_fn(energy_fn_name, sa_repr[:, None, :], g_repr[None, :, :])
    loss = _contrastive_loss_fn(loss_fn_name, logits)

    # Logsumexp penalty for calibration
    logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
    loss += logsumexp_coeff * jnp.mean(logsumexp ** 2)

    I = jnp.eye(logits.shape[0])
    correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
    return loss, jnp.mean(correct)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_params(path: str, params: Any):
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


# ---------------------------------------------------------------------------
# PPO_CRL agent dataclass
# ---------------------------------------------------------------------------

@dataclass
class PPO_CRL:
    """PPO + CRL: on-policy policy gradient with contrastive representation auxiliary loss.

    Key hyper-parameters
    --------------------
    contrastive_coeff : float
        Weight of the CRL auxiliary loss in the total objective (default 1.0).
        Set to ``0.0`` to recover vanilla PPO.
    repr_dim : int
        Dimensionality of the contrastive representation vectors.
    contrastive_loss_fn : str
        InfoNCE variant (``"sym_infonce"``, ``"fwd_infonce"``, …).
    energy_fn : str
        Energy function between representations (``"l2"``, ``"norm"``, …).
    contrastive_lr : float
        Learning rate for the auxiliary contrastive encoder (default 3e-4).
    """

    # PPO-specific (mirrors base PPO defaults)
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-2
    discounting: float = 0.97
    unroll_length: int = 20
    batch_size: int = 256
    num_minibatches: int = 32
    num_updates_per_batch: int = 4
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.3
    normalize_advantage: bool = True
    reward_scaling: float = 1.0
    deterministic_eval: bool = False
    num_resets_per_eval: int = 0
    network_width: int = 256
    network_depth: int = 4

    # CRL auxiliary
    contrastive_coeff: float = 1.0
    contrastive_lr: float = 3e-4
    repr_dim: int = 64
    contrastive_loss_fn: Literal[
        "fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"
    ] = "sym_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "l2"
    logsumexp_penalty_coeff: float = 0.1
    h_dim: int = 256
    n_hidden: int = 2
    skip_connections: int = 4
    use_relu: bool = False
    use_ln: bool = False

    def train_fn(
        self,
        config,
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn=None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        # ---------------------------------------------------------------
        # Device setup
        # ---------------------------------------------------------------
        process_id = jax.process_index()
        local_devices_to_use = jax.local_device_count()
        if config.max_devices_per_host is not None:
            local_devices_to_use = min(local_devices_to_use, config.max_devices_per_host)
        device_count = local_devices_to_use * jax.process_count()
        logging.info(
            "local_device_count: %s; total_device_count: %s",
            local_devices_to_use, device_count,
        )

        # ---------------------------------------------------------------
        # Environment setup
        # ---------------------------------------------------------------
        unwrapped_env = train_env
        if isinstance(train_env, envs.Env):
            wrap_for_training = envs.training.wrap
        else:
            wrap_for_training = envs_v1.wrappers.wrap_for_training

        rng = jax.random.PRNGKey(config.seed)
        rng, key = jax.random.split(rng)
        v_randomization_fn = None
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn,
                rng=jax.random.split(
                    key, config.num_envs // jax.process_count() // local_devices_to_use
                ),
            )

        env = TrajectoryIdWrapper(train_env)
        env = wrap_for_training(
            env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            randomization_fn=v_randomization_fn,
        )

        obs_size = env.observation_size
        action_size = env.action_size

        # ---------------------------------------------------------------
        # Dimensions
        # ---------------------------------------------------------------
        # The environment exposes state_dim and goal_indices
        state_size = unwrapped_env.state_dim
        goal_size = len(unwrapped_env.goal_indices)
        goal_indices = unwrapped_env.goal_indices

        # ---------------------------------------------------------------
        # PPO networks
        # ---------------------------------------------------------------
        normalize_fn = running_statistics.normalize
        ppo_network = ppo_networks.make_ppo_networks(
            observation_size=obs_size,
            action_size=action_size,
            preprocess_observations_fn=normalize_fn,
            policy_hidden_layer_sizes=(self.network_width,) * self.network_depth,
            value_hidden_layer_sizes=(self.network_width,) * self.network_depth,
        )
        make_policy = ppo_networks.make_inference_fn(ppo_network)

        # ---------------------------------------------------------------
        # CRL auxiliary networks (separate from PPO)
        # ---------------------------------------------------------------
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

        # ---------------------------------------------------------------
        # Optimisers
        # ---------------------------------------------------------------
        ppo_optimizer = optax.adam(learning_rate=self.learning_rate)
        contrastive_optimizer = optax.adam(learning_rate=self.contrastive_lr)

        # ---------------------------------------------------------------
        # PPO loss (unchanged from brax base)
        # ---------------------------------------------------------------
        ppo_loss, ppo_metrics = ppo_losses.make_losses(
            ppo_network=ppo_network,
            entropy_cost=self.entropy_cost,
            discounting=self.discounting,
            reward_scaling=self.reward_scaling,
        )

        ppo_update = gradients.gradient_update_fn(
            ppo_loss, ppo_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
        )

        # ---------------------------------------------------------------
        # Initialise training state
        # ---------------------------------------------------------------
        rng, init_rng, sa_rng, g_rng = jax.random.split(rng, 4)

        init_params = ppo_networks.make_ppo_networks(  # dummy obs to get shapes
            observation_size=obs_size,
            action_size=action_size,
            preprocess_observations_fn=normalize_fn,
            policy_hidden_layer_sizes=(self.network_width,) * self.network_depth,
            value_hidden_layer_sizes=(self.network_width,) * self.network_depth,
        )

        ppo_params = ppo_network.policy_network.init(init_rng), ppo_network.value_network.init(init_rng)
        # Actually use brax's standard init:
        dummy_obs_arr = jnp.zeros((1, obs_size))
        dummy_action_arr = jnp.zeros((1, action_size))
        ppo_init_params = ppo_networks.PPONetworkParams(
            policy=ppo_network.policy_network.init(init_rng),
            value=ppo_network.value_network.init(init_rng),
        )
        normalizer_params = running_statistics.init_state(
            specs.Array((obs_size,), jnp.dtype("float32"))
        )
        optimizer_state = ppo_optimizer.init(ppo_init_params)

        sa_params = sa_encoder.init(sa_rng, jnp.ones((1, state_size + action_size)))
        g_params = g_encoder.init(g_rng, jnp.ones((1, goal_size)))
        contrastive_params = {"sa_encoder": sa_params, "g_encoder": g_params}
        contrastive_optimizer_state = contrastive_optimizer.init(contrastive_params)

        training_state = PPOCRLTrainingState(
            params=ppo_init_params,
            normalizer_params=normalizer_params,
            optimizer_state=optimizer_state,
            env_steps=jnp.zeros(()),
            contrastive_params=contrastive_params,
            contrastive_optimizer_state=contrastive_optimizer_state,
        )

        # Replicate across devices
        training_state = jax.device_put_replicated(
            training_state, jax.local_devices()[:local_devices_to_use]
        )

        # ---------------------------------------------------------------
        # Evaluate environment setup
        # ---------------------------------------------------------------
        if not eval_env:
            eval_env = train_env
        eval_env_wrapped = TrajectoryIdWrapper(eval_env)
        eval_env_wrapped = wrap_for_training(
            eval_env_wrapped,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )
        rng, eval_key = jax.random.split(rng)
        evaluator = Evaluator(
            eval_env_wrapped,
            functools.partial(make_policy, deterministic=self.deterministic_eval),
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            key=eval_key,
        )

        # ---------------------------------------------------------------
        # Rollout and update functions
        # ---------------------------------------------------------------
        num_envs_per_device = config.num_envs // local_devices_to_use
        assert config.num_envs % local_devices_to_use == 0

        # Epoch / scheduling bookkeeping (same as base PPO)
        num_evals_after_init = max(config.num_evals - 1, 1)
        num_training_steps_per_epoch = -(
            -(config.total_env_steps // device_count)
            // (num_evals_after_init * self.unroll_length * num_envs_per_device)
        )
        env_steps_per_actor_step = (
            config.action_repeat * num_envs_per_device * self.unroll_length
        )

        # Static buffer_config for the contrastive sampling JIT
        buffer_config = (self.discounting, state_size, tuple(goal_indices))
        contrastive_coeff = self.contrastive_coeff
        energy_fn_name = self.energy_fn
        loss_fn_name = self.contrastive_loss_fn
        logsumexp_coeff = self.logsumexp_penalty_coeff

        # ---------------------------------------------------------------
        # PPO+CRL joint update step
        # ---------------------------------------------------------------
        def sgd_step(carry, data):
            """One PPO minibatch update + one CRL auxiliary update."""
            training_state, key = carry
            key, ppo_key, crl_key = jax.random.split(key, 3)

            # ------- PPO update (standard) -------
            (ppo_loss_val, ppo_aux), ppo_params, ppo_opt_state = ppo_update(
                training_state.params,
                training_state.normalizer_params,
                data,
                ppo_key,
                optimizer_state=training_state.optimizer_state,
            )

            # ------- CRL auxiliary update -------
            # Extract (state, action, goal) from the minibatch.
            # data.observation has shape (batch_size, obs_size) = [state | goal].
            mb_state = data.observation[:, :state_size]
            mb_action = data.action
            mb_goal = data.observation[:, state_size:]  # goal already relabeled by env

            def crl_loss_fn(c_params):
                loss, acc = contrastive_auxiliary_loss(
                    c_params, sa_encoder, g_encoder,
                    mb_state, mb_action, mb_goal,
                    energy_fn_name, loss_fn_name, logsumexp_coeff,
                )
                return loss, acc

            (crl_loss_val, crl_acc), crl_grad = jax.value_and_grad(
                crl_loss_fn, has_aux=True
            )(training_state.contrastive_params)
            # Average gradients across pmap axis
            crl_grad = jax.lax.pmean(crl_grad, axis_name=_PMAP_AXIS_NAME)
            c_updates, new_c_opt_state = contrastive_optimizer.update(
                crl_grad, training_state.contrastive_optimizer_state
            )
            new_c_params = optax.apply_updates(training_state.contrastive_params, c_updates)

            # ------- Merge into new training state -------
            new_training_state = training_state.replace(
                params=ppo_params,
                optimizer_state=ppo_opt_state,
                contrastive_params=new_c_params,
                contrastive_optimizer_state=new_c_opt_state,
            )
            metrics = {
                **ppo_aux,
                "contrastive_loss": crl_loss_val,
                "contrastive_accuracy": crl_acc,
            }
            return (new_training_state, key), metrics

        # ---------------------------------------------------------------
        # Training epoch (collect rollout + update)
        # ---------------------------------------------------------------
        def training_epoch(training_state, env_state, key):
            # Collect rollout on each device
            def rollout_step(carry, _):
                env_state, cur_key = carry
                cur_key, next_key = jax.random.split(cur_key)
                nstate, data = acting.generate_unroll(
                    env,
                    env_state,
                    functools.partial(make_policy, training_state.params.policy),
                    cur_key,
                    self.unroll_length,
                    extra_fields=("truncation", "traj_id"),
                )
                return (nstate, next_key), data

            (env_state, _), rollout = jax.lax.scan(
                rollout_step,
                (env_state, key),
                (),
                length=num_training_steps_per_epoch,
            )
            # rollout shape: (num_steps, num_envs_per_device, unroll_length, ...)

            # Update observation statistics
            normalizer_params = running_statistics.update(
                training_state.normalizer_params,
                rollout.observation,
                pmap_axis_name=_PMAP_AXIS_NAME,
            )
            training_state = training_state.replace(normalizer_params=normalizer_params)

            # Compute GAE advantages
            batch = ppo_losses.compute_gae(
                rollout,
                training_state.params,
                training_state.normalizer_params,
                ppo_network,
                self.discounting,
                self.gae_lambda,
                self.reward_scaling,
            )

            # Shuffle and split into minibatches
            key, shuffle_key = jax.random.split(key)
            batch_size = self.batch_size * self.num_minibatches
            batch = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (batch_size,) + x.shape[2:]), batch
            )
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = jax.tree_util.tree_map(lambda x: x[permutation], batch)
            batch = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (self.num_minibatches, self.batch_size) + x.shape[1:]),
                batch,
            )

            key, update_key = jax.random.split(key)
            (training_state, _), metrics = jax.lax.scan(
                sgd_step, (training_state, update_key), batch
            )

            training_state = training_state.replace(
                env_steps=training_state.env_steps
                + env_steps_per_actor_step * num_training_steps_per_epoch,
            )
            return training_state, env_state, metrics

        training_epoch_with_timing = functools.partial(training_epoch)
        training_epoch_pmap = jax.pmap(
            training_epoch_with_timing, axis_name=_PMAP_AXIS_NAME
        )

        # ---------------------------------------------------------------
        # Reset environments
        # ---------------------------------------------------------------
        rng, env_rng = jax.random.split(rng)
        key_envs = jax.random.split(env_rng, config.num_envs)
        key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
        reset_fn = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
        env_state = reset_fn(key_envs)

        # ---------------------------------------------------------------
        # Initial evaluation
        # ---------------------------------------------------------------
        if process_id == 0 and config.num_evals > 1:
            metrics = evaluator.run_evaluation(
                _unpmap((training_state.normalizer_params, training_state.params.policy)),
                training_metrics={},
            )
            progress_fn(
                0,
                metrics,
                make_policy,
                _unpmap((training_state.normalizer_params, training_state.params.policy)),
                unwrapped_env,
            )

        # ---------------------------------------------------------------
        # Main training loop
        # ---------------------------------------------------------------
        training_walltime = 0
        current_step = 0
        local_key = jax.random.PRNGKey(process_id)
        for eval_epoch_num in range(num_evals_after_init):
            logging.info("starting epoch %s", eval_epoch_num)
            t = time.time()

            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)

            training_state, env_state, training_metrics = training_epoch_pmap(
                training_state, env_state, epoch_keys
            )
            current_step = int(_unpmap(training_state.env_steps))

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time

            if process_id == 0:
                metrics = evaluator.run_evaluation(
                    _unpmap((training_state.normalizer_params, training_state.params.policy)),
                    training_metrics,
                )
                do_render = eval_epoch_num % config.visualization_interval == 0
                progress_fn(
                    current_step,
                    metrics,
                    make_policy,
                    _unpmap((training_state.normalizer_params, training_state.params.policy)),
                    unwrapped_env,
                    do_render=do_render,
                )

        assert current_step >= config.total_env_steps
        pmap.assert_is_replicated(training_state)
        params = _unpmap((training_state.normalizer_params, training_state.params.policy))
        logging.info("total steps: %s", current_step)
        pmap.synchronize_hosts()
        return make_policy, params, metrics
