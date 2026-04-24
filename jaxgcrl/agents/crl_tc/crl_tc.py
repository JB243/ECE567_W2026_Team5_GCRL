"""CRL with Twin Critics (TC).

Two independent (sa_encoder, g_encoder) pairs are trained with separate InfoNCE
losses. The actor uses min(energy1, energy2) — the TD3 overestimation fix
applied to the contrastive energy function. Everything else is identical to
the base CRL (SAC actor, entropy temperature, discounted future-state HER).
"""

import logging
import pickle
import random
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
from jaxgcrl.agents.crl.networks import Actor, Encoder
from jaxgcrl.agents.crl.losses import (
    energy_fn as _energy_fn,
    contrastive_loss_fn as _contrastive_loss_fn,
)
from jaxgcrl.agents.crl.crl import save_params


def flatten_batch(buffer_config, transition, sample_key):
    """Discounted future-state goal sampling (no JIT — called inside jit training_step)."""
    gamma, state_size, goal_indices = buffer_config

    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len, axis=0
    )
    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state = jnp.take(transition.observation, goal_index[:-1], axis=0)
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    goal = future_state[:, goal_indices]
    future_state = future_state[:, :state_size]
    state = transition.observation[:-1, :state_size]
    new_obs = jnp.concatenate([state, goal], axis=1)

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
            "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
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

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


@dataclass
class TrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState   # params: sa_encoder1, g_encoder1, sa_encoder2, g_encoder2
    alpha_state: TrainState


class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


# ---------------------------------------------------------------------------
# Twin-critic loss functions
# ---------------------------------------------------------------------------

def update_critic_tc(config, networks, transitions, training_state, key):
    """Train two encoder pairs independently; sum their InfoNCE losses."""

    def critic_loss(critic_params, transitions, key):
        state = transitions.observation[:, :config["state_size"]]
        action = transitions.action
        sa_input = jnp.concatenate([state, action], axis=-1)
        goal = transitions.observation[:, config["state_size"]:]

        # Pair 1
        sa_repr1 = networks["sa_encoder1"].apply(critic_params["sa_encoder1"], sa_input)
        g_repr1 = networks["g_encoder1"].apply(critic_params["g_encoder1"], goal)
        logits1 = _energy_fn(config["energy_fn"], sa_repr1[:, None, :], g_repr1[None, :, :])
        lse1 = jax.nn.logsumexp(logits1 + 1e-6, axis=1)
        loss1 = _contrastive_loss_fn(config["contrastive_loss_fn"], logits1)
        loss1 += config["logsumexp_penalty_coeff"] * jnp.mean(lse1 ** 2)

        # Pair 2
        sa_repr2 = networks["sa_encoder2"].apply(critic_params["sa_encoder2"], sa_input)
        g_repr2 = networks["g_encoder2"].apply(critic_params["g_encoder2"], goal)
        logits2 = _energy_fn(config["energy_fn"], sa_repr2[:, None, :], g_repr2[None, :, :])
        lse2 = jax.nn.logsumexp(logits2 + 1e-6, axis=1)
        loss2 = _contrastive_loss_fn(config["contrastive_loss_fn"], logits2)
        loss2 += config["logsumexp_penalty_coeff"] * jnp.mean(lse2 ** 2)

        total_loss = loss1 + loss2

        # Diagnostic metrics from pair 1 (representative)
        I = jnp.eye(logits1.shape[0])
        correct = jnp.argmax(logits1, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits1 * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits1 * (1 - I)) / jnp.sum(1 - I)

        return total_loss, (lse1, correct, logits_pos, logits_neg)

    (loss, (logsumexp, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
        critic_loss, has_aux=True
    )(training_state.critic_state.params, transitions, key)
    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
    training_state = training_state.replace(critic_state=new_critic_state)

    return training_state, {
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp.mean(),
        "critic_loss": loss,
    }


def update_actor_and_alpha_tc(config, networks, transitions, training_state, key):
    """Actor maximises min(energy1, energy2) — prevents exploiting noisy peaks."""

    def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
        state = transitions.observation[:, :config["state_size"]]
        future_state = transitions.extras["future_state"]
        goal = future_state[:, config["goal_indices"]]
        observation = jnp.concatenate([state, goal], axis=1)

        means, log_stds = networks["actor"].apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob = log_prob.sum(-1)

        sa_input = jnp.concatenate([state, action], axis=-1)

        sa_repr1 = networks["sa_encoder1"].apply(critic_params["sa_encoder1"], sa_input)
        g_repr1 = networks["g_encoder1"].apply(critic_params["g_encoder1"], goal)
        q1 = _energy_fn(config["energy_fn"], sa_repr1, g_repr1)

        sa_repr2 = networks["sa_encoder2"].apply(critic_params["sa_encoder2"], sa_input)
        g_repr2 = networks["g_encoder2"].apply(critic_params["g_encoder2"], goal)
        q2 = _energy_fn(config["energy_fn"], sa_repr2, g_repr2)

        qf_pi = jnp.minimum(q1, q2)
        loss = jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi)
        return loss, log_prob

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        return jnp.mean(alpha * jax.lax.stop_gradient(-log_prob - config["target_entropy"]))

    (actor_loss_val, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
        training_state.actor_state.params,
        training_state.critic_state.params,
        training_state.alpha_state.params["log_alpha"],
        transitions, key,
    )
    new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

    alpha_loss_val, alpha_grad = jax.value_and_grad(alpha_loss)(
        training_state.alpha_state.params, log_prob
    )
    new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

    training_state = training_state.replace(
        actor_state=new_actor_state, alpha_state=new_alpha_state
    )
    return training_state, {
        "entropy": -log_prob,
        "actor_loss": actor_loss_val,
        "alpha_loss": alpha_loss_val,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class CRL_TC:
    """CRL with Twin Critic encoders.

    Two independent (sa_encoder, g_encoder) pairs are trained; the actor
    policy gradient uses min(energy1, energy2) to avoid overestimating the
    contrastive Q-value signal.
    """

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    discounting: float = 0.99
    logsumexp_penalty_coeff: float = 0.1
    train_step_multiplier: int = 1
    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 62
    h_dim: int = 256
    n_hidden: int = 2
    skip_connections: int = 4
    use_relu: bool = False
    repr_dim: int = 64
    use_ln: bool = False
    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    def check_config(self, config):
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0

    def train_fn(
        self,
        config,
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn=None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        self.check_config(config)

        unwrapped_env = train_env
        train_env = TrajectoryIdWrapper(train_env)
        train_env = envs.training.wrap(
            train_env, episode_length=config.episode_length, action_repeat=config.action_repeat
        )
        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = envs.training.wrap(
            eval_env, episode_length=config.episode_length, action_repeat=config.action_repeat
        )

        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = np.ceil(self.min_replay_size / self.unroll_length)
        num_training_steps_per_epoch = (config.total_env_steps - num_prefill_env_steps) // (
            config.num_evals * env_steps_per_actor_step
        ) + 1 # originally + 0
        assert num_training_steps_per_epoch > 0

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buf_key, eval_key, env_key, actor_key, sa_key1, g_key1, sa_key2, g_key2 = jax.random.split(key, 9)

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size

        # Actor
        actor = Actor(
            action_size=action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, np.ones([1, obs_size])),
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        # Twin critics
        encoder_cfg = dict(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        sa_encoder1 = Encoder(**encoder_cfg)
        g_encoder1 = Encoder(**encoder_cfg)
        sa_encoder2 = Encoder(**encoder_cfg)
        g_encoder2 = Encoder(**encoder_cfg)

        critic_state = TrainState.create(
            apply_fn=None,
            params={
                "sa_encoder1": sa_encoder1.init(sa_key1, np.ones([1, state_size + action_size])),
                "g_encoder1":  g_encoder1.init(g_key1,  np.ones([1, goal_size])),
                "sa_encoder2": sa_encoder2.init(sa_key2, np.ones([1, state_size + action_size])),
                "g_encoder2":  g_encoder2.init(g_key2,  np.ones([1, goal_size])),
            },
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        target_entropy = -0.5 * action_size
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": jnp.asarray(0.0, dtype=jnp.float32)},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
        )

        dummy_transition = Transition(
            observation=jnp.zeros((obs_size,)),
            action=jnp.zeros((action_size,)),
            reward=0.0, discount=0.0,
            extras={"state_extras": {"truncation": 0.0, "traj_id": 0.0}},
        )

        def jit_wrap(buf):
            buf.insert_internal = jax.jit(buf.insert_internal)
            buf.sample_internal = jax.jit(buf.sample_internal)
            return buf

        replay_buffer = jit_wrap(TrajectoryUniformSamplingQueue(
            max_replay_size=self.max_replay_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=self.batch_size,
            num_envs=config.num_envs,
            episode_length=config.episode_length,
        ))
        buffer_state = jax.jit(replay_buffer.init)(buf_key)

        def deterministic_actor_step(training_state, env, env_state, extra_fields):
            means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
            actions = nn.tanh(means)
            nstate = env.step(env_state, actions)
            return nstate, Transition(
                observation=env_state.obs, action=actions,
                reward=nstate.reward, discount=1 - nstate.done,
                extras={"state_extras": {x: nstate.info[x] for x in extra_fields}},
            )

        def actor_step(actor_state, env, env_state, key, extra_fields):
            means, log_stds = actor.apply(actor_state.params, env_state.obs)
            stds = jnp.exp(log_stds)
            actions = nn.tanh(means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype))
            nstate = env.step(env_state, actions)
            return nstate, Transition(
                observation=env_state.obs, action=actions,
                reward=nstate.reward, discount=1 - nstate.done,
                extras={"state_extras": {x: nstate.info[x] for x in extra_fields}},
            )

        @jax.jit
        def get_experience(actor_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, _t):
                env_state, cur_key = carry
                cur_key, next_key = jax.random.split(cur_key)
                env_state, transition = actor_step(
                    actor_state, train_env, env_state, cur_key,
                    extra_fields=("truncation", "traj_id"),
                )
                return (env_state, next_key), transition

            (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=self.unroll_length)
            buffer_state = replay_buffer.insert(buffer_state, data)
            return env_state, buffer_state

        def prefill_replay_buffer(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, _):
                ts, es, bs, k = carry
                k, new_k = jax.random.split(k)
                es, bs = get_experience(ts.actor_state, es, bs, k)
                ts = ts.replace(env_steps=ts.env_steps + env_steps_per_actor_step)
                return (ts, es, bs, new_k), ()

            return jax.lax.scan(
                f, (training_state, env_state, buffer_state, key), (), length=num_prefill_actor_steps
            )[0]

        buffer_config = (self.discounting, state_size, tuple(train_env.goal_indices))

        @jax.jit
        def update_networks(carry, transitions):
            training_state, key = carry
            key, critic_key, actor_key = jax.random.split(key, 3)

            context = dict(
                **vars(self), **vars(config),
                state_size=state_size, action_size=action_size,
                goal_size=goal_size, obs_size=obs_size,
                goal_indices=train_env.goal_indices,
                target_entropy=target_entropy,
            )
            networks = dict(
                actor=actor,
                sa_encoder1=sa_encoder1, g_encoder1=g_encoder1,
                sa_encoder2=sa_encoder2, g_encoder2=g_encoder2,
            )

            training_state, actor_metrics = update_actor_and_alpha_tc(
                context, networks, transitions, training_state, actor_key
            )
            training_state, critic_metrics = update_critic_tc(
                context, networks, transitions, training_state, critic_key
            )
            training_state = training_state.replace(
                gradient_steps=training_state.gradient_steps + 1
            )
            return (training_state, key), {**actor_metrics, **critic_metrics}

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
            batch_keys = jax.random.split(samp_key, transitions.observation.shape[0])
            transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
                buffer_config, transitions, batch_keys
            )
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
            )
            perm = jax.random.permutation(exp_key2, len(transitions.observation))
            transitions = jax.tree_util.tree_map(lambda x: x[perm], transitions)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]), transitions
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
                f, (training_state, env_state, buffer_state, key), (),
                length=num_training_steps_per_epoch,
            )
            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            return training_state, env_state, buffer_state, metrics

        key, prefill_key = jax.random.split(key)
        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key
        )

        evaluator = ActorEvaluator(
            deterministic_actor_step, eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_key,
        )

        training_walltime = 0
        logging.info("starting CRL_TC training....")
        for ne in range(config.num_evals):
            t = time.time()
            key, epoch_key = jax.random.split(key)
            training_state, env_state, buffer_state, metrics = training_epoch(
                training_state, env_state, buffer_state, epoch_key
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            current_step = int(training_state.env_steps.item())
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": current_step,
                **{f"training/{k}": v for k, v in metrics.items()},
            }
            metrics = evaluator.run_evaluation(training_state, metrics)
            logging.info("step: %d", current_step)

            make_policy = lambda param: lambda obs, rng: actor.apply(param, obs)
            params = (
                training_state.alpha_state.params,
                training_state.actor_state.params,
                training_state.critic_state.params,
            )
            progress_fn(
                current_step, metrics, make_policy,
                training_state.actor_state.params, unwrapped_env,
                do_render=(ne % config.visualization_interval == 0),
            )
            if config.checkpoint_logdir:
                save_params(f"{config.checkpoint_logdir}/step_{current_step}.pkl", params)

        assert int(training_state.env_steps.item()) >= config.total_env_steps
        return make_policy, params, metrics
