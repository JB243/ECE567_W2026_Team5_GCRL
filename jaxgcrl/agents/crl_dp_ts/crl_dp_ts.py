"""CRL with Delayed Policy updates + Target Smoothing (DP+TS).

Combines:
  - EMA target g_encoder; actor uses stop_grad(target_g) for stable gradients.
  - Actor + alpha updated only every `policy_delay` critic steps.
  - EMA update runs every critic step regardless of policy_delay so the target
    tracks the online encoder continuously even when the actor is frozen.
"""

import logging
import random
import time
from typing import Any, Callable, Literal, NamedTuple, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue
from jaxgcrl.agents.crl.networks import Actor, Encoder
from jaxgcrl.agents.crl.losses import update_critic
from jaxgcrl.agents.crl.crl import save_params
from jaxgcrl.agents.crl_ts.crl_ts import update_actor_and_alpha_ts


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
    critic_state: TrainState
    alpha_state: TrainState
    target_g_encoder_params: Any   # EMA copy of critic_state.params["g_encoder"]


class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


@dataclass
class CRL_DP_TS:
    """CRL with Delayed Policy updates and Target Smoothing.

    The EMA target g_encoder stabilises the actor's goal representations;
    the delayed update schedule gives the critic time to converge before the
    policy follows it. EMA runs every step; the actor update is gated.
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
    policy_delay: int = 2
    tau: float = 0.005

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
        key, buf_key, eval_key, env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size

        actor = Actor(
            action_size=action_size, network_width=self.h_dim, network_depth=self.n_hidden,
            skip_connections=self.skip_connections, use_relu=self.use_relu,
        )
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, np.ones([1, obs_size])),
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        sa_encoder = Encoder(
            repr_dim=self.repr_dim, network_width=self.h_dim, network_depth=self.n_hidden,
            skip_connections=self.skip_connections, use_relu=self.use_relu, use_ln=self.use_ln,
        )
        g_encoder = Encoder(
            repr_dim=self.repr_dim, network_width=self.h_dim, network_depth=self.n_hidden,
            skip_connections=self.skip_connections, use_relu=self.use_relu, use_ln=self.use_ln,
        )
        g_params_init = g_encoder.init(g_key, np.ones([1, goal_size]))

        critic_state = TrainState.create(
            apply_fn=None,
            params={
                "sa_encoder": sa_encoder.init(sa_key, np.ones([1, state_size + action_size])),
                "g_encoder": g_params_init,
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
            target_g_encoder_params=g_params_init,
        )

        dummy_transition = Transition(
            observation=jnp.zeros((obs_size,)), action=jnp.zeros((action_size,)),
            reward=0.0, discount=0.0,
            extras={"state_extras": {"truncation": 0.0, "traj_id": 0.0}},
        )

        def jit_wrap(buf):
            buf.insert_internal = jax.jit(buf.insert_internal)
            buf.sample_internal = jax.jit(buf.sample_internal)
            return buf

        replay_buffer = jit_wrap(TrajectoryUniformSamplingQueue(
            max_replay_size=self.max_replay_size, dummy_data_sample=dummy_transition,
            sample_batch_size=self.batch_size, num_envs=config.num_envs,
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
        policy_delay = self.policy_delay
        tau = self.tau

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
            networks = dict(actor=actor, sa_encoder=sa_encoder, g_encoder=g_encoder)

            batch_size = transitions.observation.shape[0]

            # Actor uses stop-grad EMA target g_encoder; gated by policy_delay
            def do_actor_update(ts):
                return update_actor_and_alpha_ts(context, networks, transitions, ts, actor_key)

            def skip_actor_update(ts):
                return ts, {
                    "entropy": jnp.zeros((batch_size,)),
                    "actor_loss": jnp.zeros(()),
                    "alpha_loss": jnp.zeros(()),
                    "log_alpha": ts.alpha_state.params["log_alpha"],
                }

            training_state, actor_metrics = jax.lax.cond(
                training_state.gradient_steps % policy_delay == 0,
                do_actor_update, skip_actor_update, training_state,
            )

            # Critic: online g_encoder trained every step
            training_state, critic_metrics = update_critic(
                context, networks, transitions, training_state, critic_key
            )

            # EMA: runs every step (independent of policy_delay)
            new_target_g = jax.tree_util.tree_map(
                lambda t, o: (1.0 - tau) * t + tau * o,
                training_state.target_g_encoder_params,
                training_state.critic_state.params["g_encoder"],
            )
            training_state = training_state.replace(
                gradient_steps=training_state.gradient_steps + 1,
                target_g_encoder_params=new_target_g,
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
            num_eval_envs=config.num_eval_envs, episode_length=config.episode_length, key=eval_key,
        )

        training_walltime = 0
        logging.info("starting CRL_DP_TS training....")
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
                "training/sps": sps, "training/walltime": training_walltime,
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
