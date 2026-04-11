"""TD3 + CRL: Twin Delayed DDPG with a Contrastive RL critic.

Compared to the TD3 baseline:
  - Replaces the standard twin Q-networks (Bellman backup) with CRL's
    contrastive dual encoder (sa_encoder, g_encoder) trained via InfoNCE.
  - Replaces sparse reward with CRL's discounted future-state goal sampling.
  - No bootstrapping — the critic is a pure contrastive objective.

Compared to the CRL baseline:
  - Uses a *deterministic* policy instead of a stochastic Gaussian actor.
    This removes the entropy term and the temperature coefficient alpha,
    making the actor gradient purely: ``-E[energy(sa_encoder(s,π(s)), g_encoder(g))]``.
  - Adds *delayed policy updates*: the actor and target-actor networks are
    updated only every ``policy_delay`` critic steps (default 2), which
    stabilises training and prevents the actor from over-fitting to a noisy
    critic early in learning.
  - Adds *target policy smoothing*: when using the target actor to evaluate
    the "future" representation during critic training, small Gaussian noise
    is added to the target action before encoding.  This regularises the
    critic against sharp peaks in the energy landscape.
  - Keeps CRL's HER-compatible future-state goal sampling.

Usage (drop-in for JaxGCRL):
    Place this directory at ``jaxgcrl/agents/td3_crl/`` and run:
        jaxgcrl td3_crl --env reacher --num-envs 1024 ...
"""

import functools
import logging
import pickle
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
from flax.linen.initializers import variance_scaling
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

# Reuse CRL's encoder and loss primitives
from jaxgcrl.agents.crl.networks import Encoder
from jaxgcrl.agents.crl.losses import energy_fn as _energy_fn, contrastive_loss_fn as _contrastive_loss_fn

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


# ---------------------------------------------------------------------------
# Deterministic actor network (tanh-squashed MLP, no log_std head)
# ---------------------------------------------------------------------------

class DeterministicActor(nn.Module):
    """Deterministic tanh-squashed policy for TD3_CRL."""

    action_size: int
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        normalize = nn.LayerNorm() if self.use_ln else (lambda z: z)
        activation = nn.relu if self.use_relu else nn.swish

        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)
            if self.skip_connections:
                if i == 0:
                    skip = x
                if i > 0 and i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        x = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return nn.tanh(x)


# ---------------------------------------------------------------------------
# Training-state (carries target actor params separately for delayed update)
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    target_actor_params: Any        # Polyak-averaged actor params
    critic_state: TrainState


# ---------------------------------------------------------------------------
# Transition tuple
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


# ---------------------------------------------------------------------------
# Goal relabeling: CRL discounted future-state sampling (same as crl.py)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("buffer_config",))
def flatten_batch(buffer_config, transition, sample_key):
    """CRL-style discounted future-state goal sampling."""
    gamma, state_size, goal_indices = buffer_config

    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
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


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def update_critic_td3crl(config, networks, transitions, training_state, key):
    """InfoNCE critic update with target-policy smoothing.

    The target actor adds small Gaussian noise to its action before encoding,
    which regularises the critic against adversarial sharp peaks.
    """
    smoothing_noise = config["smoothing_noise"]
    noise_clip = config["noise_clip"]
    energy_fn_name = config["energy_fn"]
    loss_fn_name = config["contrastive_loss_fn"]
    logsumexp_coeff = config["logsumexp_penalty_coeff"]
    state_size = config["state_size"]
    goal_indices = config["goal_indices"]

    def critic_loss(critic_params, transitions, key):
        sa_encoder_params = critic_params["sa_encoder"]
        g_encoder_params = critic_params["g_encoder"]

        state = transitions.observation[:, :state_size]
        action = transitions.action

        # ---- positive pair: (current state, action) vs goal ---------
        sa_repr = networks["sa_encoder"].apply(
            sa_encoder_params, jnp.concatenate([state, action], axis=-1)
        )
        g_repr = networks["g_encoder"].apply(
            g_encoder_params, transitions.observation[:, state_size:]
        )
        logits = _energy_fn(energy_fn_name, sa_repr[:, None, :], g_repr[None, :, :])
        loss = _contrastive_loss_fn(loss_fn_name, logits)

        # ---- logsumexp regularisation --------------------------------
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        loss += logsumexp_coeff * jnp.mean(logsumexp ** 2)

        # ---- target-policy smoothing: noise on future action ---------
        # We also train a secondary pair using the target actor's action
        # on the future state, giving an additional contrastive signal
        # that is stabilised by the Polyak-averaged actor.
        future_state = transitions.extras["future_state"]
        future_goal = future_state[:, goal_indices]
        future_obs = jnp.concatenate(
            [future_state, future_goal], axis=-1
        )
        # Apply target actor (deterministic) + smoothing noise
        target_action_raw = networks["target_actor"].apply(
            training_state.target_actor_params, future_obs
        )
        noise = jnp.clip(
            jax.random.normal(key, shape=target_action_raw.shape) * smoothing_noise,
            -noise_clip,
            noise_clip,
        )
        target_action = jnp.clip(target_action_raw + noise, -1.0, 1.0)

        sa_repr_target = networks["sa_encoder"].apply(
            sa_encoder_params, jnp.concatenate([future_state, target_action], axis=-1)
        )
        logits_target = _energy_fn(
            energy_fn_name, sa_repr_target[:, None, :], g_repr[None, :, :]
        )
        loss_target = _contrastive_loss_fn(loss_fn_name, logits_target)
        loss += 0.5 * loss_target   # weight the auxiliary target-smoothing loss

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return loss, (logsumexp, correct, logits_pos, logits_neg)

    (loss, (logsumexp, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
        critic_loss, has_aux=True
    )(training_state.critic_state.params, transitions, key)
    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
    training_state = training_state.replace(critic_state=new_critic_state)

    metrics = {
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp.mean(),
        "critic_loss": loss,
    }
    return training_state, metrics


def update_actor_td3crl(config, networks, transitions, training_state):
    """Deterministic policy gradient using the contrastive Q-value (energy)."""
    state_size = config["state_size"]
    energy_fn_name = config["energy_fn"]
    goal_indices = config["goal_indices"]

    def actor_loss(actor_params, critic_params, transitions):
        state = transitions.observation[:, :state_size]
        future_state = transitions.extras["future_state"]
        goal = future_state[:, goal_indices]
        obs = jnp.concatenate([state, goal], axis=1)

        action = networks["actor"].apply(actor_params, obs)  # deterministic

        sa_encoder_params = critic_params["sa_encoder"]
        g_encoder_params = critic_params["g_encoder"]
        sa_repr = networks["sa_encoder"].apply(
            sa_encoder_params, jnp.concatenate([state, action], axis=-1)
        )
        g_repr = networks["g_encoder"].apply(g_encoder_params, goal)
        qf_pi = _energy_fn(energy_fn_name, sa_repr, g_repr)

        # Maximise energy → minimise negative energy
        return -jnp.mean(qf_pi)

    loss, grad = jax.value_and_grad(actor_loss)(
        training_state.actor_state.params,
        training_state.critic_state.params,
        transitions,
    )
    new_actor_state = training_state.actor_state.apply_gradients(grads=grad)
    training_state = training_state.replace(actor_state=new_actor_state)
    return training_state, {"actor_loss": loss}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_params(path: str, params: Any):
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


# ---------------------------------------------------------------------------
# Polyak soft-update
# ---------------------------------------------------------------------------

def soft_update(target_params, online_params, tau: float):
    return jax.tree_util.tree_map(
        lambda t, o: (1.0 - tau) * t + tau * o, target_params, online_params
    )


# ---------------------------------------------------------------------------
# TD3_CRL agent dataclass
# ---------------------------------------------------------------------------

@dataclass
class TD3_CRL:
    """TD3 + CRL: deterministic actor with delayed updates + contrastive critic.

    Key hyper-parameters
    --------------------
    policy_delay : int
        Actor and target-actor are updated only every ``policy_delay`` critic
        gradient steps (default 2, same as vanilla TD3).
    smoothing_noise : float
        Std dev of the Gaussian noise added to the target actor's action when
        computing the secondary target-smoothing contrastive loss.
    noise_clip : float
        Maximum absolute value of the smoothing noise.
    exploration_noise : float
        Std dev of noise added to actions during *data collection* (ε-exploration).
    tau : float
        Polyak averaging rate for the target actor (default 0.005).
    """

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4

    batch_size: int = 256
    discounting: float = 0.99
    tau: float = 0.005
    logsumexp_penalty_coeff: float = 0.1

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 62

    h_dim: int = 256
    n_hidden: int = 2
    skip_connections: int = 4
    use_relu: bool = False
    use_ln: bool = False

    repr_dim: int = 64
    contrastive_loss_fn: Literal[
        "fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"
    ] = "sym_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "l2"

    # TD3-specific
    policy_delay: int = 2
    smoothing_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1

    train_step_multiplier: int = 1

    def train_fn(
        self,
        config,
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn=None,
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
        logging.info("num_training_steps_per_epoch: %d", num_training_steps_per_epoch)

        # ---------------------------------------------------------------
        # RNG
        # ---------------------------------------------------------------
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buf_key, eval_key, env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)

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
        actor = DeterministicActor(
            action_size=action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        actor_params_init = actor.init(actor_key, np.ones([1, obs_size]))
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params_init,
            tx=optax.adam(learning_rate=self.policy_lr),
        )
        target_actor_params = actor_params_init  # initialised identically

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
                "sa_encoder": sa_encoder.init(sa_key, np.ones([1, state_size + action_size])),
                "g_encoder": g_encoder.init(g_key, np.ones([1, goal_size])),
            },
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor_state,
            target_actor_params=target_actor_params,
            critic_state=critic_state,
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
            extras={"state_extras": {"truncation": 0.0, "traj_id": 0.0}},
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
        buffer_state = jax.jit(replay_buffer.init)(buf_key)

        # ---------------------------------------------------------------
        # Actor step (deterministic + exploration noise)
        # ---------------------------------------------------------------
        exploration_noise = self.exploration_noise

        def actor_step_with_noise(actor_params, env, env_state, key, extra_fields):
            action_det = actor.apply(actor_params, env_state.obs)  # tanh output
            noise = jax.random.normal(key, shape=action_det.shape) * exploration_noise
            actions = jnp.clip(action_det + noise, -1.0, 1.0)
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        def deterministic_actor_step(training_state, env, env_state, extra_fields):
            actions = actor.apply(training_state.actor_state.params, env_state.obs)
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
        def get_experience(actor_params, env_state, buffer_state, key):
            @jax.jit
            def f(carry, _t):
                env_state, cur_key = carry
                cur_key, next_key = jax.random.split(cur_key)
                env_state, transition = actor_step_with_noise(
                    actor_params, train_env, env_state, cur_key,
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
                    training_state.actor_state.params, env_state, buffer_state, key
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
        )
        # The "target_actor" network instance is the same arch; we pass its
        # params via training_state.target_actor_params at call time.
        networks_dict = dict(
            actor=actor,
            target_actor=actor,       # same module, different params
            sa_encoder=sa_encoder,
            g_encoder=g_encoder,
        )
        buffer_config = (self.discounting, state_size, tuple(train_env.goal_indices))
        policy_delay = self.policy_delay
        tau = self.tau

        @jax.jit
        def update_networks(carry, transitions):
            training_state, key = carry
            key, critic_key = jax.random.split(key)
            grad_step = training_state.gradient_steps

            # Always update critic
            training_state, critic_metrics = update_critic_td3crl(
                context, networks_dict, transitions, training_state, critic_key
            )

            # Delayed actor + target update
            def do_actor_update(ts):
                ts, actor_metrics = update_actor_td3crl(context, networks_dict, transitions, ts)
                new_target = soft_update(ts.target_actor_params, ts.actor_state.params, tau)
                ts = ts.replace(target_actor_params=new_target)
                return ts, actor_metrics

            def skip_actor_update(ts):
                return ts, {"actor_loss": jnp.zeros(())}

            training_state, actor_metrics = jax.lax.cond(
                grad_step % policy_delay == 0,
                do_actor_update,
                skip_actor_update,
                training_state,
            )

            training_state = training_state.replace(
                gradient_steps=training_state.gradient_steps + 1
            )
            metrics = {**critic_metrics, **actor_metrics}
            return (training_state, key), metrics

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            exp_key1, exp_key2, samp_key, train_key = jax.random.split(key, 4)

            env_state, buffer_state = get_experience(
                training_state.actor_state.params, env_state, buffer_state, exp_key1
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step
            )

            buffer_state, transitions = replay_buffer.sample(buffer_state)

            batch_keys = jax.random.split(samp_key, transitions.observation.shape[0])
            transitions = jax.vmap(
                functools.partial(flatten_batch, buffer_config),
                in_axes=(0, 0),
            )(transitions, batch_keys)
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
            key=eval_key,
        )

        # ---------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------
        training_walltime = 0
        logging.info("starting TD3_CRL training....")
        params = None
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

            make_policy = lambda param: lambda obs, rng: actor.apply(param, obs)
            params = (
                training_state.actor_state.params,
                training_state.target_actor_params,
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
