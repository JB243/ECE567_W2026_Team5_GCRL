import flax.linen as nn
import jax
import jax.numpy as jnp


def energy_fn(name, x, y):
    if name == "norm":
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)
    elif name == "dot":
        return jnp.sum(x * y, axis=-1)
    elif name == "cosine":
        return jnp.sum(x * y, axis=-1) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + 1e-6)
    elif name == "l2":
        return -jnp.sum((x - y) ** 2, axis=-1)
    else:
        raise ValueError(f"Unknown energy function: {name}")


def _masked_mean(x, mask, valid_count):
    """Weighted mean of x over rows where mask=1."""
    return jnp.sum(x * mask) / valid_count


def contrastive_loss_fn(name, logits, valid_mask):
    """Per-row masked InfoNCE. Invalid rows contribute 0 to the diagonal-positive
    term; their goals are still present as negatives in the logsumexp denominator
    (legitimate random future states)."""
    valid_count = jnp.sum(valid_mask).clip(min=1.0)
    diag = jnp.diag(logits)
    if name == "fwd_infonce":
        per_row = diag - jax.nn.logsumexp(logits, axis=1)
        critic_loss = -_masked_mean(per_row, valid_mask, valid_count)
    elif name == "bwd_infonce":
        per_row = diag - jax.nn.logsumexp(logits, axis=0)
        critic_loss = -_masked_mean(per_row, valid_mask, valid_count)
    elif name == "sym_infonce":
        per_row = (
            2 * diag - jax.nn.logsumexp(logits, axis=1) - jax.nn.logsumexp(logits, axis=0)
        )
        critic_loss = -_masked_mean(per_row, valid_mask, valid_count)
    elif name == "binary_nce":
        # Per-element sigmoid mean; apply row-mask to the diagonal positive term.
        critic_loss = -_masked_mean(jax.nn.sigmoid(diag), valid_mask, valid_count)
    else:
        raise ValueError(f"Unknown contrastive loss function: {name}")
    return critic_loss


def update_actor_and_alpha(config, networks, transitions, training_state, key):
    valid_mask = transitions.extras["valid_mask"]
    valid_count = jnp.sum(valid_mask).clip(min=1.0)

    def actor_loss(actor_params, target_critic_params, log_alpha, transitions, key):
        obs = transitions.observation
        state = obs[:, : config["state_size"]]
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

        # Use EMA target critic params for actor loss
        sa_encoder_params, g_encoder_params = (
            target_critic_params["sa_encoder"],
            target_critic_params["g_encoder"],
        )
        sa_repr = networks["sa_encoder"].apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
        g_repr = networks["g_encoder"].apply(g_encoder_params, goal)

        qf_pi = energy_fn(config["energy_fn"], sa_repr, g_repr)

        per_row = jnp.exp(log_alpha) * log_prob - qf_pi
        actor_loss = _masked_mean(per_row, valid_mask, valid_count)

        return actor_loss, log_prob

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        per_row = jax.lax.stop_gradient(-log_prob - config["target_entropy"])
        alpha_loss = alpha * _masked_mean(per_row, valid_mask, valid_count)
        return alpha_loss

    (actor_loss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
        training_state.actor_state.params,
        training_state.target_critic_params,
        training_state.alpha_state.params["log_alpha"],
        transitions,
        key,
    )
    new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
    new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

    training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)

    metrics = {
        "entropy": -log_prob,
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
    }

    return training_state, metrics


def update_critic(config, networks, transitions, training_state, key):
    valid_mask = transitions.extras["valid_mask"]
    valid_count = jnp.sum(valid_mask).clip(min=1.0)

    def critic_loss(critic_params, transitions, key):
        sa_encoder_params, g_encoder_params = (
            critic_params["sa_encoder"],
            critic_params["g_encoder"],
        )

        state = transitions.observation[:, : config["state_size"]]
        action = transitions.action

        sa_repr = networks["sa_encoder"].apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
        g_repr = networks["g_encoder"].apply(
            g_encoder_params, transitions.observation[:, config["state_size"] :]
        )

        # InfoNCE (masked)
        logits = energy_fn(config["energy_fn"], sa_repr[:, None, :], g_repr[None, :, :])
        critic_loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits, valid_mask)

        # logsumexp regularisation (masked over valid source rows)
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        critic_loss += config["logsumexp_penalty_coeff"] * _masked_mean(logsumexp**2, valid_mask, valid_count)

        I = jnp.eye(logits.shape[0])
        correct = (jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)).astype(jnp.float32)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return critic_loss, (logsumexp, I, correct, logits_pos, logits_neg)

    (loss, (logsumexp, I, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
        critic_loss, has_aux=True
    )(training_state.critic_state.params, transitions, key)
    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)

    # EMA update of target critic params
    tau = config["ema_tau"]
    new_target_critic_params = jax.tree_util.tree_map(
        lambda target, online: tau * online + (1.0 - tau) * target,
        training_state.target_critic_params,
        new_critic_state.params,
    )

    training_state = training_state.replace(
        critic_state=new_critic_state,
        target_critic_params=new_target_critic_params,
    )

    metrics = {
        "categorical_accuracy": _masked_mean(correct, valid_mask, valid_count),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": _masked_mean(logsumexp, valid_mask, valid_count),
        "critic_loss": loss,
    }

    return training_state, metrics
