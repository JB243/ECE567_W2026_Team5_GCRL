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


def contrastive_loss_fn(name, logits):
    if name == "fwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
    elif name == "bwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=0))
    elif name == "sym_infonce":
        critic_loss = -jnp.mean(
            2 * jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1) - jax.nn.logsumexp(logits, axis=0)
        )
    elif name == "binary_nce":
        critic_loss = -jnp.mean(jax.nn.sigmoid(logits))
    else:
        raise ValueError(f"Unknown contrastive loss function: {name}")
    return critic_loss


def update_actor_and_alpha(config, networks, transitions, training_state, key):
    def actor_loss(actor_params, target_critic_params, target_goal_critic_params,
                   log_alpha, transitions, key, env_steps,
                   goal_logit_mean_ema, goal_logit_var_ema,
                   goal_warmup_met_step):
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

        # CRL energy using EMA target critic
        sa_encoder_params, g_encoder_params = (
            target_critic_params["sa_encoder"],
            target_critic_params["g_encoder"],
        )
        sa_repr = networks["sa_encoder"].apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
        g_repr = networks["g_encoder"].apply(g_encoder_params, goal)
        qf_pi = energy_fn(config["energy_fn"], sa_repr, g_repr)

        # Goal critic: binary MLP predicting episode success, annealed in
        commanded_goal = transitions.extras["commanded_goal"]
        goal_logit = networks["goal_critic"].apply(
            target_goal_critic_params,
            jnp.concatenate([state, action, commanded_goal], axis=-1),
        ).squeeze(-1)  # (B,)

        # Linear anneal from 0 to 1; gated by both step count and optional
        # performance threshold (goal_warmup_met_step = 0 when gating disabled)
        effective_warmup = jnp.maximum(
            config["goal_critic_warmup_steps"], goal_warmup_met_step
        )
        anneal_duration = config["goal_critic_anneal_end_steps"] - config["goal_critic_warmup_steps"]
        goal_weight = jnp.clip(
            (env_steps - effective_warmup) / jnp.maximum(anneal_duration, 1.0),
            0.0, 1.0,
        )

        # Normalize goal logit using running EMA stats (stop_gradient prevents
        # normalization from affecting the gradient computation)
        goal_advantage = (
            (goal_logit - jax.lax.stop_gradient(goal_logit_mean_ema))
            / jax.lax.stop_gradient(jnp.sqrt(goal_logit_var_ema) + 1e-6)
        )
        clamp = config["goal_logit_clamp"]
        clamp_min = jnp.where(config["goal_logit_clamp_min"] > -1e8, config["goal_logit_clamp_min"], -clamp)
        goal_contribution = goal_weight * config["goal_critic_coeff"] * jnp.clip(goal_advantage, clamp_min, clamp)

        actor_loss = jnp.mean(
            jnp.exp(log_alpha) * log_prob - qf_pi - goal_contribution
        )

        batch_mean = jnp.mean(goal_logit)
        batch_var = jnp.var(goal_logit)
        return actor_loss, (log_prob, goal_weight, jnp.mean(goal_logit),
                            jnp.mean(qf_pi), batch_mean, batch_var,
                            jnp.mean(goal_advantage))

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - config["target_entropy"]))
        return jnp.mean(alpha_loss)

    (actor_loss, (log_prob, goal_weight, mean_goal_logit, mean_crl_energy,
                  batch_mean, batch_var, mean_goal_advantage)), actor_grad = (
        jax.value_and_grad(actor_loss, has_aux=True)(
            training_state.actor_state.params,
            training_state.target_critic_params,
            training_state.target_goal_critic_params,
            training_state.alpha_state.params["log_alpha"],
            transitions,
            key,
            training_state.env_steps,
            training_state.goal_logit_mean_ema,
            training_state.goal_logit_var_ema,
            training_state.goal_warmup_met_step,
        )
    )
    new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
    new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

    # Update running EMA of goal logit mean/var
    decay = config["goal_norm_decay"]
    new_mean_ema = decay * training_state.goal_logit_mean_ema + (1 - decay) * batch_mean
    new_var_ema = decay * training_state.goal_logit_var_ema + (1 - decay) * batch_var

    training_state = training_state.replace(
        actor_state=new_actor_state,
        alpha_state=new_alpha_state,
        goal_logit_mean_ema=new_mean_ema,
        goal_logit_var_ema=new_var_ema,
    )

    metrics = {
        "entropy": -log_prob,
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
        "goal_critic_active": goal_weight,
        "goal_logit": mean_goal_logit,
        "crl_energy": mean_crl_energy,
        "goal_advantage": mean_goal_advantage,
        "goal_logit_mean_ema": new_mean_ema,
        "goal_logit_var_ema": new_var_ema,
    }

    return training_state, metrics


def update_critic(config, networks, transitions, training_state, key):
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

        logits = energy_fn(config["energy_fn"], sa_repr[:, None, :], g_repr[None, :, :])
        critic_loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)

        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        critic_loss += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp**2)

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
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
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp.mean(),
        "critic_loss": loss,
    }

    return training_state, metrics


def update_goal_critic(config, networks, transitions, training_state, key):
    """Update the goal-conditioned critic.

    Trains on binary labels: did the episode ever reach the commanded goal?
    Positive examples are weighted by min(1/success_rate, cap) to handle
    class imbalance. Gated by max(goal_critic_warmup_steps, goal_warmup_met_step)
    so training waits for both the step-based warmup and any perf-metric gate.
    """
    def goal_critic_loss(goal_critic_params, transitions, key):
        state = transitions.observation[:, : config["state_size"]]
        action = transitions.action
        commanded_goal = transitions.extras["commanded_goal"]
        episode_success = transitions.extras["episode_success"]  # (B,) float 0 or 1

        # Single MLP: (state, action, goal) → logit
        logits = networks["goal_critic"].apply(
            goal_critic_params,
            jnp.concatenate([state, action, commanded_goal], axis=-1),
        ).squeeze(-1)  # (B,)

        # Mask out samples where agent is already near the goal
        agent_pos = state[:, config["goal_indices"]]
        dist_to_goal = jnp.sqrt(jnp.sum((agent_pos - commanded_goal) ** 2, axis=-1) + 1e-6)
        near_goal_mask = dist_to_goal < 1.0  # exclude states within 1.0 of goal

        success_rate = jnp.clip(jnp.mean(episode_success), 1e-4, 1.0)
        pos_weight = jnp.minimum((1.0 - success_rate) / success_rate, config["goal_positive_weight_cap"])

        # Weighted BCE: w_pos for positives, 1.0 for negatives, 0 for near-goal
        sample_weights = jnp.where(episode_success > 0.5, pos_weight, 1.0)
        sample_weights = jnp.where(near_goal_mask, 0.0, sample_weights)
        bce = -(
            episode_success * jax.nn.log_sigmoid(logits)
            + (1.0 - episode_success) * jax.nn.log_sigmoid(-logits)
        )
        loss = jnp.mean(sample_weights * bce)

        # L2 regularization on logits to discourage extreme confidence
        logit_reg = config["goal_logit_reg"] * jnp.mean(logits ** 2)
        loss = loss + logit_reg

        # Metrics
        preds = (logits > 0).astype(jnp.float32)
        accuracy = jnp.mean(preds == episode_success)

        # Diagnostics: break out predictions by class
        n_pos = jnp.sum(episode_success) + 1e-8
        n_neg = jnp.sum(1.0 - episode_success) + 1e-8
        true_pos_rate = jnp.sum(preds * episode_success) / n_pos
        true_neg_rate = jnp.sum((1.0 - preds) * (1.0 - episode_success)) / n_neg
        mean_logit = jnp.mean(logits)
        mean_logit_pos = jnp.sum(logits * episode_success) / n_pos
        mean_logit_neg = jnp.sum(logits * (1.0 - episode_success)) / n_neg
        pred_positive_rate = jnp.mean(preds)

        return loss, (accuracy, success_rate, pos_weight,
                      true_pos_rate, true_neg_rate, mean_logit,
                      mean_logit_pos, mean_logit_neg, pred_positive_rate)

    (loss, (accuracy, success_rate, pos_weight,
            true_pos_rate, true_neg_rate, mean_logit,
            mean_logit_pos, mean_logit_neg, pred_positive_rate)), grad = jax.value_and_grad(
        goal_critic_loss, has_aux=True
    )(training_state.goal_critic_state.params, transitions, key)

    # Only apply gradients after both step warmup AND (optional) perf-metric gate.
    # goal_warmup_met_step = 0 when gating disabled, 1e12 until triggered.
    effective_warmup = jnp.maximum(
        config["goal_critic_warmup_steps"], training_state.goal_warmup_met_step
    )
    active = training_state.env_steps >= effective_warmup
    grad = jax.tree_util.tree_map(lambda g: g * active, grad)
    new_goal_critic_state = training_state.goal_critic_state.apply_gradients(grads=grad)

    # EMA update of target goal critic params
    tau = config["ema_tau"]
    new_target_goal_critic_params = jax.tree_util.tree_map(
        lambda target, online: tau * online + (1.0 - tau) * target,
        training_state.target_goal_critic_params,
        new_goal_critic_state.params,
    )

    training_state = training_state.replace(
        goal_critic_state=new_goal_critic_state,
        target_goal_critic_params=new_target_goal_critic_params,
    )

    metrics = {
        "goal_critic_loss": loss,
        "goal_critic_accuracy": accuracy,
        "goal_success_rate": success_rate,
        "goal_pos_weight": pos_weight,
        "goal_true_pos_rate": true_pos_rate,
        "goal_true_neg_rate": true_neg_rate,
        "goal_mean_logit": mean_logit,
        "goal_mean_logit_pos": mean_logit_pos,
        "goal_mean_logit_neg": mean_logit_neg,
        "goal_pred_positive_rate": pred_positive_rate,
    }

    return training_state, metrics
