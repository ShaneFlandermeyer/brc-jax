from __future__ import annotations

import copy
from functools import partial
from typing import Any, Dict, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import struct
from flax.training.train_state import TrainState
from jaxtyping import PRNGKeyArray, PyTree
from bronet import BroNet
import temperature
from util import categorical_target, cross_entropy, mish, sg
import jax

MIN_LOG_STD = -10
MAX_LOG_STD = 2


class BRC(struct.PyTreeNode):
  # Model components
  policy_model: TrainState
  value_model: TrainState
  target_value_model: TrainState
  temperature_model: TrainState
  # Optimization
  batch_size: int = struct.field(pytree_node=False)
  discount: float
  tau: float
  # Value
  num_value_nets: int = struct.field(pytree_node=False)
  num_value_bins: int = struct.field(pytree_node=False)
  support: jax.Array
  value_scale: jax.Array
  # Policy
  target_entropy: float

  @classmethod
  def create(
      cls,
      # TODO: Better handling for state and action dims
      action_dim: int,
      state_dim: int,
      # Optimization params
      batch_size: int,
      discount: float,
      learning_rate: float,
      tau: float,
      # Policy params
      policy_dim: int,
      num_policy_blocks: int,
      # Value params
      value_dim: int,
      num_value_blocks: int,
      num_value_nets: int,
      value_dropout: float,
      min_value: float,
      max_value: float,
      num_value_bins: int,
      # Temperature params
      init_temperature: float,
      target_entropy: float,
      dtype: jnp.dtype = jnp.float32,
      *,
      key: PRNGKeyArray,
  ) -> BRC:
    policy_key, value_key = jax.random.split(key, 2)

    # Policy model
    policy_module = nn.Sequential([
        BroNet(
            embed_dim=policy_dim,
            num_blocks=num_policy_blocks,
            activation=mish,
            kernel_init=nn.initializers.truncated_normal(0.02),
            dtype=dtype,
        ),
        nn.Dense(
            2*action_dim,
            kernel_init=nn.initializers.truncated_normal(0.02),
            dtype=dtype
        )
    ])

    policy_model = TrainState.create(
        apply_fn=policy_module.apply,
        params=policy_module.init(policy_key, jnp.zeros(state_dim))['params'],
        tx=optax.chain(
            optax.zero_nans(),
            optax.adamw(learning_rate),
        )
    )

    # Value model
    value_param_key, value_dropout_key = jax.random.split(value_key)
    value_base = partial(nn.Sequential, [
        BroNet(
            embed_dim=value_dim,
            num_blocks=num_value_blocks,
            activation=mish,
            dropout_rate=value_dropout,
            kernel_init=nn.initializers.truncated_normal(0.02),
            dtype=dtype,
        ),
        nn.Dense(
            num_value_bins, kernel_init=jax.nn.initializers.zeros, dtype=dtype
        )
    ])
    value_ensemble = nn.vmap(
        value_base,
        variable_axes={'params': 0},
        split_rngs={
            'params': True,
            'dropout': True
        },
        in_axes=None,
        out_axes=0,
        axis_size=num_value_nets
    )()
    value_model = TrainState.create(
        apply_fn=value_ensemble.apply,
        params=value_ensemble.init(
            {'params': value_param_key, 'dropout': value_dropout_key},
            jnp.zeros(state_dim + action_dim))['params'],
        tx=optax.chain(
            optax.zero_nans(),
            optax.adamw(learning_rate),
        )
    )
    target_value_model = TrainState.create(
        apply_fn=value_ensemble.apply,
        params=copy.deepcopy(value_model.params),
        tx=optax.GradientTransformation(lambda _: None, lambda _: None)
    )

    # Temperature
    temperature_module = temperature.Temperature(
        initial_temperature=init_temperature
    )
    temperature_model = TrainState.create(
        apply_fn=temperature_module.apply,
        params=temperature_module.init(jax.random.PRNGKey(0))['params'],
        tx=optax.chain(
            optax.zero_nans(),
            optax.adamw(learning_rate),
        )
    )

    # Tabulate
    print("Policy model")
    print("------------")
    print(
        policy_module.tabulate(
            jax.random.PRNGKey(0),
            jnp.zeros(state_dim),
        )
    )
    print("Value model")
    print("-----------")
    print(
        value_ensemble.tabulate(
            jax.random.PRNGKey(0),
            jnp.zeros(state_dim + action_dim),
        )
    )
    print("Temperature model")
    print("-----------------")
    print(
        temperature_module.tabulate(jax.random.PRNGKey(0))
    )

    return cls(
        policy_model=policy_model,
        value_model=value_model,
        target_value_model=target_value_model,
        temperature_model=temperature_model,
        batch_size=batch_size,
        discount=discount,
        tau=tau,
        num_value_nets=num_value_nets,
        num_value_bins=num_value_bins,
        support=jnp.linspace(min_value, max_value, num_value_bins),
        target_entropy=target_entropy,
        value_scale=jnp.array([1.0]),
    )

  @partial(jax.jit, static_argnames=['deterministic'])
  def act(
      self,
      obs: jax.Array,
      deterministic: bool = False,
      *,
      key: PRNGKeyArray,
  ) -> jax.Array:
    action, _, _, _ = self.sample_actions(
        obs=obs,
        params=self.policy_model.params,
        deterministic=deterministic,
        key=key,
    )
    return action

  @jax.jit
  def update(
      self,
      observations: jax.Array,
      actions: jax.Array,
      rewards: jax.Array,
      next_observations: jax.Array,
      terminated: jax.Array,
      truncated: jax.Array,
      *,
      key: PRNGKeyArray,
  ) -> Tuple[BRC, Dict[str, Any]]:
    value_loss_key, policy_loss_key = jax.random.split(key, 2)

    # Update value function
    def value_loss_fn(
        value_params: flax.core.FrozenDict
    ) -> Tuple[jax.Array, Dict[str, Any]]:
      next_action_key, value_key, value_target_key = jax.random.split(
          value_loss_key, 3
      )
      _, value_logits = self.Q(
          obs=observations,
          action=actions,
          train=True,
          params=value_params,
          key=value_key,
      )

      # Value target distribution
      alpha = self.temperature_model.apply_fn(
          {'params': self.temperature_model.params}
      )
      next_action, _, _, log_probs = self.sample_actions(
          obs=next_observations,
          params=self.policy_model.params,
          deterministic=False,
          key=next_action_key,
      )
      next_value_probs, _ = self.Q(
          obs=next_observations,
          action=next_action,
          train=True,
          params=self.target_value_model.params,
          key=value_target_key,
      )

      # Update value/reward scale
      # Unlike in the paper, values are normalized by the EMA of the max target Q value in each batch.
      next_Qs = self.value_scale / self.support[-1] * jnp.sum(
          next_value_probs * self.support, axis=-1
      )
      target_Qs = rewards + (1 - terminated) * self.discount * next_Qs
      new_scale = jnp.max(abs(target_Qs))
      value_scale = (
          self.tau * new_scale + (1 - self.tau) * self.value_scale
      ).clip(1, None)

      target_probs = categorical_target(
          next_probs=next_value_probs.mean(axis=0),
          rewards=rewards[..., None] / value_scale,
          discount=(1 - terminated[..., None]) * self.discount,
          alpha=alpha,
          entropy=-log_probs[..., None],
          support=self.support,
      )

      value_loss = cross_entropy(
          pred_logits=value_logits, target=sg(target_probs)
      ).mean()
      value_info = dict(
          value_loss=value_loss,
          value_scale=value_scale,
      )
      return value_loss, value_info

    value_grads, value_info = jax.grad(value_loss_fn, has_aux=True)(
        self.value_model.params
    )
    new_value_model = self.value_model.apply_gradients(grads=value_grads)
    new_target_value_model = self.target_value_model.replace(
        params=optax.incremental_update(
            new_value_model.params,
            self.target_value_model.params,
            step_size=self.tau,
        )
    )

    # Update policy
    def policy_loss_fn(
        policy_params: flax.core.FrozenDict
    ) -> Tuple[jax.Array, Dict[str, Any]]:
      action_key, value_key = jax.random.split(policy_loss_key, 2)
      actions, _, log_std, log_probs = self.sample_actions(
          obs=observations,
          params=policy_params,
          deterministic=False,
          key=action_key,
      )

      # Compute Q values
      probs, _ = self.Q(
          obs=observations,
          action=actions,
          train=True,
          params=self.value_model.params,
          key=value_key,
      )
      Q = jnp.sum(probs * self.support, axis=-1).mean(axis=0)

      alpha = self.temperature_model.apply_fn(
          {'params': self.temperature_model.params}
      )
      policy_loss = (alpha * log_probs - Q).mean()
      policy_info = dict(
          policy_loss=policy_loss,
          policy_entropy=-log_probs,
          policy_std=jnp.exp(log_std),
      )
      return policy_loss, policy_info
    policy_grads, policy_info = jax.grad(policy_loss_fn, has_aux=True)(
        self.policy_model.params
    )
    new_policy = self.policy_model.apply_gradients(grads=policy_grads)

    # Update temperature
    new_temperature, temperature_info = temperature.update_temperature(
        model=self.temperature_model,
        entropy=policy_info['policy_entropy'],
        target_entropy=self.target_entropy
    )

    new_agent = self.replace(
        policy_model=new_policy,
        value_model=new_value_model,
        target_value_model=new_target_value_model,
        temperature_model=new_temperature,
        value_scale=value_info['value_scale'],
    )
    info = {**value_info, **policy_info, **temperature_info}

    return new_agent, info

  def sample_actions(
      self,
      obs: jax.Array,
      params: Dict[str, Any],
      deterministic: bool = False,
      *,
      key: PRNGKeyArray,
  ):
    # Chunk the policy model output to get mean and logstd
    mean, log_std = jnp.split(
        self.policy_model.apply_fn(
            {'params': params}, obs
        ).astype(jnp.float32), 2, axis=-1
    )
    log_std = MIN_LOG_STD + 0.5 * \
        (MAX_LOG_STD - MIN_LOG_STD) * (jnp.tanh(log_std) + 1)

    action_dist = tfd.MultivariateNormalDiag(
        loc=mean, scale_diag=jnp.exp(log_std)
    )
    if deterministic:
      action = mean
    else:
      action = action_dist.sample(seed=key)
    log_probs = action_dist.log_prob(action)

    # Squash tanh
    log_probs -= jnp.sum(
        (2 * (jnp.log(2) - action - jax.nn.softplus(-2 * action))), axis=-1
    )
    mean = jnp.tanh(mean)
    action = jnp.tanh(action)
    return action, mean, log_std, log_probs

  def Q(
      self,
      obs: jax.Array,
      action: jax.Array,
      train: bool,
      params: Dict[str, Any],
      key: PRNGKeyArray,
  ) -> jax.Array:
    z = jnp.concatenate([obs, action], axis=-1)
    logits = self.value_model.apply_fn(
        {'params': params}, z, train, rngs={'dropout': key}
    ).astype(jnp.float32)

    probs = jax.nn.softmax(logits, axis=-1)
    return probs, logits
