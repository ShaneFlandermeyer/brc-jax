from typing import Tuple
from flax import linen as nn
import jax.numpy as jnp
import jax
import flax

class Temperature(nn.Module):
  initial_temperature: float = 1.0

  @nn.compact
  def __call__(self) -> jax.Array:
    log_temp = self.param(
        'log_temp',
        init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature))
    )
    return jnp.exp(log_temp)


def update_temperature(
    model: Temperature,
    entropy: float,
    target_entropy: float
) -> Tuple[Temperature, dict]:
  def temperature_loss_fn(
      params: flax.core.FrozenDict
  ) -> Tuple[jax.Array, dict]:
    alpha = model.apply_fn({'params': params})
    temperature_loss = alpha * (entropy - target_entropy).mean()
    info = dict(
        temperature=alpha,
        temperature_loss=temperature_loss,
    )
    return temperature_loss, info
  grads, info = jax.grad(temperature_loss_fn, has_aux=True)(
      model.params
  )
  return model.apply_gradients(grads=grads), info
