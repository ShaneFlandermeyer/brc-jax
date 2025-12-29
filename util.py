import jax
import jax.scipy.special
import jax.numpy as jnp


def sg(x): return jax.tree.map(jax.lax.stop_gradient, x)


def categorical_target(
    next_probs: jax.Array,
    rewards: jax.Array,
    terminated: jax.Array,
    discount: float,
    low: float,
    high: float,
    num_bins: int
) -> jax.Array:
  support = jnp.linspace(low, high, num_bins, dtype=jnp.float32)
  bin_width = support[1] - support[0]
  # Compute the projected Bellman update onto the support
  Tz = (
      rewards[..., None] + (1 - terminated[..., None]) * discount * support
  ).clip(low, high)
  b = (Tz - low) / bin_width
  l = jnp.floor(b).astype(jnp.int32)
  u = jnp.ceil(b).astype(jnp.int32)
  # Disctribute probabilities
  m = jnp.zeros_like(next_probs)
  m = m.at[..., l].add(next_probs * (u - b))
  m = m.at[..., u].add(next_probs * (b - l))
  return m


def cross_entropy(pred_logits: jax.Array, target: jax.Array) -> jax.Array:
  return -jnp.sum(
      jax.nn.log_softmax(pred_logits, axis=-1) * target, axis=-1
  )
