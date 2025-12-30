import jax
import jax.scipy.special
import jax.numpy as jnp


def sg(x): return jax.tree.map(jax.lax.stop_gradient, x)


def categorical_target(
    next_probs: jax.Array,
    rewards: jax.Array,
    discount: jax.Array,
    alpha: jax.Array,
    entropy: jax.Array,
    support: jax.Array,
) -> jax.Array:
  low, high = support[0], support[-1]
  bin_width = support[1] - support[0]
  # Compute the projected Bellman update onto the support
  Tz = (rewards + discount * (support + alpha * entropy)).clip(low, high)
  b = (Tz - low) / bin_width
  l = jnp.floor(b).astype(jnp.int32)
  u = jnp.ceil(b).astype(jnp.int32)
  # Disctribute probabilities
  # (l == u) handles the case when bj is an integer
  d_m_l = (u + (l == u).astype(jnp.float32) - b) * next_probs
  d_m_u = (b - l) * next_probs
  m = jnp.zeros_like(next_probs)
  batch_inds = jnp.arange(m.shape[0])[:, None]
  m = m.at[batch_inds, l].add(d_m_l)
  m = m.at[batch_inds, u].add(d_m_u)
  return m


def cross_entropy(pred_logits: jax.Array, target: jax.Array) -> jax.Array:
  return -jnp.sum(
      jax.nn.log_softmax(pred_logits, axis=-1) * target, axis=-1
  )
