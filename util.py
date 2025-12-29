import jax
import jax.scipy.special
import jax.numpy as jnp


def hl_gauss(
    x: jax.Array,
    low: float,
    high: float,
    num_bins: int,
    sigma: float = 0.75,
) -> jax.Array:

  x = jnp.clip(x, low, high)
  support = jnp.linspace(low, high, num_bins + 1, dtype=jnp.float32)
  bin_width = support[1] - support[0]
  cdf_evals = jax.scipy.special.erf(
      (support - x[..., None]) / (jnp.sqrt(2) * sigma * bin_width)
  )
  z = cdf_evals[..., -1] - cdf_evals[..., 0]
  bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]

  return bin_probs / z[..., None]


def hl_gauss_inv(
    probs: jax.Array,
    min_value: float,
    max_value: float,
    num_bins: int,
) -> jax.Array:
  support = jnp.linspace(min_value, max_value, num_bins + 1, dtype=jnp.float32)
  centers = (support[:-1] + support[1:]) / 2
  return jnp.sum(probs * centers, axis=-1)  # Sum over bins


def sg(x): return jax.tree.map(jax.lax.stop_gradient, x)


def cross_entropy(
    pred_logits: jax.Array,
    target: jax.Array,
    low: float,
    high: float,
    num_bins: int
) -> jax.Array:
  return -jnp.sum(
      jax.nn.log_softmax(pred_logits, axis=-1) * target, axis=-1
  )
