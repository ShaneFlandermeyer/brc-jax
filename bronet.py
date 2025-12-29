import flax.linen as nn
import jax.numpy as jnp
import jax


class BroNet(nn.Module):
  embed_dim: int
  num_blocks: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.embed_dim)(x)
    x = nn.LayerNorm()(x)
    x = nn.relu(x)

    for _ in range(self.num_blocks):
      skip = x
      x = nn.Dense(self.embed_dim)(x)
      x = nn.LayerNorm()(x)
      x = nn.relu(x)
      x = nn.Dense(self.embed_dim)(x)
      x = nn.LayerNorm()(x)
      x = x + skip  # Residual connection

    return x
