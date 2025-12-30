from typing import Callable, Optional
import flax.linen as nn
import jax.numpy as jnp
import jax


class BroNet(nn.Module):
  embed_dim: int
  num_blocks: int
  activation: Callable = nn.relu
  dropout_rate: Optional[float] = None

  kernel_init: nn.initializers.Initializer = nn.initializers.xavier_normal()
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x, train: bool = True):
    x = nn.Dense(
        self.embed_dim, dtype=self.dtype, kernel_init=self.kernel_init
    )(x)
    if self.dropout_rate is not None and self.dropout_rate > 0.0:
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
    x = nn.LayerNorm(dtype=self.dtype)(x)
    x = self.activation(x)

    for _ in range(self.num_blocks):
      skip = x
      x = nn.Dense(
          self.embed_dim, dtype=self.dtype, kernel_init=self.kernel_init
      )(x)
      if self.dropout_rate is not None and self.dropout_rate > 0.0:
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
      x = nn.LayerNorm(dtype=self.dtype)(x)
      x = self.activation(x)
      x = nn.Dense(
          self.embed_dim, dtype=self.dtype, kernel_init=self.kernel_init
      )(x)
      if self.dropout_rate is not None and self.dropout_rate > 0.0:
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
      x = nn.LayerNorm(dtype=self.dtype)(x)
      x = x + skip

    return x
