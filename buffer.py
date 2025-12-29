import gymnasium as gym
import numpy as np
from typing import *
import jax
from collections import deque
from jaxtyping import PyTree


class ReplayBuffer():

  def __init__(self,
               capacity: int,
               dummy_input: Dict,
               num_envs: int = 1,
               vectorized: bool = False,
               seed: Optional[int] = None,
               ):

    self.vectorized = vectorized
    self.num_envs = num_envs
    self.capacity = capacity // num_envs
    self.size = np.zeros(num_envs, dtype=int)
    self.current_ind = np.zeros(num_envs, dtype=int)

    self.data = jax.tree.map(
        lambda x: np.zeros(
            (self.capacity,) + np.asarray(x).shape, np.asarray(x).dtype
        ), dummy_input
    )
    self.np_random = np.random.default_rng(seed=seed)

  def insert(self,
             data: PyTree,
             mask: Optional[np.ndarray] = None
             ) -> None:
    """
    Insert data into the buffer

    Parameters
    ----------
    data : PyTree
        Data to insert
    mask : Optional[np.ndarray], optional
        A boolean mask of size self.num_envs, which specifies which env buffers receive new data. If None, all envs receive data, by default None
    """
    # Insert data for the specified envs
    if mask is None:
      mask = np.ones(self.num_envs, dtype=bool)

    if self.vectorized:
      def masked_set(x, y):
        x[self.current_ind, mask] = y[mask]
      jax.tree.map(masked_set, self.data, data)
    else:
      jax.tree.map(
          lambda x, y: x.__setitem__(self.current_ind, y), self.data, data
      )

    # Update buffer state
    self.current_ind[mask] = (self.current_ind[mask] + 1) % self.capacity
    self.size[mask] = np.clip(self.size[mask] + 1, 0, self.capacity)

  def sample(
      self,
      batch_size: int,
  ) -> Union[PyTree, Tuple[PyTree, Tuple[np.ndarray]]]:
    """
    Sample a batch of sequences from the buffer.

    Sequences are drawn uniformly from each environment buffer, and they may cross episode boundaries.

    Parameters
    ----------
    batch_size : int
    sequence_length : int
    return_inds : bool
        If True, also returns

    Returns
    -------
    Union[PyTree, Tuple[PyTree, Tuple[np.ndarray]]]
        The sampled batch. If return_inds is True, also returns the sampled indices in the batch/time dimensions
    """

    if self.vectorized:
      batch = self._sample_vectorized(batch_size)
    else:
      batch = self._sample(batch_size)

    return batch

  def _sample(self, batch_size: int) -> PyTree:
    # Sample envs and start indices
    inds = self.np_random.integers(
        low=0, high=self.size,
        size=batch_size,
        endpoint=True,
    )

    batch = jax.tree.map(lambda x: x[inds], self.data)

    return batch

  def _sample_vectorized(self, batch_size: int) -> PyTree:
    # Sample envs and start indices
    env_inds = self.np_random.integers(
        low=0, high=self.num_envs,
        size=batch_size
    )
    inds = self.np_random.integers(
        low=0, high=self.size[env_inds],
        size=batch_size,
        endpoint=True,
    )

    batch = jax.tree.map(
        lambda x: x[inds[:, None], env_inds[:, None]],
        self.data
    )

    return batch

  def get_state(self) -> Dict:
    return {
        'current_ind': self.current_ind,
        'size': self.size,
        'data': self.data,
    }

  def restore(self, state: Dict) -> None:
    self.current_ind = state['current_ind']
    self.size = state['size']
    self.data = state['data']
