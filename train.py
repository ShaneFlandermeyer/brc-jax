import os
from collections import defaultdict
from functools import partial

import gymnasium as gym
import hydra
import jax
import numpy as np
import orbax.checkpoint as ocp
# Tensorboard: Prevent tf from allocating full GPU memory
import tensorflow as tf
import tqdm
from flax.metrics import tensorboard

from brc_jax.brc import BRC
from brc_jax.buffer import ReplayBuffer
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


@hydra.main(config_name='config', config_path='.', version_base=None)
def train(cfg: dict):
  ##############################
  # Logger setup
  ##############################
  output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
  writer = tensorboard.SummaryWriter(os.path.join(output_dir, 'tensorboard'))
  writer.hparams(cfg)

  ##############################
  # Environment setup
  ##############################
  def make_env(config, seed):
    def make_gym_env(env_id, seed):
      env = gym.make(env_id)
      env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
      env = gym.wrappers.RecordEpisodeStatistics(env)
      env = gym.wrappers.Autoreset(env)
      env.action_space.seed(seed)
      env.observation_space.seed(seed)
      return env

    if config.backend == "gymnasium":
      return make_gym_env(config.env_id, seed)
    elif config.backend == "dmc":
      env = make_dmc_env(config.env_id, seed, config.dmc.obs_type)
      env = gym.wrappers.RecordEpisodeStatistics(env)
      env = gym.wrappers.Autoreset(env)
      env.action_space.seed(seed)
      env.observation_space.seed(seed)
      return env
    else:
      raise ValueError("Environment not supported:", config.env_id)

  if cfg.env.asynchronous:
    vector_env_cls = gym.vector.AsyncVectorEnv
  else:
    vector_env_cls = gym.vector.SyncVectorEnv
  env = vector_env_cls(
      [
          partial(make_env, cfg.env, seed)
          for seed in range(cfg.seed, cfg.seed+cfg.env.num_envs)
      ]
  )
  np.random.seed(cfg.seed)
  rng = jax.random.PRNGKey(cfg.seed)

  ##############################
  # Replay buffer setup
  ##############################
  dummy_obs, _ = env.reset()
  dummy_action = env.action_space.sample()
  dummy_next_obs, dummy_reward, dummy_term, dummy_trunc, _ = env.step(
      dummy_action
  )
  replay_buffer = ReplayBuffer(
      capacity=cfg.buffer_size,
      num_envs=cfg.env.num_envs,
      seed=cfg.seed,
      dummy_input=dict(
          observation=dummy_obs,
          action=dummy_action,
          reward=dummy_reward,
          next_observation=dummy_next_obs,
          terminated=dummy_term,
          truncated=dummy_trunc
      )
  )

  # Create agent
  rng, model_key = jax.random.split(rng, 2)
  agent = BRC.create(
      action_dim=np.prod(env.single_action_space.shape),
      state_dim=env.single_observation_space.shape,
      **cfg.brc,
      target_entropy=-0.5 * np.prod(env.single_action_space.shape),
      key=model_key,
  )

  global_step = 0
  options = ocp.CheckpointManagerOptions(
      max_to_keep=1, save_interval_steps=cfg.log.save_interval_steps
  )
  checkpoint_path = os.path.join(output_dir, 'checkpoint')
  with ocp.CheckpointManager(
      checkpoint_path,
      options=options,
      item_names=('agent', 'global_step', 'buffer_state')
  ) as mngr:
    if mngr.latest_step() is not None:
      print('Checkpoint folder found, restoring from', mngr.latest_step())
      abstract_buffer_state = jax.tree.map(
          ocp.utils.to_shape_dtype_struct, replay_buffer.get_state()
      )
      restored = mngr.restore(
          mngr.latest_step(),
          args=ocp.args.Composite(
              agent=ocp.args.StandardRestore(agent),
              global_step=ocp.args.JsonRestore(),
              buffer_state=ocp.args.StandardRestore(abstract_buffer_state),
          )
      )
      agent, global_step = restored.agent, restored.global_step
      replay_buffer.restore(restored.buffer_state)
    else:
      print('No checkpoint folder found, starting from scratch')
      mngr.save(
          global_step,
          args=ocp.args.Composite(
              agent=ocp.args.StandardSave(agent),
              global_step=ocp.args.JsonSave(global_step),
              buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
          ),
      )
      mngr.wait_until_finished()

    ##############################
    # Training loop
    ##############################

    ep_count = np.zeros(cfg.env.num_envs, dtype=int)
    prev_logged_step = global_step
    pbar = tqdm.tqdm(initial=global_step, total=cfg.max_steps)
    done = np.zeros(cfg.env.num_envs, dtype=bool)
    observation, _ = env.reset(seed=cfg.seed)
    for global_step in range(global_step, cfg.max_steps, cfg.env.num_envs):
      if global_step <= cfg.seed_steps:
        action = env.action_space.sample()
      else:
        rng, action_key = jax.random.split(rng)
        action = agent.act(
            observation,
            deterministic=False,
            key=action_key,
        )
        action = np.array(action)

      next_observation, reward, terminated, truncated, info = env.step(action)

      if np.any(~done):
        replay_buffer.insert(
            dict(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated
            ),
            mask=~done
        )
      observation = next_observation

      # Handle terminations/truncations
      done = np.logical_or(terminated, truncated)
      if np.any(done):
        for ienv in range(cfg.env.num_envs):
          if done[ienv]:
            r = info['episode']['r'][ienv]
            l = info['episode']['l'][ienv]
            print(
                f"Episode {ep_count[ienv]}: r = {r:.2f}, l = {l}"
            )
            writer.scalar(f'episode/return', r, global_step + ienv)
            writer.scalar(f'episode/length', l, global_step + ienv)
            ep_count[ienv] += 1

      if global_step >= cfg.seed_steps:
        if global_step == cfg.seed_steps:
          print('Pre-training on seed data...')
          num_updates = cfg.seed_steps
        else:
          num_updates = max(1, int(cfg.env.num_envs * cfg.utd_ratio))

        rng, *update_keys = jax.random.split(rng, num_updates+1)
        log_this_step = global_step >= prev_logged_step + \
            cfg.log.log_interval_steps
        if log_this_step:
          all_train_info = defaultdict(list)
          prev_logged_step = global_step

        for iupdate in range(num_updates):
          batch = replay_buffer.sample(agent.batch_size)
          agent, train_info = agent.update(
              observations=batch['observation'],
              actions=batch['action'],
              rewards=batch['reward'],
              next_observations=batch['next_observation'],
              terminated=batch['terminated'],
              truncated=batch['truncated'],
              key=update_keys[iupdate]
          )

          if log_this_step:
            for k, v in train_info.items():
              all_train_info[k].append(np.array(v))

        if log_this_step:
          for k, v in all_train_info.items():
            writer.scalar(f'train/{k}_mean', np.mean(v), global_step)
            writer.scalar(f'train/{k}_std', np.std(v), global_step)

        mngr.save(
            global_step,
            args=ocp.args.Composite(
                agent=ocp.args.StandardSave(agent),
                global_step=ocp.args.JsonSave(global_step),
                buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
            ),
        )

      pbar.update(cfg.env.num_envs)
    pbar.close()


if __name__ == '__main__':
  train()
