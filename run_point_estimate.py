# python3
# pylint: disable=g-bad-file-header
# Copyright (c) 2021 Vikranth Dwaracherla

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ============================================================================
"""Run file for Langevin DQN (using a single point estimate) on a bsuite experiment."""

from absl import app
from absl import flags

import bsuite
from bsuite import sweep

from bsuite.baselines import experiment
from bsuite.baselines.utils import pool

import sonnet as snt

# Internal imports.
import agent as langevin_dqn

# Experiment flags.
flags.DEFINE_string(
    'bsuite_id', 'deep_sea/0', 'BSuite identifier. '
    'This global flag can be used to control which environment is loaded.')
flags.DEFINE_string('save_path', '/tmp/bsuite', 'where to save bsuite results')
flags.DEFINE_enum('logging_mode', 'csv', ['csv', 'sqlite', 'terminal'],
                  'which form of logging to use for bsuite results')
flags.DEFINE_boolean('overwrite', False, 'overwrite csv logging if found')
flags.DEFINE_integer('num_episodes', None, 'Overrides number of training eps.')

# Network options
flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layers')
flags.DEFINE_boolean('with_bias', True, 'bias in the neural network architecture')
flags.DEFINE_integer('num_units', 50, 'number of units per hidden layer')
flags.DEFINE_float('relu_alpha', 0.1, 'alpha in leaky relu')
flags.DEFINE_float('prior_scale', 0., 'scale for additive prior network')

# Core DQN options
flags.DEFINE_integer('batch_size', 128, 'size of batches sampled from replay')
flags.DEFINE_float('discount', .99, 'discounting on the agent side')
flags.DEFINE_integer('replay_capacity', 100000, 'size of the replay buffer')
flags.DEFINE_integer('min_replay_size', 0, 'min transitions for sampling')
flags.DEFINE_integer('sgds_per_step', 10, 'update steps between each env step')
flags.DEFINE_boolean('update_every_step', False, 'update after every env step')
flags.DEFINE_integer('target_update_period', 4,
                     'steps between target net updates')
flags.DEFINE_float('mask_prob', 1.0, 'probability for bootstrap mask')
flags.DEFINE_float('noise_scale', 0.005, 
                     'variance of Gaussian noise term in the update')
flags.DEFINE_float('learning_rate', 1e-2, 'learning rate for optimizer')
flags.DEFINE_float('regularization_penalty', 0.005, 'regularization penalty')

flags.DEFINE_integer('seed', 42, 'seed for random number generation')
flags.DEFINE_float('epsilon', 0.0, 'fraction of exploratory random actions')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')


FLAGS = flags.FLAGS


def run(bsuite_id: str) -> str:
  """Runs a BDQN agent on a given bsuite environment, logging to CSV."""

  env = bsuite.load_and_record(
      bsuite_id=bsuite_id,
      save_path=FLAGS.save_path,
      logging_mode=FLAGS.logging_mode,
      overwrite=FLAGS.overwrite,
  )

  noise_scale = FLAGS.noise_scale
  regularization_penalty = FLAGS.regularization_penalty

  ensemble = langevin_dqn.make_ensemble(
      num_actions=env.action_spec().num_values,
      num_ensemble=1,
      num_hidden_layers=FLAGS.num_hidden_layers,
      num_units=FLAGS.num_units,
      prior_scale=FLAGS.prior_scale,
      relu_alpha=FLAGS.relu_alpha,
      with_bias=FLAGS.with_bias)

  agent = langevin_dqn.EnsembleLangevinDqn(
      obs_spec=env.observation_spec(),
      action_spec=env.action_spec(),
      ensemble=ensemble,
      num_ensemble=1,
      batch_size=FLAGS.batch_size,
      discount=FLAGS.discount,
      replay_capacity=FLAGS.replay_capacity,
      min_replay_size=FLAGS.min_replay_size,
      sgds_per_step=FLAGS.sgds_per_step,
      update_every_step=FLAGS.update_every_step,
      target_update_period=FLAGS.target_update_period,
      optimizer=snt.optimizers.Adam(learning_rate=FLAGS.learning_rate,
                                    epsilon=1e-6),
      mask_prob=FLAGS.mask_prob,
      noise_scale=noise_scale,
      epsilon_fn=lambda x: FLAGS.epsilon,
      regularization_penalty=regularization_penalty,
      seed=FLAGS.seed)

  num_episodes = FLAGS.num_episodes or getattr(env, 'bsuite_num_episodes')
  experiment.run(
      agent=agent,
      environment=env,
      num_episodes=num_episodes,
      verbose=FLAGS.verbose)

  return bsuite_id


def main(argv):
  # Parses whether to run a single bsuite_id, or multiprocess sweep.
  del argv  # Unused.
  bsuite_id = FLAGS.bsuite_id

  print(bsuite_id)

  if bsuite_id in sweep.SWEEP:
    print(f'Running single experiment: bsuite_id={bsuite_id}.')
    run(bsuite_id)

  elif hasattr(sweep, bsuite_id):
    bsuite_sweep = getattr(sweep, bsuite_id)
    print(f'Running sweep over bsuite_id in sweep.{bsuite_sweep}')
    FLAGS.verbose = True
    pool.map_mpi(run, bsuite_sweep)

  else:
    raise ValueError(f'Invalid flag: bsuite_id={bsuite_id}.')


if __name__ == '__main__':
  app.run(main)
