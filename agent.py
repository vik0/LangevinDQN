# python3
# pylint: disable=g-bad-file-header
# MIT License

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
"""A simple implementation of Langevin DQN with ensembles.

- This agent is implemented with Bsuite baselines code 
  (https://github.com/deepmind/bsuite/tree/master/bsuite/baselines). You need 
  to install bsuite to be able to run this code.
"""

import copy
from typing import Callable, NamedTuple, Sequence

import networks
from bsuite.baselines import base
from bsuite.baselines.utils import replay

import dm_env
from dm_env import specs
import numpy as np
import sonnet as snt
import tensorflow as tf
import tree


class EnsembleLangevinDqn(base.Agent):
  """Ensemble Langevin DQN with additive prior functions."""

  def __init__(
      self,
      obs_spec: specs.Array,
      action_spec: specs.DiscreteArray,
      ensemble: snt.Module,
      num_ensemble: int,
      batch_size: int,
      discount: float,
      replay_capacity: int,
      min_replay_size: int,
      sgds_per_step: int,
      target_update_period: int,
      optimizer: snt.Optimizer,
      mask_prob: float,
      noise_scale: float,
      epsilon_fn: Callable[[int], float] = lambda _: 0.,
      update_every_step: bool = True,
      regularization_penalty: float = 0.001,
      seed: int = None,
  ):
    """Langevin Ensemble DQN with additive prior functions."""
    # Agent components.
    tf.random.set_seed(seed)

    self._ensemble = ensemble
    self._forward = tf.function(ensemble)
    self._target_ensemble = copy.deepcopy(ensemble)
    self._num_ensemble = num_ensemble
    self._optimizer = optimizer
    self._replay = replay.Replay(capacity=replay_capacity)

    # Create variables for the ensemble
    snt.build(ensemble, (None, *obs_spec.shape))

    # Agent hyperparameters.
    self._num_actions = action_spec.num_values
    self._batch_size = batch_size
    self._sgds_per_step = sgds_per_step
    self._update_every_step = update_every_step
    self._target_update_period = target_update_period
    self._min_replay_size = min_replay_size
    self._epsilon_fn = epsilon_fn
    self._mask_prob = mask_prob
    self._noise_scale = noise_scale
    self._rng = np.random.RandomState(seed)
    self._discount = discount

    # Agent state.
    self._total_steps = tf.Variable(1)
    self._total_update_steps = 0
    self._active_head = 0

    self._regularization_penalty = regularization_penalty

  @tf.function
  def _langevin_step(self, transitions: Sequence[tf.Tensor],
                     replay_size: tf.Tensor):
    """Does a step of SGD for the whole ensemble over `transitions`."""
    o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t = transitions
    variables = tree.flatten(self._ensemble.trainable_variables)
    with tf.GradientTape() as tape:
      q_values = self._ensemble(o_tm1)
      one_hot_actions = tf.one_hot(a_tm1, depth=self._num_actions)
      one_hot_actions = tf.expand_dims(one_hot_actions, axis=-1)
      train_value = tf.reduce_sum(q_values * one_hot_actions, axis=1)

      target_value = tf.stop_gradient(tf.reduce_max(
                                        self._target_ensemble(o_t), axis=1))

      r_t = tf.expand_dims(r_t, axis=-1)
      d_t = tf.expand_dims(d_t, axis=-1)
      target_y = r_t + z_t + self._discount * d_t * target_value
      loss = tf.square(train_value - target_y) * m_t
      loss = tf.reduce_mean(loss)
      reg_loss = 0
      for var in variables:
        reg_loss += tf.reduce_mean(tf.square(var))

      reg_loss *= self._regularization_penalty / replay_size
      gradients = tape.gradient(loss + reg_loss, variables)

    self._total_steps.assign_add(1)
    self._optimizer.apply(gradients, variables)

    # Langevin update
    opt_steps = tf.cast(self._optimizer.step, tf.float32)
    scaled_lr = 2 * self._optimizer.learning_rate * tf.sqrt(
      1 - tf.pow(self._optimizer.beta2, opt_steps)) / (
        1 - tf.pow(self._optimizer.beta1, opt_steps)) / replay_size
    for ind, var in enumerate(variables):
      grad_perturb = tf.divide(
        tf.random.normal(shape=var.shape, dtype=var.dtype),
        tf.sqrt(tf.sqrt(self._optimizer.v[ind]) + self._optimizer.epsilon))
      variables[ind].assign(var - self._noise_scale * tf.sqrt(scaled_lr) * 
                 grad_perturb)

    # Periodically update the target network.
    if tf.math.mod(self._total_steps, self._target_update_period) == 0:
      for src, dest in zip(self._ensemble.variables,
                           self._target_ensemble.variables):
          dest.assign(src)

  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
    """Select values via Thompson sampling, then use epsilon-greedy policy."""
    if self._rng.rand() < self._epsilon_fn(self._total_steps.numpy()):
      return self._rng.randint(self._num_actions)

    # Greedy policy, breaking ties uniformly at random.
    batched_obs = tf.expand_dims(timestep.observation, axis=0)
    q_values = self._forward(batched_obs)[0, :, self._active_head].numpy()
    action = self._rng.choice(np.flatnonzero(q_values == q_values.max()))
    return int(action)

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: base.Action,
      new_timestep: dm_env.TimeStep,
  ):
    """Update the agent: add transition to replay and periodically do SGD."""
    self._replay.add(
        TransitionWithMaskAndNoise(
            o_tm1=timestep.observation,
            a_tm1=action,
            r_t=np.float32(new_timestep.reward),
            d_t=np.float32(new_timestep.discount),
            o_t=new_timestep.observation,
            m_t=self._rng.binomial(1, self._mask_prob,
                                   self._num_ensemble).astype(np.float32),
            z_t=self._rng.randn(self._num_ensemble).astype(np.float32) *
            self._noise_scale,
        ))

    if new_timestep.last():
      self._active_head = self._rng.randint(self._num_ensemble)


    if self._replay.size < self._min_replay_size:
      return


    self._total_update_steps += 1

    if new_timestep.last() or self._update_every_step:
      while(self._total_update_steps * self._sgds_per_step > 
          self._total_steps.numpy()):
        # if tf.math.mod(self._total_steps, self._sgd_period) == 0:
        minibatch = self._replay.sample(self._batch_size)
        minibatch = [tf.convert_to_tensor(x) for x in minibatch]
        replay_size = tf.cast(self._replay.size, tf.float32)
        self._langevin_step(minibatch, replay_size)


class TransitionWithMaskAndNoise(NamedTuple):
  o_tm1: np.ndarray
  a_tm1: base.Action
  r_t: float
  d_t: float
  o_t: np.ndarray
  m_t: np.ndarray
  z_t: np.ndarray


def make_ensemble(num_actions: int,
                  num_ensemble: int = 20,
                  num_hidden_layers: int = 2,
                  num_units: int = 50,
                  prior_scale: float = 3.,
                  relu_alpha : float = 0.,
                  with_bias: bool = True) -> snt.Module:
  """Convenience function to make an ensemble from flags."""
  output_sizes = [num_units] * num_hidden_layers + [num_actions]
  network = networks.EfficientEnsemble(output_sizes, num_ensemble,
                                       relu_alpha, with_bias)
  prior_network = networks.EfficientEnsemble(output_sizes, num_ensemble,
                                       relu_alpha, with_bias)
  ensemble = networks.NetworkWithPrior(network, prior_network, prior_scale)
  return ensemble


def default_agent(
    obs_spec: specs.Array,
    action_spec: specs.DiscreteArray,
    num_ensemble: int = 20,
) -> EnsembleLangevinDqn:
  """Initialize a Ensemble Langevin DQN agent with default parameters."""
  ensemble = make_ensemble(
      num_actions=action_spec.num_values, num_ensemble=num_ensemble)
  optimizer = snt.optimizers.Adam(learning_rate=1e-3)
  return LangevinBootstrappedDqn(
      obs_spec=obs_spec,
      action_spec=action_spec,
      ensemble=ensemble,
      num_ensemble=num_ensemble,
      batch_size=128,
      discount=.99,
      replay_capacity=10000,
      min_replay_size=128,
      sgds_per_step=1,
      target_update_period=4,
      optimizer=optimizer,
      mask_prob=0.5,
      noise_scale=0.1,
      epsilon_fn=lambda t: 0,
      update_every_step=True,
      seed=42,
  )
