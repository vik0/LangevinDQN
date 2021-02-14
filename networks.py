import copy
from typing import Callable, NamedTuple, Sequence

import dm_env
from dm_env import specs
import numpy as np
import sonnet as snt
import tensorflow as tf
import tree


class EfficientEnsembleLayer(snt.Module):
    """A single layer to implement efficient ensemble network
    with dense layers."""

    def __init__(self,
                 output_dim: int, 
                 name: str = 'efficient_ensemble',
                 with_bias: bool = True,
                 first_layer: bool = False):
        super(EfficientEnsembleLayer, self).__init__(name=name)
        self._weights = None
        self._bias = None
        self._with_bias = with_bias
        self._output_dim = output_dim
        self._first_layer = first_layer

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        assert inputs.get_shape().ndims == 3 
        # input.shape = data_batch_size x input_dim x num_ensembles 
        self.make_weights_and_bias(inputs)

        outputs = tf.einsum('bhk, hjk->bjk', inputs, self._weights)
        if self._with_bias:
            outputs += self._bias
        return outputs

    @snt.once
    def make_weights_and_bias(self, inputs: tf.Tensor):
        _ , input_dim, num_ensembles = inputs.get_shape().as_list()
        stddev = 1.0 / np.sqrt(input_dim)

        if self._with_bias:
            self._bias = tf.Variable(
                tf.zeros(shape=[1, self._output_dim, num_ensembles],
                        dtype=tf.float32), name='bias')
        else:
            self._bias = 0

        self._weights = tf.Variable(
            tf.random.truncated_normal(
                shape=[input_dim, self._output_dim, num_ensembles],
                stddev=stddev, dtype=tf.float32),
            name='weights')

        return


class EfficientEnsemble(snt.Module):
    """Efficient ensemble with Leaky ReLU activation."""

    def __init__(self, output_sizes: Sequence[int], num_ensembles: int,
                 relu_alpha: float = 0, with_bias: bool=True):
        super(EfficientEnsemble, self).__init__(name='efficient_ensemble')
        self._num_ensembles = num_ensembles
        self._num_layers = len(output_sizes)

        self._relu_alpha = relu_alpha
        self._layers = [EfficientEnsembleLayer(output_sizes[0], 
            with_bias=with_bias, first_layer=True)]
        for output_dim in output_sizes[1:]:
            self._layers.append(EfficientEnsembleLayer(output_dim, with_bias=with_bias))

    def __call__(self, inputs) -> tf.Tensor:
        inputs = snt.flatten(inputs)
        outputs = tf.stack([inputs] * self._num_ensembles, axis=2)
        for layer_num, layer in enumerate(self._layers):
            outputs = layer(outputs)
            if layer_num != self._num_layers - 1:
                outputs = tf.nn.leaky_relu(outputs, alpha=self._relu_alpha)
        return outputs


class NetworkWithPrior(snt.Module):
  """Combines network with additive untrainable "prior network"."""

  def __init__(self,
               network: snt.Module,
               prior_network: snt.Module,
               prior_scale: float = 1.):
    super().__init__(name='network_with_prior')
    self._network = network
    self._prior_network = prior_network
    self._prior_scale = prior_scale

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    q_values = self._network(inputs)
    prior_q_values = self._prior_network(inputs)
    return q_values + self._prior_scale * tf.stop_gradient(prior_q_values)
