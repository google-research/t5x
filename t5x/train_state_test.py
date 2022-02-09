# Copyright 2022 The T5X Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for train_state."""
from absl.testing import absltest
from flax import linen as nn
from flax import optim
import flax.core
from flax.linen import partitioning as flax_partitioning
import jax
import numpy as np
from t5x import adafactor
from t5x import partitioning
from t5x import train_state as train_state_lib

mock = absltest.mock
AxisMetadata = flax_partitioning.AxisMetadata
FactorDim = adafactor.FactorDim


class FlaxOptimTrainState(absltest.TestCase):

  def test_init(self):
    model = nn.Dense(10)
    inputs = np.ones([2, 3], dtype=np.float32)
    params = model.init(jax.random.PRNGKey(0), inputs)['params']
    optimizer_def = optim.Adam(learning_rate=0.1)
    optimizer = optimizer_def.create(params)
    flax_mutables = flax.core.freeze({'flax_mutable1': np.ones(10)})
    state = train_state_lib.FlaxOptimTrainState(
        optimizer, flax_mutables=flax_mutables)
    self.assertEqual(state.step, 0)
    self.assertIsInstance(state._optimizer, optim.Optimizer)
    self.assertEqual(state.state_dict()['flax_mutables'],
                     flax.core.unfreeze(flax_mutables))
    jax.tree_multimap(np.testing.assert_array_equal, params, state.params)
    jax.tree_multimap(np.testing.assert_array_equal,
                      optimizer.state.param_states, state.param_states)

  def test_create(self):
    model_variables = flax.core.freeze({
        'params': {
            'dense': {
                'bias': np.zeros(4),
                'kernel': np.zeros((2, 4))
            }
        },
        'mutables': np.ones(3)
    })
    optmizer_def = optim.GradientDescent(0.42)
    state = train_state_lib.FlaxOptimTrainState.create(optmizer_def,
                                                       model_variables)
    self.assertEqual(state.step, 0)
    self.assertIsInstance(state._optimizer, optim.Optimizer)
    self.assertEqual(state._optimizer.optimizer_def, optmizer_def)
    jax.tree_multimap(np.testing.assert_array_equal, state.flax_mutables,
                      flax.core.freeze({'mutables': np.ones(3)}))
    jax.tree_multimap(np.testing.assert_array_equal, model_variables['params'],
                      state.params)
    self.assertIsNone(state.params_axes)

  def test_create_with_params_axes(self):
    model_variables = flax.core.freeze({
        'params': {
            'dense': {
                'bias': np.zeros(4),
                'kernel': np.zeros((2, 4))
            }
        },
        'params_axes': {
            'dense': {
                'bias_axes': AxisMetadata(names=('embed',)),
                'kernel_axes': AxisMetadata(names=('vocab', 'embed')),
            }
        },
    })
    optmizer_def = adafactor.Adafactor(
        0.42,
        logical_factor_rules={
            'vocab': FactorDim.COLUMN,
            'embed': FactorDim.ROW
        })
    state = train_state_lib.FlaxOptimTrainState.create(optmizer_def,
                                                       model_variables)
    self.assertEqual(state.step, 0)
    self.assertIsInstance(state._optimizer, optim.Optimizer)
    self.assertEqual(state._optimizer.optimizer_def, optmizer_def)
    self.assertDictEqual(
        state._optimizer.optimizer_def.hyper_params.factor_map, {
            'dense/bias': (FactorDim.NONE,),
            'dense/kernel': (FactorDim.COLUMN, FactorDim.ROW)
        })
    self.assertEqual(state.flax_mutables, flax.core.freeze({}))
    jax.tree_multimap(np.testing.assert_array_equal, model_variables['params'],
                      state.params)
    jax.tree_multimap(np.testing.assert_array_equal,
                      model_variables['params_axes'], state.params_axes)

  def test_create_missing_params_axes(self):
    model_variables = flax.core.freeze({
        'params': {
            'dense': {
                'bias': np.zeros(4),
                'kernel': np.zeros((2, 4))
            }
        },
        'mutables': np.ones(3)
    })
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'The optimizer supports params_axes for model-based partitioning, but '
        'the model is not emitting them.'):
      train_state_lib.FlaxOptimTrainState.create(adafactor.Adafactor(),
                                                 model_variables)

  def test_replace_params(self):
    optimizer_def = optim.GradientDescent(learning_rate=0.1)
    optimizer = optimizer_def.create({'test': np.ones(10)})
    state = train_state_lib.FlaxOptimTrainState(optimizer)

    new_params = {'test': np.zeros(10)}
    new_state = state.replace_params(new_params)
    jax.tree_multimap(np.testing.assert_array_equal, new_params,
                      new_state.params)
    expected_state_dict = state.state_dict()
    expected_state_dict['target'] = new_params
    jax.tree_multimap(np.testing.assert_array_equal, expected_state_dict,
                      new_state.state_dict())

  def test_replace_step(self):
    optimizer_def = optim.Adam(learning_rate=0.1)
    optimizer = optimizer_def.create({'test': np.ones(10)})
    state = train_state_lib.FlaxOptimTrainState(optimizer)

    self.assertEqual(state.step, 0)
    self.assertEqual(state.replace_step(jax.numpy.array(1)).step, 1)

  def test_apply_gradient(self):
    updated_optimizer = object()
    optimizer = mock.Mock(
        apply_gradient=mock.Mock(return_value=updated_optimizer))
    state = train_state_lib.FlaxOptimTrainState(optimizer)

    new_flax_mutables = {'test': 44}
    new_state = state.apply_gradient(
        grads=42, learning_rate=43, flax_mutables={'test': 44})

    optimizer.apply_gradient.assert_called_once_with(42, learning_rate=43)

    self.assertEqual(new_state._optimizer, updated_optimizer)
    self.assertEqual(
        new_state,
        train_state_lib.FlaxOptimTrainState(
            updated_optimizer, flax_mutables=new_flax_mutables))

  def test_as_logical_axes(self):
    model_variables = flax.core.freeze({
        'params': {
            'dense': {
                'bias': np.zeros(4),
                'kernel': np.zeros((2, 4))
            }
        },
        'params_axes': {
            'dense': {
                'bias_axes': AxisMetadata(names=('embed',)),
                'kernel_axes': AxisMetadata(names=('vocab', 'embed')),
            }
        },
    })
    optmizer_def = adafactor.Adafactor(
        0.42,
        logical_factor_rules={
            'vocab': FactorDim.COLUMN,
            'embed': FactorDim.ROW
        })
    state = train_state_lib.FlaxOptimTrainState.create(optmizer_def,
                                                       model_variables)
    axes_state = state.as_logical_axes()
    self.assertIsNone(axes_state.params_axes)
    jax.tree_multimap(
        np.testing.assert_array_equal, axes_state.params,
        flax.core.freeze({
            'dense': {
                'bias': partitioning.PartitionSpec('embed'),
                'kernel': partitioning.PartitionSpec('vocab', 'embed'),
            }
        }))

  def test_as_logical_axes_unsupported_optimizer(self):
    model_variables = flax.core.freeze({
        'params': {
            'dense': {
                'bias': np.zeros(4),
                'kernel': np.zeros((2, 4))
            }
        },
        'params_axes': {
            'dense': {
                'bias_axes': AxisMetadata(names=('embed',)),
                'kernel_axes': AxisMetadata(names=('vocab', 'embed')),
            }
        },
    })
    optmizer_def = optim.GradientDescent(0.42)
    state = train_state_lib.FlaxOptimTrainState.create(optmizer_def,
                                                       model_variables)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Optimizer 'GradientDescent' requires a `derive_logical_axes` method "
        'to be used with named axis partitioning.'):
      state.as_logical_axes()


if __name__ == '__main__':
  absltest.main()
