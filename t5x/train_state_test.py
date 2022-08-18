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
import flax.core
from flax.linen import partitioning as flax_partitioning
import jax
import numpy as np
from t5x import adafactor
from t5x import optimizers
from t5x import partitioning
from t5x import train_state as train_state_lib

mock = absltest.mock
AxisMetadata = flax_partitioning.AxisMetadata
FactorDim = adafactor.FactorDim


class FlaxOptimTrainStateTest(absltest.TestCase):

  def test_init(self):
    model = nn.Dense(10)
    inputs = np.ones([2, 3], dtype=np.float32)
    params = model.init(jax.random.PRNGKey(0), inputs)['params']
    optimizer_def = optimizers.adam(0.1)
    optimizer = optimizer_def.create(params)
    flax_mutables = flax.core.freeze({'flax_mutable1': np.ones(10)})
    state = train_state_lib.FlaxOptimTrainState(
        optimizer, flax_mutables=flax_mutables)
    self.assertEqual(state.step, 0)
    self.assertIsInstance(state._optimizer, optimizers.Optimizer)
    self.assertEqual(state.state_dict()['flax_mutables'],
                     flax.core.unfreeze(flax_mutables))
    jax.tree_map(np.testing.assert_array_equal, params, state.params)
    jax.tree_map(np.testing.assert_array_equal, optimizer.state.param_states,
                 state.param_states)

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
    optimizer_def = optimizers.sgd(0.42)
    state = train_state_lib.FlaxOptimTrainState.create(optimizer_def,
                                                       model_variables)
    self.assertEqual(state.step, 0)
    self.assertIsInstance(state._optimizer, optimizers.Optimizer)
    self.assertEqual(state._optimizer.optimizer_def, optimizer_def)
    jax.tree_map(np.testing.assert_array_equal, state.flax_mutables,
                 flax.core.freeze({'mutables': np.ones(3)}))
    jax.tree_map(np.testing.assert_array_equal, state.params,
                 model_variables['params'])
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
    optimizer_def = adafactor.Adafactor(
        0.42,
        logical_factor_rules={
            'vocab': FactorDim.COLUMN,
            'embed': FactorDim.ROW
        })
    state = train_state_lib.FlaxOptimTrainState.create(optimizer_def,
                                                       model_variables)
    self.assertEqual(state.step, 0)
    self.assertIsInstance(state._optimizer, optimizers.Optimizer)
    self.assertEqual(state._optimizer.optimizer_def, optimizer_def)
    self.assertDictEqual(
        state._optimizer.optimizer_def.hyper_params.factor_map, {
            'dense/bias': (FactorDim.NONE,),
            'dense/kernel': (FactorDim.COLUMN, FactorDim.ROW)
        })
    self.assertEqual(state.flax_mutables, flax.core.freeze({}))
    jax.tree_map(np.testing.assert_array_equal, model_variables['params'],
                 state.params)
    jax.tree_map(np.testing.assert_array_equal, model_variables['params_axes'],
                 state.params_axes)

  def test_create_with_flax_mutables_axes(self):
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
        'grads': {
            'dense': {
                'output_grad': np.zeros(4),
            }
        },
        'grads_axes': {
            'dense': {
                'output_grad': AxisMetadata(names=('embed',)),
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
    self.assertIsInstance(state._optimizer, optimizers.Optimizer)
    self.assertEqual(state._optimizer.optimizer_def, optmizer_def)
    self.assertDictEqual(
        state._optimizer.optimizer_def.hyper_params.factor_map, {
            'dense/bias': (FactorDim.NONE,),
            'dense/kernel': (FactorDim.COLUMN, FactorDim.ROW)
        })
    self.assertEqual(state.flax_mutables,
                     flax.core.freeze({'grads': model_variables['grads']}))
    jax.tree_map(np.testing.assert_array_equal, model_variables['params'],
                 state.params)
    jax.tree_map(np.testing.assert_array_equal, model_variables['params_axes'],
                 state.params_axes)
    jax.tree_map(np.testing.assert_array_equal, model_variables['grads_axes'],
                 state.flax_mutables_axes['grads'])

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

  def test_create_mismatched_params_axes(self):
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
            }
        },
        'mutables': np.ones(3)
    })
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Missing axis names for parameters: {'dense/kernel'}"):
      train_state_lib.FlaxOptimTrainState.create(adafactor.Adafactor(),
                                                 model_variables)

  def test_replace_params(self):
    optimizer_def = optimizers.sgd(0.1)
    optimizer = optimizer_def.create({'test': np.ones(10)})
    state = train_state_lib.FlaxOptimTrainState(optimizer)

    new_params = {'test': np.zeros(10)}
    new_state = state.replace_params(new_params)
    jax.tree_map(np.testing.assert_array_equal, new_params, new_state.params)
    expected_state_dict = state.state_dict()
    expected_state_dict['target'] = new_params
    jax.tree_map(np.testing.assert_array_equal, expected_state_dict,
                 new_state.state_dict())

  def test_replace_step(self):
    optimizer_def = optimizers.adam(0.1)
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
    optimizer_def = adafactor.Adafactor(
        0.42,
        logical_factor_rules={
            'vocab': FactorDim.COLUMN,
            'embed': FactorDim.ROW
        })
    state = train_state_lib.FlaxOptimTrainState.create(optimizer_def,
                                                       model_variables)
    axes_state = state.as_logical_axes()
    self.assertIsNone(axes_state.params_axes)
    jax.tree_map(
        np.testing.assert_array_equal, axes_state.params,
        flax.core.freeze({
            'dense': {
                'bias': partitioning.PartitionSpec('embed'),
                'kernel': partitioning.PartitionSpec('vocab', 'embed'),
            }
        }))

  def test_as_logical_axes_with_flax_mutables(self):
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
        'grads': {
            'dense': {
                'output_grad': np.zeros(4),
            }
        },
        'grads_axes': {
            'dense': {
                'output_grad': AxisMetadata(names=('embed',)),
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
    jax.tree_map(
        np.testing.assert_array_equal, axes_state.flax_mutables,
        flax.core.freeze({
            'grads': {
                'dense': {
                    'output_grad': partitioning.PartitionSpec('embed'),
                }
            }
        }))

  def test_to_state_dict(self):
    model_variables = flax.core.freeze({
        'params': {
            'kernel': np.zeros((2, 4))
        },
        'params_axes': {
            'kernel_axes': AxisMetadata(names=('vocab', 'embed')),
        },
        'mutables': np.ones(3)
    })
    optimizer_def = adafactor.Adafactor(
        0.42,
        logical_factor_rules={
            'vocab': FactorDim.COLUMN,
            'embed': FactorDim.ROW
        })
    state = train_state_lib.FlaxOptimTrainState.create(optimizer_def,
                                                       model_variables)
    jax.tree_map(
        np.testing.assert_array_equal, state.state_dict(), {
            'state': {
                'step': np.array(0),
                'param_states': {
                    'kernel': {
                        'm': np.zeros(1),
                        'v': np.zeros((2, 4)),
                        'v_col': np.zeros(1),
                        'v_row': np.zeros(1)
                    },
                }
            },
            'target': {
                'kernel': np.zeros((2, 4))
            },
            'flax_mutables': {
                'mutables': np.ones(3)
            }
        })

  def test_restore_state(self):
    model_variables = flax.core.freeze({
        'params': {
            'kernel': np.zeros((2, 4))
        },
        'params_axes': {
            'kernel_axes': AxisMetadata(names=('vocab', 'embed')),
        },
        'mutables': np.ones(3)
    })
    optimizer_def = adafactor.Adafactor(
        0.42,
        logical_factor_rules={
            'vocab': FactorDim.COLUMN,
            'embed': FactorDim.ROW
        })
    state = train_state_lib.FlaxOptimTrainState.create(optimizer_def,
                                                       model_variables)
    restored = state.restore_state({
        'state': {
            'step': np.array(1),
            'param_states': {
                'kernel': {
                    'm': np.ones(1),
                    'v': np.ones((2, 4)),
                    'v_col': np.ones(1),
                    'v_row': np.ones(1)
                },
            }
        },
        'target': {
            'kernel': np.ones((2, 4))
        },
        'flax_mutables': {
            'mutables': np.zeros(3)
        }
    })

    self.assertEqual(restored.step, 1)
    self.assertIsInstance(restored._optimizer, optimizers.Optimizer)
    self.assertEqual(restored._optimizer.optimizer_def, optimizer_def)
    jax.tree_map(np.testing.assert_array_equal, restored.flax_mutables,
                 flax.core.freeze({'mutables': np.zeros(3)}))
    jax.tree_map(np.testing.assert_array_equal, restored.params,
                 flax.core.freeze({'kernel': np.ones((2, 4))}))
    jax.tree_map(
        np.testing.assert_array_equal, restored.param_states,
        flax.core.freeze({
            'kernel':
                adafactor._AdafactorParamState(
                    np.ones(1), np.ones(1), np.ones((2, 4)), np.ones(1))
        }))
    jax.tree_map(np.testing.assert_array_equal, restored.params_axes,
                 model_variables['params_axes'])


class InferenceStateTest(absltest.TestCase):

  def test_init(self):
    model = nn.Dense(10)
    inputs = np.ones([2, 3], dtype=np.float32)
    params = model.init(jax.random.PRNGKey(0), inputs)['params']
    flax_mutables = flax.core.freeze({'flax_mutable1': np.ones(10)})
    state = train_state_lib.InferenceState(
        step=jax.numpy.array(1), params=params, flax_mutables=flax_mutables)
    self.assertEqual(state.step, 1)
    self.assertEqual(state.flax_mutables, flax.core.unfreeze(flax_mutables))
    jax.tree_map(np.testing.assert_array_equal, params, state.params)
    self.assertIsNone(state.params_axes)

  def test_create(self):
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
        'mutables': np.ones(3)
    })
    state = train_state_lib.InferenceState.create(model_variables)
    self.assertEqual(state.step, 0)
    jax.tree_map(np.testing.assert_array_equal, state.flax_mutables,
                 flax.core.freeze({'mutables': np.ones(3)}))
    jax.tree_map(np.testing.assert_array_equal, state.params,
                 model_variables['params'])
    jax.tree_map(np.testing.assert_array_equal, state.params_axes,
                 model_variables['params_axes'])

  def test_create_mismatched_params_axes(self):
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
            }
        },
        'mutables': np.ones(3)
    })
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Missing axis names for parameters: {'dense/kernel'}"):
      train_state_lib.InferenceState.create(model_variables)

  def test_replace_params(self):
    model_variables = flax.core.freeze({'params': {'test': np.ones(10)}})
    state = train_state_lib.InferenceState.create(model_variables)

    new_params = {'test': np.zeros(10)}
    new_state = state.replace_params(new_params)
    jax.tree_map(np.testing.assert_array_equal, new_params, new_state.params)

  def test_replace_step(self):
    model_variables = flax.core.freeze({'params': {'test': np.ones(10)}})
    state = train_state_lib.InferenceState.create(model_variables)

    self.assertEqual(state.step, 0)
    self.assertEqual(state.replace_step(jax.numpy.array(1)).step, 1)

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
    state = train_state_lib.InferenceState.create(model_variables)
    axes_state = state.as_logical_axes()
    self.assertIsNone(axes_state.params_axes)
    jax.tree_map(
        np.testing.assert_array_equal, axes_state.params,
        flax.core.freeze({
            'dense': {
                'bias': partitioning.PartitionSpec('embed'),
                'kernel': partitioning.PartitionSpec('vocab', 'embed'),
            }
        }))

  def test_to_state_dict(self):
    model_variables = flax.core.freeze({
        'params': {
            'bias': np.zeros(4),
        },
        'params_axes': {
            'bias_axes': AxisMetadata(names=('embed',)),
        },
        'mutables': np.ones(3)
    })
    state = train_state_lib.InferenceState.create(model_variables)
    jax.tree_map(
        np.testing.assert_array_equal, state.state_dict(), {
            'state': {
                'step': np.array(0)
            },
            'target': {
                'bias': np.zeros(4),
            },
            'flax_mutables': {
                'mutables': np.ones(3)
            }
        })

  def test_to_state_dict_no_mutables(self):
    model_variables = flax.core.freeze({
        'params': {
            'bias': np.zeros(4),
        },
        'params_axes': {
            'bias_axes': AxisMetadata(names=('embed',)),
        },
    })
    state = train_state_lib.InferenceState.create(model_variables)
    jax.tree_map(np.testing.assert_array_equal, state.state_dict(), {
        'state': {
            'step': np.array(0)
        },
        'target': {
            'bias': np.zeros(4),
        },
    })

  def test_restore_state(self):
    state = train_state_lib.InferenceState(
        np.array(0), {'bias': np.zeros(4)},
        {'bias_axes': AxisMetadata(names=('embed',))})

    state_dict = {
        'state': {
            'step': np.array(10)
        },
        'target': {
            'bias': np.ones(4),
        },
        'flax_mutables': {
            'mutables': np.ones(3)
        }
    }
    restored = state.restore_state(state_dict)

    self.assertEqual(restored.step, 10)
    jax.tree_map(np.testing.assert_array_equal, restored.flax_mutables,
                 flax.core.freeze(state_dict['flax_mutables']))
    jax.tree_map(np.testing.assert_array_equal, restored.params,
                 flax.core.freeze(state_dict['target']))
    self.assertEqual(restored.params_axes,
                     {'bias_axes': AxisMetadata(names=('embed',))})

  def test_restore_state_no_mutables_no_axes(self):
    state = train_state_lib.InferenceState(np.array(0), {})

    state_dict = {
        'state': {
            'step': np.array(10)
        },
        'target': {
            'bias': np.zeros(4),
        },
    }
    restored = state.restore_state(state_dict)

    self.assertEqual(restored.step, 10)
    self.assertEqual(restored.flax_mutables, train_state_lib.EMPTY_DICT)
    jax.tree_map(np.testing.assert_array_equal, restored.params,
                 flax.core.freeze(state_dict['target']))
    self.assertIsNone(restored.params_axes)


if __name__ == '__main__':
  absltest.main()
