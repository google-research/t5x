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

"""Tests for t5x.adafactor."""

import functools
import operator
from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized

import flax
from flax import traverse_util
import jax
from jax import numpy as jnp
from jax import random
import numpy as np

from t5x import adafactor
from t5x import optimizers

OptimizerState = optimizers.OptimizerState

_AdafactorHyperParams = adafactor._AdafactorHyperParams
_AdafactorParamState = adafactor._AdafactorParamState

_BATCH = adafactor.FactorDim.BATCH
_ROW = adafactor.FactorDim.ROW
_COL = adafactor.FactorDim.COLUMN

# Testing helpers


def _assert_numpy_allclose(a, b, atol=None, rtol=None):
  a, b = jnp.array(a), jnp.array(b)
  a = a.astype(np.float32) if a.dtype == jnp.bfloat16 else a
  b = b.astype(np.float32) if b.dtype == jnp.bfloat16 else b
  kw = {}
  if atol:
    kw['atol'] = atol
  if rtol:
    kw['rtol'] = rtol
  np.testing.assert_allclose(a, b, **kw)


def check_eq(xs, ys, atol=None, rtol=None):
  xs_leaves, xs_tree = jax.tree_flatten(xs)
  ys_leaves, ys_tree = jax.tree_flatten(ys)
  assert xs_tree == ys_tree, f"Tree shapes don't match. \n{xs_tree}\n{ys_tree}"
  assert jax.tree_util.tree_all(
      jax.tree_map(lambda x, y: np.array(x).shape == np.array(y).shape,
                   xs_leaves, ys_leaves)), "Leaves' shapes don't match."
  assert jax.tree_map(
      functools.partial(_assert_numpy_allclose, atol=atol, rtol=rtol),
      xs_leaves, ys_leaves)


def flattened_state_dict(x):
  s = flax.serialization.to_state_dict(x)
  return flax.traverse_util.flatten_dict(s, sep='/')


def tree_shape(x):
  return jax.tree_map(jnp.shape, x)


def tree_equals(x, y):
  return jax.tree_util.tree_all(jax.tree_map(operator.eq, x, y))


def _get_multi_adafactor(
    learning_rate: float, step_offset: int,
    adafactor_exclude_from_parameter_scale: Sequence[str]
) -> optimizers.MultiOptimizer:
  """Get adafactor with support for excluding some parameters from scaling."""

  def _should_not_scale(path):
    return any([s in path for s in adafactor_exclude_from_parameter_scale])

  scaled_vars = traverse_util.ModelParamTraversal(
      lambda path, _: not _should_not_scale(path))
  unscaled_vars = traverse_util.ModelParamTraversal(
      lambda path, _: _should_not_scale(path))
  scaled_opt = adafactor.Adafactor(
      learning_rate, decay_rate=0.8, step_offset=step_offset)
  unscaled_opt = adafactor.Adafactor(
      learning_rate,
      decay_rate=0.8,
      step_offset=step_offset,
      multiply_by_parameter_scale=False)
  return optimizers.MultiOptimizer(
      ((scaled_vars, scaled_opt), (unscaled_vars, unscaled_opt)))


# Inline test data

MODEL_SHAPE = {
    'decoder': {
        'decoder_norm': {'scale': [128]},
        'layers_0': {
            'encoder_decoder_attention': {
                'key': {'kernel': [128, 256]},
                'out': {'kernel': [256, 128]},
                'query': {'kernel': [128, 256]},
                'value': {'kernel': [128, 256]}},
            'mlp': {
                'wi': {'kernel': [128, 512]},
                'wo': {'kernel': [512, 128]}},
            'pre_cross_attention_layer_norm': {'scale': [128]},
            'pre_mlp_layer_norm': {'scale': [128]},
            'pre_self_attention_layer_norm': {'scale': [128]},
            'self_attention': {
                'key': {'kernel': [128, 256]},
                'out': {'kernel': [256, 128]},
                'query': {'kernel': [128, 256]},
                'value': {'kernel': [128, 256]}}},
        'layers_1': {
            'encoder_decoder_attention': {
                'key': {'kernel': [128, 128]},
                'out': {'kernel': [128, 128]},
                'query': {'kernel': [128, 128]},
                'value': {'kernel': [128, 128]}},
            'mlp': {
                'wi': {'kernel': [128, 512]},
                'wo': {'kernel': [512, 128]}},
            'pre_cross_attention_layer_norm': {'scale': [128]},
            'pre_mlp_layer_norm': {'scale': [128]},
            'pre_self_attention_layer_norm': {'scale': [128]},
            'self_attention': {
                'key': {'kernel': [128, 256]},
                'out': {'kernel': [256, 128]},
                'query': {'kernel': [128, 256]},
                'value': {'kernel': [128, 256]}}},
        'relpos_bias': {'rel_embedding': [2, 32]}},
    'encoder': {
        'encoder_norm': {'scale': [128]},
        'layers_0': {
            'attention': {
                'key': {'kernel': [128, 256]},
                'out': {'kernel': [256, 128]},
                'query': {'kernel': [128, 256]},
                'value': {'kernel': [128, 256]}},
            'mlp': {
                'wi': {'kernel': [128, 512]},
                'wo': {'kernel': [512, 128]}},
            'pre_attention_layer_norm': {'scale': [128]},
            'pre_mlp_layer_norm': {'scale': [128]}},
        'layers_1': {
            'attention': {
                'key': {'kernel': [128, 256]},
                'out': {'kernel': [256, 128]},
                'query': {'kernel': [128, 256]},
                'value': {'kernel': [128, 256]}},
            'mlp': {
                'wi': {'kernel': [128, 512]},
                'wo': {'kernel': [512, 128]}},
            'pre_attention_layer_norm': {'scale': [128]},
            'pre_mlp_layer_norm': {'scale': [128]}},
        'relpos_bias': {'rel_embedding': [2, 32]}},
    'token_embedder': {'embedding': [32128, 128]}}  # pyformat: disable


class AdafactorTest(parameterized.TestCase):

  # Classic Adafactor Behavior Tests

  def test_2D_simple(self):
    x = {'a': jnp.ones((24, 16))}
    opt_def = adafactor.Adafactor(min_dim_size_to_factor=8)
    optimizer = opt_def.create(x)
    shapes = tree_shape(flattened_state_dict(optimizer.state.param_states))
    ref = {'a/m': (1,), 'a/v': (1,), 'a/v_col': (24,), 'a/v_row': (16,)}
    self.assertTrue(tree_equals(shapes, ref))

  def test_2D_simple_nofactor(self):
    x = {'a': jnp.ones((24, 16))}
    opt_def = adafactor.Adafactor(min_dim_size_to_factor=32)
    optimizer = opt_def.create(x)
    shapes = tree_shape(flattened_state_dict(optimizer.state.param_states))
    ref = {'a/m': (1,), 'a/v': (24, 16), 'a/v_col': (1,), 'a/v_row': (1,)}
    self.assertTrue(tree_equals(shapes, ref))

  def test_2D_simple_nofactor_momentum(self):
    x = {'a': jnp.ones((24, 16))}
    opt_def = adafactor.Adafactor(min_dim_size_to_factor=32, beta1=0.1)
    optimizer = opt_def.create(x)
    shapes = tree_shape(flattened_state_dict(optimizer.state.param_states))
    ref = {'a/m': (24, 16), 'a/v': (24, 16), 'a/v_col': (1,), 'a/v_row': (1,)}
    self.assertTrue(tree_equals(shapes, ref))

  def test_3D_simple(self):
    x = {'a': jnp.ones((24, 4, 16))}
    factor_map = adafactor.HParamMap((('a', (_COL, _BATCH, _ROW)),))
    opt_def = adafactor.Adafactor(
        min_dim_size_to_factor=8, factor_map=factor_map)
    optimizer = opt_def.create(x)
    shapes = tree_shape(flattened_state_dict(optimizer.state.param_states))
    ref = {'a/m': (1,), 'a/v': (1,), 'a/v_col': (24, 4), 'a/v_row': (4, 16)}
    self.assertTrue(tree_equals(shapes, ref))

  def test_init_state(self):
    params = {'x': np.zeros((3, 2))}
    optimizer_def = adafactor.Adafactor(
        learning_rate=0.1, decay_rate=0.8, beta1=None, min_dim_size_to_factor=0)
    state = optimizer_def.init_state(params)

    expected_hyper_params = _AdafactorHyperParams(0.1, True, True, None, 0.8, 0,
                                                  1.0, None, 0, 1e-30, 1e-3)
    self.assertEqual(optimizer_def.hyper_params, expected_hyper_params)
    expected_state = OptimizerState(
        0, {
            'x':
                _AdafactorParamState(
                    np.zeros((2,)), np.zeros((3,)), np.zeros(
                        (1,)), np.zeros((1,)))
        })
    check_eq(state, expected_state)

    # unfactorized
    optimizer_def = adafactor.Adafactor(
        learning_rate=0.1, decay_rate=0.8, beta1=0.0, min_dim_size_to_factor=32)
    state = optimizer_def.init_state(params)

    expected_hyper_params = _AdafactorHyperParams(0.1, True, True, 0.0, 0.8, 0,
                                                  1.0, None, 32, 1e-30, 1e-3)
    self.assertEqual(optimizer_def.hyper_params, expected_hyper_params)
    expected_state = OptimizerState(
        0, {
            'x':
                _AdafactorParamState(
                    np.zeros((1,)), np.zeros((1,)), np.zeros(
                        (3, 2)), np.zeros((3, 2)))
        })
    check_eq(state, expected_state)

  def test_apply_gradient(self):
    optimizer_def = adafactor.Adafactor(
        learning_rate=0.1, decay_rate=0.8, min_dim_size_to_factor=0)
    params = {'x': np.ones((3, 2), np.float32)}
    state = OptimizerState(
        1, {
            'x':
                _AdafactorParamState(
                    np.array([0.9, 0.9]), np.array([0.1, 0.1, 0.1]),
                    np.zeros((1,)), np.zeros((1,)))
        })
    grads = {'x': np.ones((3, 2), np.float32)}
    new_params, new_state = optimizer_def.apply_gradient(
        optimizer_def.hyper_params, params, state, grads)
    expected_new_state = OptimizerState(
        2, {
            'x':
                _AdafactorParamState(
                    np.array([0.9574349, 0.9574349]),
                    np.array([0.6169143, 0.6169143, 0.6169143]), np.zeros(
                        (1,)), np.zeros((1,)))
        })
    expected_new_params = {'x': 0.9 * np.ones((3, 2))}
    check_eq(new_params, expected_new_params)
    check_eq(new_state, expected_new_state, rtol=1e-6)

    # unfactored w momentum
    optimizer_def = adafactor.Adafactor(
        learning_rate=0.1, beta1=0.0, decay_rate=0.8, min_dim_size_to_factor=32)
    params = {'x': np.ones((3, 2), np.float32)}
    state = OptimizerState(
        1, {
            'x':
                _AdafactorParamState(
                    np.zeros(1,), np.zeros(1,), 0.5 * np.ones(
                        (3, 2)), np.zeros((3, 2)))
        })
    grads = {'x': np.ones((3, 2), np.float32)}
    new_params, new_state = optimizer_def.apply_gradient(
        optimizer_def.hyper_params, params, state, grads)
    expected_new_params = {'x': 0.9 * np.ones((3, 2))}
    check_eq(new_params, expected_new_params)
    expected_new_state = OptimizerState(
        2, {
            'x':
                _AdafactorParamState(
                    np.array([0.0]), np.array([0.0]), 0.787174 * np.ones(
                        (3, 2)), 0.1 * np.ones((3, 2)))
        })
    check_eq(new_state, expected_new_state, rtol=1e-6)

  def test_apply_gradient_with_global_norm_clipping(self):
    optimizer_def = adafactor.Adafactor(
        learning_rate=0.1,
        decay_rate=0.8,
        min_dim_size_to_factor=0,
        global_norm_clip_threshold=1.0)
    params = {'x': np.ones((3, 2), np.float32)}
    state = OptimizerState(
        1, {
            'x':
                _AdafactorParamState(
                    np.array([0.9, 0.9]), np.array([0.1, 0.1, 0.1]),
                    np.zeros((1,)), np.zeros((1,)))
        })
    grads = {'x': np.ones((3, 2), np.float32)}
    new_params, new_state = optimizer_def.apply_gradient(
        optimizer_def.hyper_params, params, state, grads)
    expected_new_state = OptimizerState(
        2, {
            'x':
                _AdafactorParamState(
                    np.array([0.478811, 0.478811]),
                    np.array([0.13829, 0.13829, 0.13829]), np.zeros(
                        (1,)), np.zeros((1,)))
        })
    expected_new_params = {'x': 0.9 * np.ones((3, 2))}
    check_eq(new_params, expected_new_params)
    check_eq(new_state, expected_new_state, rtol=1e-6)

  def test_factorizes(self):
    params = {'x': np.zeros((64, 64))}
    optimizer_def = adafactor.Adafactor(
        learning_rate=0.1,
        decay_rate=0.8,
        beta1=None,
        min_dim_size_to_factor=32)
    state = optimizer_def.init_state(params)
    self.assertEqual(state.param_states['x'].v.shape, (1,))
    self.assertEqual(state.param_states['x'].m.shape, (1,))
    self.assertEqual(state.param_states['x'].v_row.shape, (64,))
    self.assertEqual(state.param_states['x'].v_col.shape, (64,))

    params = {'x': np.zeros((31, 64))}
    optimizer_def = adafactor.Adafactor(
        learning_rate=0.1,
        decay_rate=0.8,
        beta1=None,
        min_dim_size_to_factor=32)
    state = optimizer_def.init_state(params)
    self.assertEqual(state.param_states['x'].v.shape, (31, 64))
    self.assertEqual(state.param_states['x'].m.shape, (1,))
    self.assertEqual(state.param_states['x'].v_row.shape, (1,))
    self.assertEqual(state.param_states['x'].v_col.shape, (1,))

  # Manually specified factorization rules tests.

  @parameterized.parameters(
      {'rule': (_ROW, _COL)},
      {'rule': (_COL, _ROW)},
  )
  def test_2D_ignore_specified_factor_rule(self, rule):
    x = {'a': jnp.ones((24, 16))}
    factor_map = adafactor.HParamMap((('a', rule),))
    opt_def = adafactor.Adafactor(
        min_dim_size_to_factor=8, factor_map=factor_map)
    optimizer = opt_def.create(x)
    shapes = tree_shape(flattened_state_dict(optimizer.state.param_states))
    # Since param is 2D, the explicit factor rule should be ignored and falls
    # back to heuristics where v_row corresponds to the smaller dim.
    ref = {'a/m': (1,), 'a/v': (1,), 'a/v_col': (24,), 'a/v_row': (16,)}
    self.assertTrue(tree_equals(shapes, ref))

  def test_3D_simple_manual_rules(self):
    x = {'a': jnp.ones((24, 4, 16))}

    factor_map = adafactor.HParamMap((('a', (_COL, _BATCH, _ROW)),))
    opt_def = adafactor.Adafactor(
        min_dim_size_to_factor=8, factor_map=factor_map)
    optimizer = opt_def.create(x)
    shapes = tree_shape(flattened_state_dict(optimizer.state.param_states))
    ref = {'a/m': (1,), 'a/v': (1,), 'a/v_col': (24, 4), 'a/v_row': (4, 16)}
    self.assertTrue(tree_equals(shapes, ref))

    factor_map = adafactor.HParamMap((('a', (_ROW, _BATCH, _COL)),))
    opt_def = adafactor.Adafactor(
        min_dim_size_to_factor=8, factor_map=factor_map)
    optimizer = opt_def.create(x)
    shapes = tree_shape(flattened_state_dict(optimizer.state.param_states))
    ref = {'a/m': (1,), 'a/v': (1,), 'a/v_col': (4, 16), 'a/v_row': (24, 4)}
    self.assertTrue(tree_equals(shapes, ref))

    factor_map = adafactor.HParamMap((('a', (_COL, _ROW, _ROW)),))
    opt_def = adafactor.Adafactor(
        min_dim_size_to_factor=8, factor_map=factor_map)
    optimizer = opt_def.create(x)
    shapes = tree_shape(flattened_state_dict(optimizer.state.param_states))
    ref = {'a/m': (1,), 'a/v': (1,), 'a/v_col': (24,), 'a/v_row': (4, 16)}
    self.assertTrue(tree_equals(shapes, ref))

    factor_map = adafactor.HParamMap((('a', (_COL, _COL, _ROW)),))
    opt_def = adafactor.Adafactor(
        min_dim_size_to_factor=8, factor_map=factor_map)
    optimizer = opt_def.create(x)
    shapes = tree_shape(flattened_state_dict(optimizer.state.param_states))
    ref = {'a/m': (1,), 'a/v': (1,), 'a/v_col': (24, 4), 'a/v_row': (16,)}
    self.assertTrue(tree_equals(shapes, ref))

  def test_standard_factor_rules(self):
    # one-off test to double-check that we're following the previous
    # heuristic convention for rows/columns.
    def test_standard_factor_rules():
      token_embedding = (_COL, _ROW)
      attn_qkv = (_ROW, _COL)
      attn_out = (_COL, _ROW)
      mlp_in = (_ROW, _COL)
      mlp_out = (_COL, _ROW)
      return ((r'_layer_norm/(bias|scale)',
               None), (r'(encoder|decoder)_norm/(bias|scale)', None),
              (r'(encoder_decoder_|self_|\b)attention/(query|key|value)/kernel',
               attn_qkv), (r'(encoder_decoder_|self_|\b)attention/out/kernel',
                           attn_out), (r'mlp/DenseGeneral_\d+/bias', None),
              (r'mlp/wi(_\d+)?/kernel', mlp_in), (r'mlp/wo/kernel', mlp_out),
              (r'\brelpos_bias', None), (r'token_embedder', token_embedding),
              (r'.*', adafactor.HEURISTIC_RULE))

    # create fake model parameters
    k = jax.random.PRNGKey(0)
    params = jax.tree_map(
        lambda shape: jax.random.uniform(k, shape),
        MODEL_SHAPE,
        is_leaf=lambda x: isinstance(x, list))
    # make traditional adafactor state with heuristic
    factor_map1 = adafactor.HParamMap(((r'.*', adafactor.HEURISTIC_RULE),))
    optimizer_def1 = adafactor.Adafactor(
        0.1,
        decay_rate=0.8,
        step_offset=0,
        multiply_by_parameter_scale=True,
        factor_map=factor_map1)
    optimizer1 = optimizer_def1.create(params)
    # make traditional adafactor state with explicit rules
    factor_map2 = adafactor.HParamMap(test_standard_factor_rules())
    optimizer_def2 = adafactor.Adafactor(
        0.1,
        decay_rate=0.8,
        step_offset=0,
        multiply_by_parameter_scale=True,
        factor_map=factor_map2)
    optimizer2 = optimizer_def2.create(params)
    # are they the same?
    check_eq(optimizer1.state.param_states, optimizer2.state.param_states)

  @parameterized.parameters(
      {'shape': (64, 64)},
      {'shape': (64, 132)},
      {'shape': (132, 64)},
      {'shape': (132, 132)},
      {'shape': (132, 140)},
      {'shape': (140, 132)},
  )
  def test_no_factor_map_equivalence(self, shape):
    k = random.PRNGKey(0)
    k1, k2 = random.split(k)
    p = {'a': random.uniform(k1, shape)}
    g = {'a': random.uniform(k2, shape)}

    orig_opt = adafactor.Adafactor(0.1).create(p)
    new_opt = adafactor.Adafactor(0.1, factor_map=None).create(p)
    check_eq(orig_opt.state_dict(), new_opt.state_dict())

    orig_opt1 = orig_opt.apply_gradient(g)
    new_opt1 = new_opt.apply_gradient(g)
    check_eq(orig_opt1.state_dict(), new_opt1.state_dict())

  @parameterized.parameters({
      'shape': (128, 128),
      'rule': (_ROW, _COL)
  }, {
      'shape': (132, 128),
      'rule': (_COL, _ROW)
  }, {
      'shape': (128, 132),
      'rule': (_ROW, _COL)
  })
  def test_simple_equivalence(self, shape, rule):
    k = random.PRNGKey(0)
    k1, k2 = random.split(k)
    k3, k4 = random.split(k1)
    k5, k6 = random.split(k2)

    p = {'a': random.uniform(k3, shape), 'b': random.uniform(k4, shape)}
    g = {'a': random.uniform(k5, shape), 'b': random.uniform(k6, shape)}

    orig_opt = adafactor.Adafactor(0.1).create(p)
    factor_map = adafactor.HParamMap(
        rules=((('a'), rule), ('.*', adafactor.HEURISTIC_RULE)))
    new_opt = adafactor.Adafactor(0.1, factor_map=factor_map).create(p)
    check_eq(orig_opt.state_dict(), new_opt.state_dict())

    orig_opt1 = orig_opt.apply_gradient(g)
    new_opt1 = new_opt.apply_gradient(g)
    check_eq(orig_opt1.state_dict(), new_opt1.state_dict())

  @parameterized.parameters({'shape': (64, 64)}, {'shape': (132, 132)})
  def test_multiply_by_parameter_scale_equivalence(self, shape):
    # Use large parameter values to magnify the parameter scaling effect.
    p = {'a': np.random.randn(*shape) * 100, 'b': np.random.randn(*shape) * 100}
    g = {'a': np.random.randn(*shape), 'b': np.random.randn(*shape)}
    orig_opt = _get_multi_adafactor(
        3.0, 0, adafactor_exclude_from_parameter_scale=('a',)).create(p)
    scaling_map = adafactor.HParamMap([('a', False), ('.*', True)])
    new_opt = adafactor.Adafactor(
        3.0, multiply_by_parameter_scale=scaling_map).create(p)
    check_eq(orig_opt.state_dict(), new_opt.state_dict())

    orig_opt1 = orig_opt.apply_gradient(g)
    new_opt1 = new_opt.apply_gradient(g)
    check_eq(orig_opt1.state_dict(), new_opt1.state_dict())

  def test_3d_without_factor_map(self):
    x = {'a': jnp.ones((24, 4, 16))}
    opt_def = adafactor.Adafactor(factor_map=None)
    with self.assertRaises(ValueError):
      _ = opt_def.create(x)


if __name__ == '__main__':
  absltest.main()
