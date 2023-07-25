# Copyright 2023 The T5X Authors.
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

"""Tests for t5x.optimizers."""

import dataclasses
import functools
import operator

from absl.testing import absltest
from absl.testing import parameterized
import chex
import flax
from flax.core import frozen_dict
import jax
import jax.numpy as jnp
import numpy as np
import optax
import seqio
from t5x import models
from t5x import optimizers
from t5x import partitioning
from t5x import test_utils
from t5x import trainer
from t5x import utils
from t5x.examples.t5 import network


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
  xs_leaves, xs_tree = jax.tree_util.tree_flatten(xs)
  ys_leaves, ys_tree = jax.tree_util.tree_flatten(ys)
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


def get_fake_tokenized_dataset_no_pretokenized(*_, split='validation', **__):
  return test_utils.get_fake_tokenized_dataset(split=split).map(
      lambda x: {k: v for k, v in x.items() if not k.endswith('_pretokenized')})


def get_t5_test_model(optimizer_def,
                      **config_overrides) -> models.EncoderDecoderModel:
  """Returns a tiny T5 1.1 model to use for testing."""
  tiny_config = network.T5Config(
      vocab_size=128,
      dtype='bfloat16',
      emb_dim=8,
      num_heads=4,
      num_encoder_layers=2,
      num_decoder_layers=2,
      head_dim=3,
      mlp_dim=16,
      mlp_activations=('gelu', 'linear'),
      dropout_rate=0.0,
      logits_via_embedding=False,
  )
  tiny_config = dataclasses.replace(tiny_config, **config_overrides)
  vocabulary = test_utils.get_fake_vocab()
  return models.EncoderDecoderModel(
      module=network.Transformer(tiny_config),
      input_vocabulary=vocabulary,
      output_vocabulary=vocabulary,
      optimizer_def=optimizer_def)


def sgd_with_multi_transform():
  """Uses optax.multi_transform to train only decoder parameters."""

  def _mask_fn(params):
    mask = jax.tree_util.tree_map(lambda _: False, params)
    mask = mask.copy(
        {'decoder': jax.tree_util.tree_map(lambda _: True, mask['decoder'])}
    )
    return mask

  return optax.multi_transform(
      {
          False: optax.set_to_zero(),
          True: optax.sgd(1e-2, 0.0),
      },
      _mask_fn,
  )


class BasicTest(chex.TestCase):

  @classmethod
  def get_params(cls):
    return frozen_dict.FrozenDict({
        'forward': {
            'input_layer': {
                'embedding': jnp.zeros([16, 8], dtype=jnp.float32),
            },
            'output_layer': {
                'layer_norm': {
                    'scale': jnp.zeros([8], dtype=jnp.float32),
                },
                'proj': {
                    'bias': jnp.zeros([1], dtype=jnp.float32),
                    'kernel': jnp.zeros([8, 1], dtype=jnp.float32),
                },
            },
        },
        'loss': {
            'loss_fn': {
                'loss_biases': jnp.zeros([2], dtype=jnp.float32),
            },
        },
    })

  @classmethod
  def get_params_shapes(cls):
    return jax.tree_map(jnp.shape, cls.get_params())

  @classmethod
  def get_param_logical_axes(cls):
    return frozen_dict.FrozenDict({
        'forward': {
            'input_layer': {
                'embedding': partitioning.PartitionSpec('vocab', 'embed'),
            },
            'output_layer': {
                'layer_norm': {
                    'scale': partitioning.PartitionSpec('embed',),
                },
                'proj': {
                    'bias':
                        partitioning.PartitionSpec('output_head',),
                    'kernel':
                        partitioning.PartitionSpec('embed', 'output_head'),
                },
            },
        },
        'loss': {
            'loss_fn': {
                'loss_biases': partitioning.PartitionSpec('unmodeled',),
            },
        },
    })

  def test_logical_axes_adamw(self):
    opt = optax.adamw(0.001, weight_decay=0.001)
    wrapper = optimizers.OptaxWrapper(opt)
    optimizer = wrapper.create(self.get_params())
    got = wrapper.derive_logical_axes(optimizer, self.get_param_logical_axes())
    want = optimizers.Optimizer(
        optimizer_def=wrapper,
        state=optimizers.OptimizerState(
            step=None,
            param_states=(
                optax.ScaleByAdamState(
                    count=None,
                    mu=self.get_param_logical_axes(),
                    nu=self.get_param_logical_axes()),
                optax.EmptyState(),
                optax.EmptyState(),
            )),
        target=self.get_param_logical_axes())
    chex.assert_trees_all_equal(got, want)

  @parameterized.parameters(
      ('sgd', lambda: optax.sgd(1e-2, 0.0)),
      ('adam', lambda: optax.adam(1e-1)),
      ('adamw', lambda: optax.adamw(1e-1)),
      ('lamb', lambda: optax.adamw(1e-1)),
      ('lion', lambda: optax.lion(1e-2)),
      ('rmsprop', lambda: optax.rmsprop(1e-1)),
      ('rmsprop_momentum', lambda: optax.rmsprop(5e-2, momentum=0.9)),
      ('fromage', lambda: optax.fromage(1e-2)),
      ('adabelief', lambda: optax.adabelief(1e-1)),
      ('radam', lambda: optax.radam(1e-1)),
      ('yogi', lambda: optax.yogi(1.0)),
  )
  def test_sanity_check_logical_axes(self, opt_name, opt_fn):
    opt = opt_fn()

    wrapper = optimizers.OptaxWrapper(opt)
    optimizer = wrapper.create(self.get_params())
    _ = wrapper.derive_logical_axes(optimizer, self.get_param_logical_axes())

    # TODO(rosun): basic sanity check, we just want to make sure if a param
    # name, e.g., `loss_biases` appear in the tree, the corresponding value is
    # always a PartitionSpec.

  def test_adamw_state_serialization(self):
    opt = optax.adamw(0.001, weight_decay=0.001)
    wrapper = optimizers.OptaxWrapper(opt)
    optimizer = wrapper.create(self.get_params())

    state_dict = optimizer.state_dict()

    chex.assert_trees_all_equal(
        frozen_dict.FrozenDict(jax.tree_map(jnp.shape, state_dict)),
        frozen_dict.FrozenDict({
            'target': self.get_params_shapes(),
            'state': {
                'step': (),
                'param_states': {
                    '0': {
                        'count': (),
                        'mu': self.get_params_shapes(),
                        'nu': self.get_params_shapes(),
                    },
                    # NB: We eliminate empty tuple leaves from EmptyState() in
                    # OptaxWrapper to avoid having the rest of T5X have to
                    # correctly handle this detail. e.g. we omit these:
                    # '1': {},
                    # '2': {},
                },
            }
        }))

    new_optimizer = optimizer.restore_state(state_dict)

    chex.assert_trees_all_equal(optimizer, new_optimizer)


class OptaxWrapperTest(chex.TestCase):

  def run_train_loop(self, optimizer_def):
    # Construct input data.

    ds = get_fake_tokenized_dataset_no_pretokenized(split='validation')
    ds = seqio.EncDecFeatureConverter()(
        ds, task_feature_lengths={
            'inputs': 8,
            'targets': 8
        })
    ds = ds.repeat().batch(8)
    ds_iter = ds.as_numpy_iterator()
    first_batch = next(ds_iter)

    model = get_t5_test_model(optimizer_def, vocab_size=128)

    learning_rate_fn = utils.create_learning_rate_scheduler()

    input_shapes = jax.tree_map(jnp.shape, first_batch)
    input_types = jax.tree_map(lambda x: jnp.dtype(x.dtype), first_batch)

    partitioner = partitioning.PjitPartitioner(
        num_partitions=2,
        logical_axis_rules=partitioning.standard_logical_axis_rules(),
    )

    train_state_initializer = utils.TrainStateInitializer(
        optimizer_def=model.optimizer_def,
        init_fn=model.get_initial_variables,
        input_shapes=input_shapes,
        input_types=input_types,
        partitioner=partitioner)

    train_state_axes = train_state_initializer.train_state_axes
    train_state = train_state_initializer.from_scratch(jax.random.PRNGKey(0))

    trainer_instance = trainer.Trainer(
        model,
        train_state=train_state,
        partitioner=partitioner,
        eval_names=[],
        summary_dir=None,
        train_state_axes=train_state_axes,
        rng=jax.random.PRNGKey(0),
        learning_rate_fn=learning_rate_fn,
        num_microbatches=1)

    chex.assert_tree_all_finite(trainer_instance.train_state.params)
    for _ in range(2):
      trainer_instance.train(ds_iter, 1)
      chex.assert_tree_all_finite(trainer_instance.train_state.params)

    # check save/restore structural equality
    restored_instance = trainer_instance.train_state.restore_state(
        trainer_instance.train_state.state_dict()
    )
    chex.assert_trees_all_equal_structs(
        trainer_instance.train_state, restored_instance
    )

  # NOTE(levskaya): these are surprisingly slow tests on CPU.
  @parameterized.parameters(
      ('sgd', lambda: optax.sgd(1e-2, 0.0)),
      ('adam', lambda: optax.adam(1e-1)),
      ('adamw', lambda: optax.adamw(1e-1)),
      ('lamb', lambda: optax.adamw(1e-1)),
      ('lion', lambda: optax.lion(1e-2)),
      # ('rmsprop', lambda: optax.rmsprop(1e-1)),
      # ('rmsprop_momentum', lambda: optax.rmsprop(5e-2, momentum=0.9)),
      # ('fromage', lambda: optax.fromage(1e-2)),
      ('adabelief', lambda: optax.adabelief(1e-1)),
      # ('radam', lambda: optax.radam(1e-1)),
      ('yogi', lambda: optax.yogi(1.0)),
      ('multi_transform', sgd_with_multi_transform),
  )
  def test_optimizer(self, opt_name, opt_fn):
    opt = opt_fn()
    optimizer_def = optimizers.OptaxWrapper(opt)
    self.run_train_loop(optimizer_def)


if __name__ == '__main__':
  absltest.main()
