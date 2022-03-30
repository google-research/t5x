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

"""Tests for training_utils."""

import functools
import os
# Emulate 2 devices on CPU. Import before JAX.
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
from flax import core as flax_core
import jax
from jax import numpy as jnp
import numpy as np

from t5x.contrib.moe import training_utils


class MatchFnTest(absltest.TestCase):

  def test_regex_prefix(self):
    match_fn = training_utils.match_fn(r'.*test.*')
    self.assertTrue(match_fn('/test/something'))
    self.assertTrue(match_fn('to/test/or/not/'))
    self.assertFalse(match_fn('no/match'))

  def test_empty_prefix(self):
    match_fn = training_utils.match_fn(None)
    self.assertFalse(match_fn('/test/something'))
    self.assertFalse(match_fn('to/test/or/not/'))


class ScaleShardedGradsTest(absltest.TestCase):

  def test_scale_sharded_grads(self):
    grads = flax_core.freeze({
        'encoder': {
            'expert_layer': jnp.ones((2, 3)),
            'regular_layer': jnp.ones((1, 2))
        }
    })
    sharded_match_fn = training_utils.match_fn(r'.*expert.*')
    scaled_grads = training_utils.scale_sharded_grads(
        grads, sharded_match_fn, scale_factor=100.)

    expected_grads = flax_core.freeze({
        'encoder': {
            'expert_layer': 100. * jnp.ones((2, 3)),
            'regular_layer': jnp.ones((1, 2))
        }
    })
    jax.tree_map(
        functools.partial(np.testing.assert_allclose, rtol=3e-7), scaled_grads,
        expected_grads)


class TreeTest(absltest.TestCase):

  def test_tree_flatten_with_names(self):
    tree = {'ff_0': {'kernel': 0, 'bias': 1}, 'ff_1': {'kernel': 2, 'bias': 3}}
    names_and_values, _ = training_utils._tree_flatten_with_names(tree)

    expected_names_and_values = [('ff_0/bias', 1), ('ff_0/kernel', 0),
                                 ('ff_1/bias', 3), ('ff_1/kernel', 2)]
    self.assertEqual(names_and_values, expected_names_and_values)

    # Check that values match regular JAX tree_flatten.
    self.assertEqual([x for _, x in names_and_values],
                     jax.tree_flatten(tree)[0])

  def test_tree_map_with_names(self):
    tree = {'a': 1, 'b': 2}
    mapped_tree = training_utils.tree_map_with_names(
        f=lambda x: -x, param_tree=tree, match_name_fn=lambda name: name == 'b')

    self.assertEqual(mapped_tree, {'a': 1, 'b': -2})


if __name__ == '__main__':
  absltest.main()
