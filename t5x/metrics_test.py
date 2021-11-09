# Copyright 2021 The T5X Authors.
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

"""Tests for clu.metrics."""

from absl.testing import parameterized
import jax
import jax.numpy as jnp
from t5x import metrics
import tensorflow as tf


class MetricsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("0d_values_no_weights", 1, None, 1.),
      ("1d_values_no_weights", [1, 2, 3], None, 2.),
      ("1d_values_1d_weights", [1, 2, 3], [True, True, False], 1.5),
      ("2d_values_no_weights", [[1, 2], [2, 3], [3, 4]], None, 2.5),
      ("2d_values_1d_weights", [[1, 2], [2, 3], [3, 4]], [False, True, True
                                                         ], 3.),
      ("2d_values_2d_weights", [[1, 2], [2, 3], [3, 4]],
       [[False, True], [True, True], [True, True]], 2.8),
      ("3d_values_no_weights", [[[1, 2], [2, 3]], [[2, 1], [3, 4]],
                                [[3, 1], [4, 1]]], None, 2.25),
      ("3d_values_1d_weights", [[[1, 2], [2, 3]], [[2, 1], [3, 4]],
                                [[3, 1], [4, 1]]], [False, True, True], 2.375),
  )
  def test_weighted_average(self, values, weights, expected_result):
    self.assertAllClose(
        metrics.WeightedAverage.from_model_output(values,
                                                  weights=weights).compute(),
        expected_result)

  @parameterized.named_parameters(
      ("0d_values", 2., 2.), ("1d_values", [1, 2, 3], 6.),
      ("2d_values", [[1, 2], [2, 3], [3, 4]], 15.),
      ("3d_values", [[[1, 2], [2, 3]], [[2, 1], [3, 4]], [[3, 1], [4, 1]]], 27.)
  )
  def test_sum(self, values, expected_result):
    self.assertAllClose(
        metrics.Sum.from_model_output(values).compute(), expected_result)

  @parameterized.named_parameters(
      ("WeightedAverage", metrics.WeightedAverage),
      ("Sum", metrics.Sum),
  )
  def test_merge_asserts_shape(self, metric_cls):
    metric1 = metric_cls.from_model_output(jnp.arange(3.))
    metric2 = jax.tree_multimap(lambda *args: jnp.stack(args), metric1, metric1)
    with self.assertRaisesRegex(
        ValueError,
        r"^{} metric expected same shape".format(metric_cls.__name__)):
      metric1.merge(metric2)


if __name__ == "__main__":
  tf.test.main()
