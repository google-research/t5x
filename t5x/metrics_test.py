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
import numpy as np
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
      ("0d_values_no_weights", 1, 2, None, 0.5),
      ("1d_values_no_weights", [1, 2, 3], 1., None, 6.),
      ("1d_values_1d_weights", [1, 2, 3], 0.5, [True, True, False], 6.),
      ("2d_values_no_weights", [[1, 2], [2, 3], [3, 4]], 0.5, None, 30.),
      ("2d_values_1d_weights", [[1, 2], [2, 3], [3, 4]
                               ], 3., [False, True, True], 4.),
      ("2d_values_2d_weights", [[1, 2], [2, 3], [3, 4]], 2.,
       [[False, True], [True, True], [True, True]], 7.),
      ("3d_values_no_weights", [[[1, 2], [2, 3]], [[2, 1], [3, 4]],
                                [[3, 1], [4, 1]]], 3, None, 9),
      ("3d_values_1d_weights", [[[1, 2], [2, 3]], [[2, 1], [3, 4]],
                                [[3, 1], [4, 1]]], 1., [False, True, True], 19),
  )
  def test_weighted_average_rate(self, values, count, weights, expected_result):
    self.assertAllClose(
        metrics.WeightedAverageRate.from_model_output(
            values, count=count, weights=weights).compute(), expected_result)

  @parameterized.named_parameters(
      ("0d_values", 2., 2.), ("1d_values", [1, 2, 3], 6.),
      ("2d_values", [[1, 2], [2, 3], [3, 4]], 15.),
      ("3d_values", [[[1, 2], [2, 3]], [[2, 1], [3, 4]], [[3, 1], [4, 1]]], 27.)
  )
  def test_sum(self, values, expected_result):
    self.assertAllClose(
        metrics.Sum.from_model_output(values).compute(), expected_result)

  @parameterized.named_parameters(
      ("2d_values_no_weights", [[5, 10], [0.1, 0.9], [1, 2]], [1, 1, 0
                                                              ], None, 2. / 3),
      ("2d_values_1d_weights", [[1, 2], [20, 10], [0.9, 0.1]], [1, 1, 1
                                                               ], [0, 1, 1], 0),
      ("3d_values_no_weights", [[[0.9, 0.1], [0.8, 0.2]], [[7, 3], [0, 10]],
                                [[1, 9], [5, 20]]], [[0, 0], [1, 0], [1, 1]
                                                    ], None, 2. / 3),
      ("3d_values_1d_weights", [[[0.9, 0.1], [0.8, 0.2]], [[7, 3], [0, 10]],
                                [[1, 9], [5, 20]]], [[0, 0], [1, 0], [1, 1]
                                                    ], [0, 1, 1], 0.5),
  )
  def test_weighted_accuracy(self, logits, labels, weights, expected_result):
    self.assertAllClose(
        metrics.WeightedAccuracy.from_model_output(
            logits=logits, labels=labels, weights=weights).compute(),
        expected_result)

  def test_time_rate(self):
    value = np.array([3.])
    duration = 2.
    metric = metrics.TimeRate.from_model_output(value).replace_duration(
        duration)
    self.assertAllClose(metric.compute(), value / duration)


if __name__ == "__main__":
  tf.test.main()
