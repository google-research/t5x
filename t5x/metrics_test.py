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

"""Tests for clu.metrics."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from t5x import metrics


class MetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("0d_values", 2., 2.), ("1d_values", [1, 2, 3], 6.),
      ("2d_values", [[1, 2], [2, 3], [3, 4]], 15.),
      ("3d_values", [[[1, 2], [2, 3]], [[2, 1], [3, 4]], [[3, 1], [4, 1]]], 27.)
  )
  def test_sum(self, values, expected_result):
    self.assertAlmostEqual(
        metrics.Sum.from_model_output(values).compute(), expected_result)

  def test_time_rate(self):
    value = np.array([3.])
    duration = 2.
    metric = metrics.TimeRate.from_model_output(value).replace_duration(
        duration)
    self.assertAlmostEqual(metric.compute(), value / duration)

  def test_time_rate_unset_duration(self):
    value = jnp.array([3.])
    metric = metrics.TimeRate.from_model_output(value)
    with self.assertRaises(ValueError):
      metric.compute()

  def test_time_rate_sets_duration_inside_jitted_fn(self):

    @jax.jit
    def fn():
      value = jnp.array([3.])
      duration = 2.
      metric = metrics.TimeRate.from_model_output(value).replace_duration(
          duration)
      return metric

    with self.assertRaises(ValueError):
      fn()

  def test_time(self):
    duration = 2.
    metric = metrics.Time().replace_duration(duration)
    self.assertAlmostEqual(metric.compute(), duration)

  def test_time_unset_duration(self):
    metric = metrics.Time()
    with self.assertRaises(ValueError):
      metric.compute()

  @parameterized.named_parameters(
      ("0d_values", 2., 2.),
      ("1d_values", [1, 2, 3], 6.),
  )
  def test_average_per_step(self, values, expected_result):
    a = metrics.AveragePerStep.from_model_output(values)
    m = metrics.set_step_metrics_num_steps({"a": a}, 1)
    self.assertAlmostEqual(m["a"].compute(), expected_result)

    steps = 5
    b = metrics.AveragePerStep.from_model_output(values, steps=steps)
    m = metrics.set_step_metrics_num_steps({"b": b}, steps)
    self.assertAlmostEqual(m["b"].compute(), expected_result / steps)

  def test_steps_per_time(self):
    steps = 8.
    duration = 2.
    metric = metrics.StepsPerTime.from_model_output(
        steps=steps).replace_duration(duration)
    metrics_dict = metrics.set_step_metrics_num_steps({"metric": metric}, steps)
    self.assertAlmostEqual(metrics_dict["metric"].compute(), steps / duration)


if __name__ == "__main__":
  absltest.main()
