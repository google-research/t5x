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

"""T5X Metrics.

Defines Metric objects and collections used by T5X models. These objects use the
CLU metrics library
"""

import dataclasses
from typing import MutableMapping, Optional, Union

from clu import metrics as clu_metrics
import flax  # Only used for flax.struct.dataclass.
import jax
import jax.numpy as jnp
import numpy as np

MetricsMap = MutableMapping[str, clu_metrics.Metric]
Scalar = Union[int, float, np.number, np.ndarray, jnp.ndarray]


def _check_param(value, *, ndim=None, dtype=jnp.float32):
  """Raises a `ValueError` if `value` does not match ndim/dtype.

  Args:
    value: Value to be tested.
    ndim: Expected dimensions.
    dtype: Expected dtype.

  Raises:
    A `ValueError` if `value` does not match `ndim` or `dtype`, or if `value`
    is not an instance of `jnp.ndarray`.
  """
  if ndim is not None and value.ndim != ndim:
    raise ValueError(f"Expected ndim={ndim}, got ndim={value.ndim}")
  if dtype is not None and value.dtype != dtype:
    raise ValueError(f"Expected dtype={dtype}, got dtype={value.dtype}")


@flax.struct.dataclass
class Sum(clu_metrics.Metric):
  """Computes the sum of a scalar or a batch of tensors.

  See also documentation of `Metric`.
  """

  total: Scalar

  @classmethod
  def from_model_output(cls, values: Scalar, **_) -> clu_metrics.Metric:
    """Initializes a Sum Metric from array (or singular) values.

    Args:
      values: array of values to sum (or a single value).

    Returns:
      A Sum object.
    """
    values = jnp.asarray(values)
    if values.ndim == 0:
      values = values[None]
    return cls(total=values.sum())

  def merge(self, other: "Sum") -> "Sum":
    return type(self)(total=self.total + other.total)

  def compute(self) -> Scalar:
    return self.total


@flax.struct.dataclass
class Step(clu_metrics.Metric):
  """Abstract class representing a per-step or step-per metric.

  Tracks number of steps. Must be set manually using replace_steps, since the
  use of microbatches may otherwise cause the computation to be incorrect.

  See also documentation of `Metric`.
  """
  steps: Optional[int] = 1

  def replace_steps(self, steps) -> "Step":
    return self.replace(steps=steps)

  def compute(self) -> Scalar:
    if self.steps is None:
      raise ValueError(
          "`steps` must be set by calling `replace_steps` before computing metric."
      )
    return self.steps


@flax.struct.dataclass
class AveragePerStep(Step):
  """Represents per-step average (total divided by number of steps).

  See also documentation of `Step`.
  """
  total: Optional[Scalar] = None

  @classmethod
  def from_model_output(cls,
                        values: Scalar,
                        steps: Optional[int] = 1,
                        **_) -> clu_metrics.Metric:
    """Initializes an AveragePerStep Metric from array (or singular) values.

    Args:
      values: array of values to sum (or a single value).
      steps: number of steps, defaults to 1.

    Returns:
      AveragePerStep object.
    """
    values = jnp.asarray(values)
    if values.ndim == 0:
      values = values[None]
    return cls(total=values.sum(), steps=steps)

  def merge(self, other: "AveragePerStep") -> "AveragePerStep":
    assert type(self) is type(other)
    return type(self)(
        total=self.total + other.total, steps=self.steps + other.steps)

  def compute(self) -> Scalar:
    steps = super().compute()
    if self.total is None:
      raise ValueError("`AveragePerStep` `total` cannot be None.")
    return self.total / steps


@flax.struct.dataclass
class Time(clu_metrics.Metric):
  """Computes the sum of a float-valued metric over a period of time.

  Duration (the denominator) must be set manually. This is because JAX does not
  properly support time functions inside compiled functions. Calling time.time()
  inside a compiled function results in the stored time being the compilation
  time, not the run time.

  See also documentation of `Metric`.
  """
  duration: Optional[Scalar] = None

  def merge(self, other: "Time") -> "Time":
    return self

  def compute(self) -> Scalar:
    if self.duration is None:
      raise ValueError(
          "`Time` `duration` must be set by calling `replace_duration` before computing."
      )
    return self.duration

  def replace_duration(self, duration: Scalar) -> "Time":
    """Replaces duration with the given value.

    Should be used outside a compiled function to set the duration of the
    metric.

    Args:
      duration: metric duration

    Returns:
      A new Time object.
    """
    return self.replace(duration=duration)


@flax.struct.dataclass
class TimeRate(Time):
  """Computes the sum of a float-valued metric over a period of time.

  Duration (the denominator) must be set using replace_duration. This is because
  JAX does not properly support time functions inside compiled functions.
  Calling time.time() inside a compiled function results in the stored time
  being the compilation time, not the run time.

  See also documentation of `Time` and `Metric`.
  """

  numerator: Optional[jnp.ndarray] = None

  @classmethod
  def from_model_output(cls, numerator: float, **_) -> clu_metrics.Metric:
    """Initializes a TimeRate Metric from a float value (the numerator).

    Args:
      numerator: a float (numerator of the metric)

    Returns:
      A TimeRate object.
    """
    return cls(numerator=numerator)  # pytype: disable=wrong-arg-types  # jax-ndarray

  def merge(self, other: "TimeRate") -> "TimeRate":
    assert_msg = "Merging with non-None durations is currently not supported."
    assert self.duration is None and other.duration is None, assert_msg
    return type(self)(numerator=self.numerator + other.numerator)

  def compute(self) -> Scalar:
    duration = super().compute()
    return self.numerator / duration


@flax.struct.dataclass
class StepsPerTime(Step, Time):
  """Represents a metric computed as number of steps per time.

  See also documentation of `Step`.
  """

  @classmethod
  def from_model_output(cls,
                        steps: Optional[int] = 1,
                        **_) -> clu_metrics.Metric:
    """Initializes an StepsPerTime Metric.

    Args:
      steps: number of steps, defaults to 1.

    Returns:
      StepsPerTime object.
    """
    return cls(steps=steps)

  def merge(self, other: "StepsPerTime") -> "StepsPerTime":
    assert type(self) is type(other)
    return type(self)(steps=self.steps + other.steps)

  def compute(self) -> Scalar:
    steps = Step.compute(self)
    duration = Time.compute(self)
    return steps / duration


def is_metric_obj(obj):
  return isinstance(obj, clu_metrics.Metric)


def is_time_metric(obj):
  return isinstance(obj, Time)


def create_metrics_dict(float_metrics_dict):
  """Input: dict{str: float} | Output: dict{str: Metric}."""
  return {k: Sum(v) for k, v in float_metrics_dict.items()}


def shape_obj_to_defined_obj(obj: clu_metrics.Metric):
  """Converts shapes in Metric to zero arrays.

  obj should be a Metric object subclass where each member variable is a
  ShapeDtypeStruct (from jax.eval_shape). A new object of the same class where
  each member variable is an array of zeros with the same shape and type as
  the corresponding variable defined by ShapeDtypeStruct.

  Args:
    obj: a clu.metrics.Metric object where each member variable is a
      ShapeDtypeStruct (from jax.eval_shape)

  Returns:
    A Metric object with class variables initialized as zero arrays.
  """

  def class_attr_shape(a):
    attr = getattr(obj, a.name)
    if isinstance(attr, clu_metrics.Metric):
      return shape_obj_to_defined_obj(attr)
    else:
      if hasattr(attr, "shape"):
        return jnp.zeros(shape=attr.shape, dtype=attr.dtype)
      else:
        return attr

  return obj.__class__(
      **{a.name: class_attr_shape(a) for a in dataclasses.fields(obj)})  # pytype: disable=wrong-arg-types  # re-none


def set_time_metrics_duration(metrics, duration):
  """Sets duration for TimeRate objects in metrics pytree."""

  def fn(o):
    if isinstance(o, Time):
      return o.replace_duration(duration)
    else:
      return o

  return jax.tree_map(fn, metrics, is_leaf=lambda obj: isinstance(obj, Time))


def set_step_metrics_num_steps(metrics, num_steps):
  """Sets steps for Step objects in metrics pytree."""

  def fn(o):
    if isinstance(o, Step):
      return o.replace_steps(num_steps)
    else:
      return o

  return jax.tree_map(fn, metrics, is_leaf=is_metric_obj)
