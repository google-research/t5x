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

"""T5X Metrics.

Defines Metric objects and collections used by T5X models. These objects use the
CLU metrics library
"""

import dataclasses
from typing import MutableMapping, Optional, Union

from clu import metrics as clu_metrics
import flax  # Only used for flax.struct.dataclass.
from flax import struct
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
class MicrobatchAdjusted(clu_metrics.Metric):
  """Metric class that allows adjusting for values depending on microbatch size.

  Attributes:
    metric: the underlying Metric.
    per_step: whether metric is computed per step (True) or step per (False)
    num_microbatches: number of microbatches. 1 by default if no microbatches
      are used. Use replace_num_microbatches to set.

  Returns:
    A MicrobatchAdjusted Metric.
  """

  metric: clu_metrics.Metric
  per_step: bool = struct.field(pytree_node=False)
  num_microbatches: Optional[int] = None

  def merge(self, other: "MicrobatchAdjusted") -> "MicrobatchAdjusted":
    assert type(self.metric) is type(other.metric)
    assert self.per_step == other.per_step
    return type(self)(
        metric=self.metric.merge(other.metric),
        per_step=self.per_step,
        num_microbatches=self.num_microbatches)

  def replace_num_microbatches(self, num_microbatches) -> "MicrobatchAdjusted":
    if num_microbatches is None:
      return self
    return self.replace(num_microbatches=num_microbatches)

  def compute(self) -> Scalar:
    if self.num_microbatches is None:
      raise ValueError(
          "`num_microbatches` must be set by calling `replace_num_microbatches` before computing metric."
      )
    if self.per_step:
      return self.metric.compute() * self.num_microbatches
    else:
      return self.metric.compute() / self.num_microbatches


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
    return cls(numerator=numerator)

  def merge(self, other: "TimeRate") -> "TimeRate":
    assert_msg = "Merging with non-None durations is currently not supported."
    assert self.duration is None and other.duration is None, assert_msg
    return type(self)(numerator=self.numerator + other.numerator)

  def compute(self) -> Scalar:
    duration = super().compute()
    return self.numerator / duration

  def replace_duration(self, duration: Scalar) -> "Time":
    if not isinstance(self.numerator, np.ndarray):
      raise ValueError(
          "Expected numerator to be of type np.ndarray since method should be "
          "called outside of a compiled function. Got ", type(self.numerator))
    return super().replace_duration(duration)


def is_metric_obj(obj):
  return isinstance(obj, clu_metrics.Metric)


def is_time_metric(obj):
  return isinstance(obj, Time) or (isinstance(obj, MicrobatchAdjusted) and
                                   isinstance(obj.metric, Time))


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
      **{a.name: class_attr_shape(a) for a in dataclasses.fields(obj)})


def set_time_metrics_duration(metrics, duration):
  """Sets duration for TimeRate objects in metrics pytree."""

  def fn(o):
    if isinstance(o, Time):
      return o.replace_duration(duration)
    else:
      return o

  return jax.tree_map(fn, metrics, is_leaf=lambda obj: isinstance(obj, Time))


def set_microbatch_adjusted_metrics_microbatches(metrics, num_microbatches):
  """Sets num_microbatches for MicrobatchAdjusted objects in metrics pytree."""

  @jax.jit
  def fn(o):
    if isinstance(o, MicrobatchAdjusted):
      return o.replace_num_microbatches(num_microbatches)
    else:
      return o

  return jax.tree_map(fn, metrics, is_leaf=is_metric_obj)
