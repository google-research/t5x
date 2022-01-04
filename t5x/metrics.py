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

from typing import Any, MutableMapping, Optional

from clu import metrics
import flax  # Only used for flax.struct.dataclass.
import jax.numpy as jnp
import numpy as np

MetricsMap = MutableMapping[str, metrics.Metric]


def _assert_same_shape(metric_name: str, a: jnp.array, b: jnp.array):
  """Raises a `ValueError` if shapes of `a` and `b` don't match."""
  if a.shape != b.shape:
    raise ValueError(
        f"{metric_name} metric expected same shape: {a.shape} != {b.shape}")


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
class Sum(metrics.Metric):
  """Computes the sum of a scalar or a batch of tensors.

  See also documentation of `Metric`.
  """

  total: jnp.array

  @classmethod
  def from_model_output(cls, values: jnp.array, **_) -> metrics.Metric:
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
    _assert_same_shape("Sum", self.total, other.total)
    return type(self)(total=self.total + other.total)

  def compute(self) -> Any:
    return self.total


@flax.struct.dataclass
class WeightedAverage(metrics.Metric):
  """Computes the average of a scalar or a batch of tensors.

  Weights default to ones (unweighted). Supports the following dimensions of
  weights:

  - One-dimensional weights with the same leading dimension as the scalars, or,
  - Multi-dimensional weights with the exact same dimensions as the scalars.
    This allows the use of per-example weights for examples in a batch, as well
    as per-target weights for targets for examples in a batch.

  The result is always a scalar.

  See also documentation of `Metric`.
  """

  total: jnp.array
  count: jnp.array

  @classmethod
  def from_model_output(cls,
                        values: jnp.array,
                        weights: Optional[jnp.array] = None,
                        **_) -> metrics.Metric:
    """Initializes a WeightedAverage Metric from array (or singular) values.

    Args:
      values: array of values to average (or a single value).
      weights: weights to multiply values element-wise.

    Returns:
      A WeightedAverage object.
    """
    values = jnp.asarray(values)
    if values.ndim == 0:
      values = values[None]
    if weights is None:
      weights = jnp.ones_like(values)
    else:
      weights = jnp.asarray(weights)
    # Leading dimensions of weights and values must match.
    if weights.shape[0] != values.shape[0]:
      raise ValueError(
          f"Arg `weights` must have the same leading dimension as `values`. "
          f"Received weights of dimension {weights.shape} "
          f"and values of dimension {values.shape}.")
    # Broadcast weights to the same number of dimensions as values.
    remaining_dims_product = 1.
    if weights.ndim < values.ndim:
      remaining_dims_product = np.prod(values.shape[weights.ndim:])
      weights = jnp.expand_dims(
          weights, axis=tuple(jnp.arange(weights.ndim, values.ndim)))
    weights = weights.astype(np.float32)
    _check_param(weights, dtype=np.float32, ndim=values.ndim)
    return cls(
        total=(weights * values).sum(),
        count=(weights * remaining_dims_product).sum(),
    )

  def merge(self, other: "WeightedAverage") -> "WeightedAverage":
    _assert_same_shape("WeightedAverage", self.total, other.total)
    return type(self)(
        total=self.total + other.total,
        count=self.count + other.count,
    )

  def compute(self) -> Any:
    return self.total / self.count


def create_metrics_dict(float_metrics_dict):
  """Input: dict{str: float} | Output: dict{str: Metric}."""
  return {k: Sum.from_model_output(v) for k, v in float_metrics_dict.items()}
