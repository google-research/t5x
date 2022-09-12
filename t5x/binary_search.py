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

"""Binary search over float32 bits.

Includes fast algorithms top-k masking and top-p masking on probability
distributions.
"""

from typing import Callable, Sequence

import jax
from jax import lax
from jax import numpy as jnp


def int32_bsearch(batch_shape: Sequence[int], predicate: Callable[[jnp.ndarray],
                                                                  jnp.ndarray]):
  """Batched binary search over int32 values.

  For each element of the batch, search for the largest int32 (closest to
  positive infinity) for which the predicate is False. If the predicate is
  always True, returns the minimum int32 value.

  Args:
    batch_shape: Shape of the search that we're batching over.
    predicate: the query we're searching for. For every batch element, this is
      required to be a monotonic function from int32 to bool. In other words,
      the predicate must return False for all numbers <= some threshold and then
      return True for all numbers > that threshold. The threshold may be
      different for different elements of the batch.

  Returns:
    For each element of the batch, the largest int32 for which the predicate
    returns False. Shape: batch_shape.
  """
  current_bits = jnp.zeros(batch_shape, dtype=jnp.int32)

  # bit 31 is special, because it compares in the opposite order of all other
  # bits
  midpoint = current_bits
  predicate_satisfied = predicate(midpoint)
  current_bits = current_bits | jnp.where(predicate_satisfied,
                                          jnp.int32(1 << 31), jnp.int32(0))
  del midpoint, predicate_satisfied

  def loop_body(i, current_bits):
    bit_index = 30 - i
    bit = jnp.int32(1 << bit_index)
    midpoint = current_bits | bit
    predicate_satisfied = predicate(midpoint)
    current_bits = current_bits | jnp.where(predicate_satisfied, jnp.int32(0),
                                            bit)
    return current_bits

  current_bits = lax.fori_loop(0, 31, loop_body, current_bits)
  return current_bits


def _monotonic_int32_to_float32_bit_pattern(x: int) -> int:
  """Converts an int32 to a float32 bit pattern with consistent ordering.

  This function is the unique function that is monotonic with respect to the
  floating point total order, see
  https://en.wikipedia.org/wiki/IEEE_754#Total-ordering_predicate. Note that
  this function returns an int32, not a float32. For the function that returns
  float32, see `monotonic_int32_to_float32`.

  Args:
    x: int bit pattern.

  Returns:
    Bit pattern of a float32 number.
  """
  non_sign_bits = jnp.int32((1 << 31) - 1)
  # See
  # https://stackoverflow.com/questions/20097380/iee-754-total-order-in-standard-c11
  # for the relationship between int32 order and f32 total order, including
  # the "xor trick".

  # Flip the sort order for numbers where the sign bit is set. On int32,
  # the bit pattern with sign bit set and all other bits clear is the most
  # negative bit pattern (it's int32::MIN), whereas on float32 it's the least
  # negative bit pattern (it's -0.0). Flipping all the non-sign bits makes the
  # int32 sort order consistent with the float32 sort order.
  x = x ^ jnp.where(x < 0, non_sign_bits, jnp.int32(0))
  return x


def _monotonic_int32_to_float32(x: int) -> float:
  """Converts an int32 to a float32 with consistent ordering.

  This function is the unique function that is monotonic with respect to the
  floating point total order, see
  https://en.wikipedia.org/wiki/IEEE_754#Total-ordering_predicate.

  Args:
    x: int bit pattern.

  Returns:
    float32 number with consistent ordering.
  """
  x = _monotonic_int32_to_float32_bit_pattern(x)
  return lax.bitcast_convert_type(x, jnp.float32)


def float32_bsearch(batch_shape, predicate):
  """Binary search on finite float32 numbers.

  For each element of the batch, this function searches for the largest finite
  non-NaN float32 for which the predicate is False.

  Args:
    batch_shape: Shape of the search that we're batching over.
    predicate: the query we're searching for. This is required to be monotonic
      with respect to the floating point order, i.e. it must be False for all
      numbers <= a threshold, and then True for all numbers > the threshold. The
      threshold may be different for different elements of the batch.

  Returns:
    For each element of the batch, the largest float32 for which the predicate
    returns False. Shape: f32[batch_shape].
  """
  exponent_bits = jnp.int32((1 << 31) - (1 << (31 - 8)))

  def int32_predicate(x):
    x = _monotonic_int32_to_float32_bit_pattern(x)
    is_finite = (x & exponent_bits) != exponent_bits

    # Non-finite numbers (infinity and NaN) are at the very extremes of the
    # int32 range, i.e. they include int32::MAX and int32::MIN, plus the numbers
    # adjacent to them. For the nonfinite numbers touching int32::MIN, we
    # arrange for them to return False from the predicate, and for the nonfinite
    # numbers touching int32::MAX, we arrange for them to return True from the
    # predicate. x>=0 is an easy way to achieve that.
    predicate_on_nonfinite = x >= 0
    x_float32 = lax.bitcast_convert_type(x, jnp.float32)
    return jnp.where(is_finite, predicate(x_float32), predicate_on_nonfinite)

  # We search over bit patterns, which requires bit shifting and ordering of bit
  # patterns. This is natively supported on int32 but not on float32.
  # Additionally, it's more common to reason about int32 bit arithmetic and
  # ordering than float32 bit arithmetic and ordering, so we do the core of our
  # search in int32. Additionally, this allows us to test the underlying binary
  # search on int32 values.
  #
  # The function _monotonic_int32_to_float32 encapsulates all of the knowledge
  # we need about float32 bit patterns.
  result = int32_bsearch(batch_shape, int32_predicate)
  return _monotonic_int32_to_float32(result)


def topk_mask(x: jnp.ndarray, k: int, replace_val: jnp.ndarray) -> jnp.ndarray:
  """Sets everything to replace_val, except the top k values per batch element.

  Sharding considerations: this function does 32 reductions over the vocab_size
  axis of the input array. To avoid excessive latency from these reductions, you
  should ensure that the vocab_size axis is unsharded on input to this function.
  Prefer to shard the batch axes instead.

  Scratchpad memory considerations: this function is most efficient if the
  entire input array can fit in a fast memory tier. To help ensure this, you may
  wish to split the batch axes into microbatches and the microbatches in a
  sequential loop.

  Args:
    x: Values before masking. [batch..., vocab_size]
    k: Number of masked values to return. In presence of ties, more than k
      values might be returned.
    replace_val: For the masked values of x, what to overwrite them with.

  Returns:
    masked version of x. [batch..., vocab_size]
  """
  batch_shape = tuple(list(x.shape)[:-1])  # [batch...]

  x_for_loop = x
  reduce_axis = x.ndim - 1
  if x.ndim > 1:
    # We're going to be doing 32 reductions over 'reduce_axis'. Generally,
    # reductions over the last dimension are the most expensive, because they
    # involve reducing across vector lanes, which is often not efficient. So
    # we transpose the reduce_axis to be the second-last dimension, to avoid
    # this inefficiency.
    #
    # Normaly the XLA compiler would automatically perform this optimization,
    # but it doesn't yet see through loops to do so. So we do it ourselves.
    x_for_loop = jnp.swapaxes(x_for_loop, -1, -2)
    reduce_axis = x.ndim - 2

  # x: [batch..., vocab_size, batch]
  def predicate(threshold):
    # threshold: [batch...]

    # Since we've negated, we now want a predicate that is True for small
    # numbers and False for large numbers. The result of the bsearch is the
    # smallest float32 for which the predicate is False.
    threshold = -threshold

    threshold = lax.expand_dims(threshold, (reduce_axis,))
    # threshold: [batch..., 1, last_batch]

    # count_ge: [batch...]
    count_gt = jnp.sum(x_for_loop > threshold, axis=reduce_axis)

    return count_gt >= k

  # cutoff: [batch...]
  cutoff = float32_bsearch(batch_shape, predicate)
  cutoff = -cutoff
  # cutoff: [batch..., 1]
  cutoff = lax.expand_dims(cutoff, (cutoff.ndim,))
  return jnp.where(x >= cutoff, x, jnp.full_like(x, replace_val))


def topp_mask(logits: jnp.ndarray, p: float,
              replace_val: jnp.ndarray) -> jnp.ndarray:
  """Applies top-p masking to logits.

  Masks logits down to the smallest set of choices, such that the total
  probability mass is >= p. Values in this set are left as they are. All other
  values are set with `replace_val`.

  Sharding considerations: this function does 33 reductions over the vocab_size
  axis of the input array. To avoid excessive latency from these reductions, you
  should ensure that the vocab_size axis is unsharded on input to this function.
  Prefer to shard the batch axes instead.

  Scratchpad memory considerations: this function is most efficient if the
  entire input array can fit in a fast memory tier. To help ensure this, you may
  wish to split the batch axes into microbatches and the microbatches in a
  sequential loop.

  Args:
    logits: Logits before masking. [batch..., vocab_size]
    p: Minimum probability mass requested.
    replace_val: For the masked values of logits, what to overwrite them with.

  Returns:
    masked version of x. [batch..., vocab_size]
  """
  batch_shape = tuple(list(logits.shape)[:-1])  # [batch...]

  probs = jax.nn.softmax(logits, axis=-1)

  probs_for_reduction = probs
  reduce_axis = probs_for_reduction.ndim - 1
  if probs_for_reduction.ndim > 1:
    # We're going to be doing 33 reductions over 'reduce_axis'. Generally,
    # reductions over the last dimension are the most expensive, because they
    # involve reducing across vector lanes, which is often not efficient. So
    # we transpose the reduce_axis to be the second-last dimension, to avoid
    # this inefficiency.
    probs_for_reduction = jnp.swapaxes(probs_for_reduction, -1, -2)
    reduce_axis = probs_for_reduction.ndim - 2

  # As we increase the threshold, the probability mass decreases, and the number
  # selected decreases.
  #
  # We want the largest threshold with the probability mass >= p. Binary search
  # searches for when the predicate is False, so we negate the output of the
  # predicate, i.e. probability mass < p.

  # probs_for_reduction: [batch..., vocab_size, batch]
  def predicate(threshold):
    # threshold: [batch...]
    threshold = lax.expand_dims(threshold, (reduce_axis,))
    # threshold: [batch..., 1, last_batch]

    # count_ge: [batch...]
    probability_mass = jnp.sum(
        jnp.where(probs_for_reduction >= threshold, probs_for_reduction, 0.0),
        axis=reduce_axis)

    return probability_mass < p

  # threshold: [batch...]
  threshold = float32_bsearch(batch_shape, predicate)
  # threshold: [batch..., 1]
  threshold = lax.expand_dims(threshold, (threshold.ndim,))
  return jnp.where(probs >= threshold, logits,
                   jnp.full_like(logits, replace_val))
