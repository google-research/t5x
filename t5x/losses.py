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

"""Loss functions."""
import enum
from typing import Tuple, Mapping, Optional, Union

from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np


@jax.custom_vjp
def cross_entropy_with_logits(logits: jnp.ndarray, targets: jnp.ndarray,
                              z_loss: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes cross entropy loss with stable custom gradient.

  Computes a stabilized-gradient version of:
    -jnp.sum(targets * nn.log_softmax(logits), axis=-1)

  If z_loss > 0, then an auxiliary loss equal to z_loss*log(z)^2
  will be added to the cross entropy loss (z = softmax normalization constant).
  The two uses of z_loss are:
  1. To keep the logits from drifting too far from zero, which can cause
     unacceptable roundoff errors in bfloat16.
  2. To encourage the logits to be normalized log-probabilities.

  Args:
    logits: [batch, length, num_classes] float array.
    targets: categorical one-hot targets [batch, length, num_classes] float
      array.
    z_loss: coefficient for auxilliary z-loss loss term.

  Returns:
    tuple with the total loss and the z_loss, both
    float arrays with shape [batch, length].
  """
  logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  log_softmax = logits - logits_sum
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  # Add auxilliary z-loss term.
  log_z = jnp.squeeze(logits_sum, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return loss, total_z_loss


def _cross_entropy_with_logits_fwd(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    z_loss: float = 0.0
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray],
           Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
                 jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
  """Forward-mode of `cross_entropy_with_logits`."""
  max_logit = logits.max(axis=-1, keepdims=True)
  shifted = logits - max_logit
  exp_shifted = jnp.exp(shifted)
  sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
  log_softmax = shifted - jnp.log(sum_exp)
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  # Add auxilliary z-loss term.
  log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return (loss, total_z_loss), (logits, targets, z_loss, exp_shifted, sum_exp,  # pytype: disable=bad-return-type  # jax-ndarray
                                log_softmax, log_z)


def _cross_entropy_with_logits_bwd(
    res: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
               jnp.ndarray, jnp.ndarray], g: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Backward-mode of `cross_entropy_with_logits`."""
  g = g[0]  # Ignore z_loss component as that is only used for logging.
  logits, targets, z_loss, exp_shifted, sum_exp, log_softmax, log_z = res
  # z-loss term adds the (2 * z_loss * log_z) factor.
  deriv = (
      jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp -
      targets)
  g_logits = jnp.expand_dims(g, axis=-1) * deriv
  g_targets = -jnp.expand_dims(g, axis=-1) * log_softmax
  return (jnp.asarray(g_logits,
                      logits.dtype), jnp.asarray(g_targets, targets.dtype),
          jnp.array(0.0))  # sets z-loss coeff gradient to 0


cross_entropy_with_logits.defvjp(_cross_entropy_with_logits_fwd,
                                 _cross_entropy_with_logits_bwd)


def compute_weighted_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    loss_normalizing_factor: Optional[float] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length].
   label_smoothing: label smoothing constant, used to determine the on and off
     values.
   z_loss: coefficient for auxiliary z-loss loss term.
   loss_normalizing_factor: Constant to divide loss by. If not specified, loss
     will not be normalized. Intended for backward compatibility with T5-MTF
     training. Should not normally be used.

  Returns:
    Tuple of scalar loss, z_loss, and weight sum.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  vocab_size = logits.shape[-1]
  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(
      confidence * jnp.log(confidence) +
      (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
  soft_targets = common_utils.onehot(
      targets, vocab_size, on_value=confidence, off_value=low_confidence)
  total_loss, total_z_loss = cross_entropy_with_logits(
      logits, soft_targets, z_loss=z_loss)
  total_loss = total_loss - normalizing_constant

  weight_sum = np.prod(targets.shape)
  if weights is not None:
    total_loss = total_loss * weights
    total_z_loss = total_z_loss * weights
    weight_sum = jnp.sum(weights)

  # By default, we do not normalize loss based on anything.
  # We don't normalize based on batch size because the optimizers we use are
  # pretty much scale invariant, so this simplifies things.
  # We don't normalize based on number of non-padding tokens in order to treat
  # each token as equally important regardless of sequence length.
  if loss_normalizing_factor is not None:
    total_loss /= loss_normalizing_factor
    total_z_loss /= loss_normalizing_factor
  return jnp.sum(total_loss), jnp.sum(total_z_loss), weight_sum


@enum.unique
class SpecialLossNormalizingFactor(enum.Enum):
  """Specially calculated loss_normalizing_factors, that are not a constant.

  Attributes:
    NUM_REAL_TARGET_TOKENS: Whether to divide the loss by the number of real
      (non-padding) tokens in the current target batch. If
      'decoder_loss_weights' are specified, it will be the sum of the weights.
      Otherwise it will be the number of non-zero 'decoder_target_tokens'.
    NUM_TOTAL_TARGET_TOKENS: Whether to divide the loss by the total number of
      target tokens, i.e., batch_size * target_seq_length (including padding).
    AVERAGE_PER_SEQUENCE: This will first compute the per-sequence loss
      (averaged over the number of real target tokens in the sequence), and then
      compute the average of that over the sequences. This can be preferable to
      NUM_REAL_TARGET_TOKENS for finetuning, because it will weigh all examples
      equally, regardless of sequence length (which can be especially important
      for multi-task finetuning).
  """
  NUM_REAL_TARGET_TOKENS = 1
  NUM_TOTAL_TARGET_TOKENS = 2
  AVERAGE_PER_SEQUENCE = 3


def convert_special_loss_normalizing_factor_to_enum(
    x: str) -> SpecialLossNormalizingFactor:
  """Converts stringified version of LNF to an enum.

  This is useful because gin dynamic registration does not (currently)
  have support for enum.

  Args:
    x: stringified version of SpecialLossNormalizingFactor enum.

  Returns:
    SpecialLossNormalizingFactor enum instance.
  """
  x = x.upper()
  if x == 'NUM_REAL_TARGET_TOKENS':
    return SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS
  if x == 'NUM_TOTAL_TARGET_TOKENS':
    return SpecialLossNormalizingFactor.NUM_TOTAL_TARGET_TOKENS
  if x == 'AVERAGE_PER_SEQUENCE':
    return SpecialLossNormalizingFactor.AVERAGE_PER_SEQUENCE
  raise ValueError(
      'Could not convert string \"%s\" to SpecialLossNormalizingFactor' % x)


@jax.vmap
def _sum_weights_per_segment(positions: jnp.ndarray, segment_ids: jnp.ndarray,
                             weights: jnp.ndarray) -> jnp.ndarray:
  """Sums weights per packed segment to produce a normalizing vector."""

  # NB: Assumes padding only occurs at the end of a sequence.

  def _repeat_last_nonnegative(xs, reverse=False):

    def fn(prev, x):
      y = jnp.where(x == 0, prev, x)
      return y, y

    return jax.lax.scan(fn, jnp.zeros_like(xs[0]), xs, reverse=reverse)[1]

  # Compute final positions per sequence.
  start_positions = positions == 0
  final_positions = jnp.concatenate([start_positions[1:], jnp.ones(1)])
  # Clear padded positions.
  final_positions *= segment_ids != 0
  # Compute cumulative weights, clearing all but final position per sequence.
  final_cumulative_weights = final_positions * jnp.cumsum(weights)
  # Subtract sequences' final weights from cumulative weights of following ones.
  final_total_weights = jnp.concatenate([
      final_cumulative_weights[0:1],
      jnp.diff(_repeat_last_nonnegative(final_cumulative_weights))
  ])
  # Copy final sequence weight to all positions in sequence.
  normalizer = _repeat_last_nonnegative(final_total_weights, reverse=True)
  return normalizer


def get_loss_normalizing_factor_and_weights(
    loss_normalizing_factor: Optional[Union[float, int, str,
                                            SpecialLossNormalizingFactor]],
    batch: Mapping[str, jnp.ndarray]):
  """Get the float loss_normalizing_factor and loss weights.

  If loss_normalizing_factor is float or None, this will simply return the
  input loss_normalizing_factor and batch.

  If loss_normalizing_factor is a SpecialLossNormalizingFactor, it will
  return a float loss_normalizing_factor and loss weights corresponding to
  the special LNF. See SpecialLossNormalizingFactor for more details.

  Args:
    loss_normalizing_factor: The input LNF, which may be a float, None, or
      SpecialLossNormalizingFactor (or a stringified SLNF).
    batch: Input data batch.

  Returns:
    Tuple of (output_loss_normalizing_factor, loss_weights).
      'output_loss_normalizing_factor' is a scalar float (Python float
      or jnp float).
      'loss_weights' is the per token loss weight JNP array.
  """

  loss_weights = batch.get('decoder_loss_weights', None)
  if (loss_normalizing_factor is None or
      not isinstance(loss_normalizing_factor,
                     (str, SpecialLossNormalizingFactor))):
    return (loss_normalizing_factor, loss_weights)

  if isinstance(loss_normalizing_factor, str):
    loss_normalizing_factor = convert_special_loss_normalizing_factor_to_enum(
        loss_normalizing_factor)

  # If `loss_weights` are not provided, we assume that the padding id is 0 and
  # that non-padding tokens in the decoder all correspond to the positions
  # where loss should be taken. If more fine-grained behavior (e.g., taking
  # loss on subset of 'decoder_target_tokens') is desired, provide
  # `loss_weights` that account for this.
  if loss_weights is None:
    loss_weights = jnp.asarray(batch['decoder_target_tokens'] > 0, jnp.float32)

  output_normalizing_factor = None
  if (loss_normalizing_factor ==
      SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS):
    output_normalizing_factor = jnp.sum(loss_weights)
  elif (loss_normalizing_factor ==
        SpecialLossNormalizingFactor.NUM_TOTAL_TARGET_TOKENS):
    output_normalizing_factor = np.prod(batch['decoder_target_tokens'].shape)
  elif (loss_normalizing_factor ==
        SpecialLossNormalizingFactor.AVERAGE_PER_SEQUENCE):
    if 'decoder_segment_ids' in batch:  # is packed
      norm_vec = _sum_weights_per_segment(batch['decoder_positions'],
                                          batch['decoder_segment_ids'],
                                          loss_weights)
    else:
      norm_vec = jnp.sum(loss_weights, axis=-1, keepdims=True)
    # Handle divide-by-zero.
    loss_weights = jnp.nan_to_num(
        loss_weights / norm_vec, nan=0, posinf=0, neginf=0)
    output_normalizing_factor = jnp.sum(loss_weights)
  else:
    raise ValueError('Unsupported value of loss_normalizing_factor: %s' %
                     str(loss_normalizing_factor))

  return (output_normalizing_factor, loss_weights)
