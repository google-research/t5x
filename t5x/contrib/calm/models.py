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

"""Models with CALM early exit functionality."""

import copy
import functools
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple, Union

import clu.metrics as clu_metrics
import flax
from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import seqio
from t5x import decoding
from t5x import losses
from t5x import metrics as metrics_lib
from t5x import models
from t5x import optimizers
from t5x.contrib.calm import decoding as calm_decoding
import tensorflow as tf
import typing_extensions


# Remove _ShardedDeviceArray when users of t5x have their types updated
_ShardedDeviceArray = Any
Array = Union[np.ndarray, jnp.ndarray, _ShardedDeviceArray, tf.Tensor]
MetricsMap = metrics_lib.MetricsMap
PyTree = Any




class TokensIdsToLogitsCallable(typing_extensions.Protocol):
  """Token ids to logits mapping call signature."""

  def __call__(
      self, decoding_state: calm_decoding.DecodingState
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """Performs forward pass to convert token ids to logits.

    Args:
      decoding_state: Current decoding state, including current token ids and
        cache.

    Returns:
      logits: logits with a shape [batch_size, vocab_size].
      cache: An updated cache.
      confidences: Float array of shape [batch_size, max_decode_len] with the
      confidence values measured at the exit layer.
      layres: Int array of shape [batch_size, max_decode_len] with the exited
        layer per token.
    """
    ...


class DecodeFnCallable(typing_extensions.Protocol):
  """Decoding function call signature."""

  def __call__(
      self, *, inputs: jnp.ndarray, cache: Mapping[str, jnp.ndarray],
      tokens_to_logits: TokensIdsToLogitsCallable, eos_id: int,
      num_decodes: int, decode_rng: Optional[jax.random.KeyArray],
      cache_offset: int, **kwargs
  ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Decoding function interface.

    Args:
      inputs: [batch_size, max_decode_len] int32 sequence of tokens, with non-0
        prefix tokens to be used as a forced prompt.
      cache: flax attention cache.
      tokens_to_logits: fast autoregressive decoder function taking single token
        slices and cache and returning next-token logits and updated cache.
      eos_id: end-of-sentence token for target vocabulary.
      num_decodes: number of decoded sequences to be returned.
      decode_rng: an optional JAX PRNG Key for stochastic sampling routines.
      cache_offset: axis offset for cache, arising from scanned layers.
      **kwargs: an optional kwargs. One common usecase of this is passing
        decoding parameters at the callsite.

    Returns:
      decodes: Array of sequences: [batch_size, num_decodes, max_decode_len].
        The `num_decodes` dimension is expected to be sorted by the `scores`,
        i.e., `decodes[:, -1, :] has the highest scores among `num_decodes`
        decoded sequences.
      scores: Array of log likelihood scores: [batch_size, num_decodes]
      confidences: Float array of shape [batch_size, max_decode_len] with the
      confidence values measured at the exit layer.
      layres: Int array of shape [batch_size, max_decode_len] with the exited
      layer per token.
    """
    ...


class EncoderDecoderModel(models.EncoderDecoderModel):
  """Wrapper class for the models.Transformer nn.module.

  Incorporates CALM early exit functionalities.
  """

  def __init__(
      self,
      module: nn.Module,
      input_vocabulary: seqio.Vocabulary,
      output_vocabulary: seqio.Vocabulary,
      optimizer_def: optimizers.OptimizerDefType,
      decode_fn: DecodeFnCallable = calm_decoding.temperature_sample,
      feature_converter_cls: Optional[Callable[...,
                                               seqio.FeatureConverter]] = None,
      label_smoothing: float = 0.0,
      z_loss: float = 0.0,
      loss_normalizing_factor: Optional[float] = None,

      apply_early_inference: bool = False,
      decoder_layers: int = 12,
      conf_threshold: float = 1.0,
      min_exit: int = 0,
      first_exit: int = 0,
      exit_interval: int = 1,
      aggregation_weights: int = 1,
      oracle_tok_consistency: bool = False,
      oracle_cache: bool = False,
      oracle_tok_noisy_cache: bool = False,
      conf_method: str = 'softmax_max',
      train_meta_cls: bool = False,
      geomlike_loss: bool = False,
      position_adjusted_threshold: bool = False,
      position_temp: int = 4,
  ):
    self.apply_early_inference = apply_early_inference
    self.decoder_layers = decoder_layers
    self.conf_threshold = conf_threshold
    self.min_exit = min_exit
    self.first_exit = first_exit
    self.exit_interval = exit_interval
    self.aggregation_weights = aggregation_weights
    self.oracle_tok_consistency = oracle_tok_consistency
    self.oracle_cache = oracle_cache
    self.oracle_tok_noisy_cache = oracle_tok_noisy_cache
    self.conf_method = conf_method
    self.train_meta_cls = train_meta_cls
    self.geomlike_loss = geomlike_loss
    self.position_adjusted_threshold = position_adjusted_threshold
    self.position_temp = position_temp
    super().__init__(
        module=module,
        input_vocabulary=input_vocabulary,
        output_vocabulary=output_vocabulary,
        optimizer_def=optimizer_def,
        decode_fn=decode_fn,
        feature_converter_cls=feature_converter_cls,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
    )

  def get_pred_confidence(self,  # pytype: disable=annotation-type-mismatch  # jax-ndarray
                          logits: jnp.ndarray = None,
                          prev_state: jnp.ndarray = None,
                          new_state: jnp.ndarray = None,
                          meta_score: jnp.ndarray = None) -> jnp.ndarray:
    """Computes the of decoder in its current prediction.

    The confidence function is determined by self.conf_method.

    Args:
      logits: Array with last dimension holding the logits over
        the output vocabulary.
      prev_state: Hidden state from previous layer.
      new_state: Hidden state of current layer.
      meta_score: The confidence score of an early-exit classifier.

    Returns:
      confidence: Per example confidence scores.
    """
    if self.conf_method == 'softmax_diff':
      # Computes confidence by taking the difference between the top two softmax
      # scores. This implementation can be slow due to sorting all logits.
      assert logits is not None
      logits_sorted = jnp.sort(logits, axis=-1)[..., ::-1]  # sort descending.
      sorted_probs = nn.softmax(logits_sorted, axis=-1)
      return sorted_probs[..., 0] - sorted_probs[..., 1]

    if self.conf_method == 'softmax_diff_approx':
      # A faster softmax approximate difference implementation.
      assert logits is not None
      probs = nn.softmax(logits, axis=-1)
      top_2 = jax.lax.approx_max_k(probs, k=2)[0]
      return top_2[..., 0] - top_2[..., 1]

    if self.conf_method == 'softmax_max':
      # Computes confidence by taking the maximum softmax value.
      assert logits is not None
      return nn.softmax(logits, axis=-1).max(axis=-1)

    elif self.conf_method == 'state':
      # Computes the confidence by the cosine similarity between the current
      # hidden state and the previous one.
      assert prev_state is not None and new_state is not None
      conf = jnp.inner(prev_state, new_state) / (
          jnp.linalg.norm(prev_state) * jnp.linalg.norm(new_state))
      return conf.squeeze()

    elif self.conf_method == 'meta':
      # Using scores from early-exit classifier (already given in input).
      assert meta_score is not None
      return meta_score

    else:
      raise NotImplementedError(
          f'Confidence method {self.conf_method} is not implemented.')

  def loss_fn_meta_cls(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Loss function for the meta early exit classifier (meta_cls).

    Should also be used with 'return_all_logits' option for the decoder. Can be
    used for a second training step where we freeze the rest of the model,
    (non-meta_cls parameters).

    Args:
      params: model parameters.
      batch: a batch of inputs.
      dropout_rng: rng to use for dropout, or None for deterministic mode.

    Returns:
      loss: the loss computed for the given inputs and parameters.
      metrics: a mapping of metrics computed for this batch.
    """

    loss_normalizing_factor: Optional[Union[
        float, int, str, losses.SpecialLossNormalizingFactor]]
    (loss_normalizing_factor,
     weights) = losses.get_loss_normalizing_factor_and_weights(
         self._loss_normalizing_factor, batch)

    all_logits = self._compute_logits(params, batch, dropout_rng)
    assert isinstance(all_logits,
                      tuple), 'Verify that meta_cls was initialized in decoder.'
    all_meta_logits = all_logits[1]
    all_logits = all_logits[0]

    # Create meta labels based on consistency of intermediate prediction with
    # the top prediction.
    predictions = all_logits.argmax(-1)
    top_pred = predictions[-1]
    all_meta_labels = jnp.array(top_pred == predictions, dtype=jnp.int32)

    # Aggregate meta loss across layers.
    all_loss, all_total_z_loss, = [], []
    for meta_logits, meta_labels in zip(all_meta_logits[:-1],
                                        all_meta_labels[:-1]):
      # Balance across the positive/ negative labels.
      balanced_weights = weights.copy().astype(float)

      pos_num = (meta_labels * weights == 1).sum()
      neg_num = ((1 - meta_labels) * weights == 1).sum()

      pos_weight = 1 - (pos_num / (pos_num + neg_num))
      neg_weight = 1 - (neg_num / (pos_num + neg_num))
      balanced_weights = weights * meta_labels * pos_weight + weights * (
          1 - meta_labels) * neg_weight

      # Compute layer loss.
      all_loss_i, all_total_z_loss_i, _ = losses.compute_weighted_cross_entropy(
          meta_logits,
          targets=meta_labels,
          label_smoothing=self._label_smoothing,
          z_loss=self._z_loss,
          weights=balanced_weights,
          loss_normalizing_factor=loss_normalizing_factor)
      all_loss.append(all_loss_i)
      all_total_z_loss.append(all_total_z_loss_i)

    loss = jnp.average(jnp.array(all_loss), 0)
    total_z_loss = jnp.average(jnp.array(all_total_z_loss), 0)

    metrics = self._compute_metrics(
        logits=all_logits[-1],
        targets=batch['decoder_target_tokens'],
        mask=weights,
        loss=loss,
        z_loss=total_z_loss)

    # Meta metrics.
    for i, (meta_logits, meta_labels) in enumerate(
        zip(all_meta_logits[:-1], all_meta_labels[:-1])):
      meta_metrics = {
          f'meta_accuracy/layer_{i}':
              clu_metrics.Accuracy.from_model_output(
                  logits=meta_logits,
                  labels=meta_labels.astype(jnp.int32),
                  mask=weights),
          f'meta_loss/layer_{i}':
              metrics_lib.AveragePerStep(total=all_loss[i]),
      }
      metrics.update(meta_metrics)

    return loss, metrics

  def loss_fn_meta_cls_geom_like(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Geometric-like loss function for the meta early exit classifier.

    This version follows the geometric-like (consistency-based) of
    https://arxiv.org/pdf/1910.10073.pdf
    that optimizes the meta cls as a sequence of decisions instead of
    independent predcitions over layers.

    Should also be used with 'return_all_logits' option for the decoder. Can be
    used for a second training step where we freeze the rest of the model,
    (non-meta_cls parameters).


    Args:
      params: model parameters.
      batch: a batch of inputs.
      dropout_rng: rng to use for dropout, or None for deterministic mode.

    Returns:
      loss: the loss computed for the given inputs and parameters.
      metrics: a mapping of metrics computed for this batch.
    """

    loss_normalizing_factor: Optional[Union[
        float, int, str, losses.SpecialLossNormalizingFactor]]
    (loss_normalizing_factor,
     weights) = losses.get_loss_normalizing_factor_and_weights(
         self._loss_normalizing_factor, batch)

    all_logits = self._compute_logits(params, batch, dropout_rng)
    assert isinstance(all_logits,
                      tuple), 'Verify that meta_cls was initialized in decoder.'
    all_meta_logits = all_logits[1]
    all_logits = all_logits[0]

    # Create meta labels based on consistency of intermediate prediction with
    # the top prediction.
    predictions = all_logits.argmax(-1)
    top_pred = predictions[-1]
    all_meta_labels = jnp.array(top_pred == predictions, dtype=jnp.int32)

    # Here, this is treated as an L-way classification task (L=decoder layers).
    all_meta_labels_multiclass = all_meta_labels.argmax(0)

    # Geometric-like aggregation.
    all_meta_scores = nn.log_softmax(all_meta_logits, axis=-1)
    all_meta_scores_pos = all_meta_scores[..., 1]
    all_meta_scores_neg = all_meta_scores[..., 0]
    non_stop_probs = all_meta_scores_neg.cumsum(0) - all_meta_scores_neg
    geom_like_probs = non_stop_probs + all_meta_scores_pos
    geom_like_probs = jnp.moveaxis(geom_like_probs, 0, -1)

    loss, total_z_loss, _ = losses.compute_weighted_cross_entropy(
        geom_like_probs,
        targets=all_meta_labels_multiclass,
        label_smoothing=self._label_smoothing,
        z_loss=self._z_loss,
        weights=weights,
        loss_normalizing_factor=loss_normalizing_factor)

    total_z_loss = 0.0  # hardcoded

    metrics = self._compute_metrics(  # pytype: disable=wrong-arg-types  # jax-ndarray
        logits=all_logits[-1],
        targets=batch['decoder_target_tokens'],
        mask=weights,
        loss=loss,
        z_loss=total_z_loss)

    # Meta metrics.
    for i, (meta_logits, meta_labels) in enumerate(
        zip(all_meta_logits[:-1], all_meta_labels[:-1])):
      meta_metrics = {
          f'meta_accuracy/layer_{i}':
              clu_metrics.Accuracy.from_model_output(
                  logits=meta_logits,
                  labels=meta_labels.astype(jnp.int32),
                  mask=weights),
      }
      meta_metrics.update({
          'meta_accuracy/multiclass':
              clu_metrics.Accuracy.from_model_output(
                  logits=geom_like_probs,
                  labels=all_meta_labels_multiclass.astype(jnp.int32),
                  mask=weights),
      })
      metrics.update(meta_metrics)

    return loss, metrics

  def loss_fn(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Loss function for anytime predictions across model layers.

    Should be used with `return_all_logits` option for the decoder. Per-layer
    loss is aggregated with a weighted sum, according `aggregation_weights`.

    if `train_meta_cls` is True, will call `loss_fn_meta_cls` instead.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      dropout_rng: rng to use for dropout, or None for deterministic mode.

    Returns:
      loss: the loss computed for the given inputs and parameters.
      metrics: a mapping of metrics computed for this batch.
    """

    if self.train_meta_cls:
      if self.geomlike_loss:
        return self.loss_fn_meta_cls_geom_like(params, batch, dropout_rng)
      else:
        return self.loss_fn_meta_cls(params, batch, dropout_rng)

    loss_normalizing_factor: Optional[Union[
        float, int, str, losses.SpecialLossNormalizingFactor]]
    (loss_normalizing_factor,
     weights) = losses.get_loss_normalizing_factor_and_weights(
         self._loss_normalizing_factor, batch)

    all_logits = self._compute_logits(params, batch, dropout_rng)
    all_loss, all_total_z_loss = [], []
    for logits in all_logits:
      all_loss_i, all_total_z_loss_i, _ = losses.compute_weighted_cross_entropy(
          logits,
          targets=batch['decoder_target_tokens'],
          weights=weights,
          label_smoothing=self._label_smoothing,
          z_loss=self._z_loss,
          loss_normalizing_factor=loss_normalizing_factor)
      all_loss.append(all_loss_i)
      all_total_z_loss.append(all_total_z_loss_i)

    if self.aggregation_weights == -1:
      # Geometric series with a=1, r=2.
      avg_weights = jnp.geomspace(1, 2**(len(all_loss) - 1), len(all_loss))
    elif self.aggregation_weights == 0:
      avg_weights = jnp.ones(len(all_loss))
    else:
      avg_weights = jnp.arange(
          1,
          self.aggregation_weights * len(all_loss) + 1,
          step=self.aggregation_weights)
    loss = jnp.average(jnp.array(all_loss), 0, avg_weights)
    total_z_loss = jnp.average(jnp.array(all_total_z_loss), 0, avg_weights)

    # Based on last logits.
    metrics = self._compute_metrics(
        logits=all_logits[-1],
        targets=batch['decoder_target_tokens'],
        mask=weights,
        loss=loss,
        z_loss=total_z_loss)

    # Per layer metrics.
    for i, logits in enumerate(all_logits[:-1]):
      meta_metrics = {
          f'accuracy_per_layer/layer_{i}':
              clu_metrics.Accuracy.from_model_output(
                  logits=logits,
                  labels=batch['decoder_target_tokens'],
                  mask=weights),
          f'loss_per_layer/layer_{i}':
              metrics_lib.AveragePerStep(total=all_loss[i]),
      }
      metrics.update(meta_metrics)

    return loss, metrics

  def _compute_logits_from_slice_early_exit(
      self,
      decoding_state: calm_decoding.DecodingState,
      params: PyTree,
      encoded_inputs: jnp.ndarray,
      raw_inputs: jnp.ndarray,
      max_decode_length: int,
      conf_threshold: float,
      pos: int = 0,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """Token slice to logits from decoder model with early exit mechanism."""

    num_layers = self.decoder_layers
    flat_ids: jnp.ndarray = decoding_state.cur_token
    flat_cache: Mapping[str, jnp.ndarray] = decoding_state.cache

    if self.oracle_tok_consistency or self.oracle_tok_noisy_cache:
      # Compute the prediction of the full model for reference for the oracle.
      oracle_flat_cache = copy.deepcopy(flat_cache)
      oracle_flat_logits, _ = self.module.apply(
          {
              'params': params,
              'cache': oracle_flat_cache
          },
          encoded_inputs,
          raw_inputs,  # only needed for encoder padding mask
          flat_ids,
          flat_ids,
          enable_dropout=False,
          decode=True,
          max_decode_length=max_decode_length,
          start_idx=0,
          end_idx=None,
          return_prelogits=False,
          mutable=['cache'],
          method=self.module.decode)
      oracle_tok_pred = oracle_flat_logits.argmax()

    # Get the computation intervals (per layers) of the decoder between exits.
    keep_inds = list(range(self.first_exit + 1, num_layers, self.exit_interval))
    comp_intervals = [(0, self.first_exit + 1)] + [
        (i, j) for i, j in zip(keep_inds, keep_inds[1:] + [num_layers])
    ]

    # First run the decoder but only up to the first exit.
    decoder_hidden, new_vars = self.module.apply(
        {
            'params': params,
            'cache': flat_cache
        },
        encoded_inputs,
        raw_inputs,  # only needed for encoder padding mask
        flat_ids,
        flat_ids,
        enable_dropout=False,
        decode=True,
        max_decode_length=max_decode_length,
        start_idx=0,
        end_idx=comp_intervals[0][1],
        return_prelogits=True,
        mutable=['cache'],
        method=self.module.decode)

    # If using meta_cls.
    if isinstance(decoder_hidden, tuple):
      meta_score = nn.softmax(decoder_hidden[1], axis=-1)[..., 1]
      decoder_hidden = decoder_hidden[0]
    else:
      meta_score = None

    new_flat_cache = new_vars['cache']
    if 'softmax' in self.conf_method:
      flat_logits = self.module.apply({
          'params': params,
          'cache': flat_cache
      },
                                      decoder_hidden,
                                      logit_mask=None,
                                      enable_dropout=False,
                                      method=self.module.compute_logits)
    else:
      flat_logits = None

    if self.conf_method == 'state':
      # Always skip the first 'exit' since previous state is missing.
      conf = 0
    else:
      conf = self.get_pred_confidence(logits=flat_logits, meta_score=meta_score)

    # Used to enable a positional argument (decoder_embedded_input) for switch.
    def prt(a,
            b,
            c,
            d,
            e,
            f,
            start_idx=0,
            end_idx=None,
            only_propagate_state=False,
            **kwargs):  # pylint: disable=unused-argument
      return self.module.apply(
          a,
          b,
          c,
          d,
          e,
          enable_dropout=False,
          decode=True,
          max_decode_length=max_decode_length,
          start_idx=start_idx,
          end_idx=end_idx,
          decoder_embedded_input=f,
          return_prelogits=True,
          only_propagate_state=only_propagate_state,
          mutable=['cache'],
          method=self.module.decode)

    # Segments of the model between exits, passed to lax.switch.
    branches = [
        functools.partial(  # pylint: disable=g-complex-comprehension
            prt,
            enable_dropout=False,
            decode=True,
            max_decode_length=max_decode_length,
            start_idx=interval[0],
            end_idx=interval[1],
            mutable=['cache'],
            method=self.module.decode) for interval in comp_intervals
    ]

    # Switch branches for state propagation. Last branch has zero layers, to be
    # used if no propagation is needed (i.e., didn't exit early).
    state_prop_branches = [
        functools.partial(  # pylint: disable=g-complex-comprehension
            prt,
            enable_dropout=False,
            decode=True,
            max_decode_length=max_decode_length,
            start_idx=interval[0],
            end_idx=None,
            mutable=['cache'],
            only_propagate_state=True,
            method=self.module.decode) for interval in comp_intervals
    ] + [
        functools.partial(  # pylint: disable=g-complex-comprehension
            prt,
            enable_dropout=False,
            decode=True,
            max_decode_length=max_decode_length,
            start_idx=comp_intervals[-1][1],
            end_idx=comp_intervals[-1][1],  # same idx (to just skip)
            mutable=['cache'],
            only_propagate_state=True,
            method=self.module.decode)
    ]

    # TODO(talschuster) convert to named tuple.
    init_state = (flat_logits, decoder_hidden, new_flat_cache, conf, 1,
                  comp_intervals[0][1], meta_score)

    # Runs a segment of the model.
    def body_fn(state):
      _, decoder_hidden, new_flat_cache, _, interval, layer, _ = state

      new_decoder_hidden, new_vars = lax.switch(
          interval,
          branches,
          {
              'params': params,
              'cache': new_flat_cache
          },
          encoded_inputs,
          raw_inputs,  # only needed for encoder padding mask
          flat_ids,
          flat_ids,
          decoder_hidden)

      # If using meta_cls.
      if isinstance(new_decoder_hidden, tuple):
        meta_score = nn.softmax(new_decoder_hidden[1], axis=-1)[..., 1]
        new_decoder_hidden = new_decoder_hidden[0]
      else:
        meta_score = None

      if 'softmax' in self.conf_method:
        cur_flat_logits = self.module.apply(
            {
                'params': params,
                'cache': new_flat_cache
            },
            new_decoder_hidden,
            logit_mask=None,
            enable_dropout=False,
            method=self.module.compute_logits)
        new_flat_logits = cur_flat_logits
      else:
        new_flat_logits = None

      new_flat_cache = new_vars['cache']

      new_conf = self.get_pred_confidence(
          logits=new_flat_logits,
          prev_state=decoder_hidden,
          new_state=new_decoder_hidden,
          meta_score=meta_score)

      layer = lax.min(layer + self.exit_interval, num_layers)
      return (new_flat_logits, new_decoder_hidden, new_flat_cache, new_conf,
              interval + 1, layer, meta_score)

    # Stopping condition (loop continues until it's False).
    def cond_fn(state):
      if self.position_adjusted_threshold:
        # Decays the confidence threshold with decoding time step.
        correct_by_pos = lambda i: conf_threshold * jnp.exp(  # pylint: disable=g-long-lambda
            -self.position_temp * i / max_decode_length
        ) / 10 + 9 * conf_threshold / 10
        adjusted_threshold = correct_by_pos(jnp.min(pos))
      else:
        adjusted_threshold = conf_threshold

      if self.oracle_tok_consistency:
        # Oracle to exit first time predictiong is the same as top layer.
        flat_logits, _, _, _, _, layer, _ = state
        return (flat_logits.argmax() != oracle_tok_pred) & (
            layer < num_layers)
      else:
        # Continues until average batch confidence reaches the threshold, or
        # until all layers were exhausted. Also, doesn't exit before min_exit.
        _, _, _, conf, _, layer, _ = state
        return ((jnp.min(conf) < adjusted_threshold) &
                (layer < num_layers)) | (
                    layer < self.min_exit)

    flat_logits, new_decoder_hidden, new_flat_cache, conf, interval, layer, new_meta_score = lax.while_loop(
        cond_fn, body_fn, init_state)

    if 'softmax' not in self.conf_method:
      # Computes the softmax over the output vocabulary only after exiting.
      flat_logits = self.module.apply(
          {
              'params': params,
              'cache': new_flat_cache
          },
          new_decoder_hidden,
          logit_mask=None,
          enable_dropout=False,
          method=self.module.compute_logits)

    if self.oracle_cache:
      # Run the rest of the layers to compute the real cache (oracle setting).
      def cond_fn_complete_run(state):
        _, _, _, _, _, layer, _ = state
        return layer < num_layers - 1

      post_exit_state = (flat_logits, new_decoder_hidden, new_flat_cache, conf,
                         interval, layer - self.exit_interval, new_meta_score)
      _, _, new_flat_cache, _, _, _, _ = lax.while_loop(cond_fn_complete_run,
                                                        body_fn,
                                                        post_exit_state)
    else:
      # If some decoding layers were skipped, we want to pass the state
      # from the last computed layer to all the upstream skipped layers. This
      # way, the next tokens, if continued to higher layers, can attend back to
      # previous token states. Here, the hidden-state is passed to the higher
      # layers and let each layer compute its own key-value projections.
      _, new_vars = lax.switch(
          interval,
          state_prop_branches,
          {
              'params': params,
              'cache': new_flat_cache
          },
          encoded_inputs,
          raw_inputs,  # only needed for encoder padding mask
          flat_ids,
          flat_ids,
          decoder_hidden)
      new_flat_cache = new_vars['cache']

    if self.oracle_tok_noisy_cache:
      # Takes the logits of the top layer, but uses the hidden state from the
      # exited layer.
      oracle_flat_logits = jnp.squeeze(oracle_flat_logits, axis=1)
      return oracle_flat_logits, new_flat_cache, conf, layer

    # Remove sequence length dimension since it's always 1 during decoding.
    flat_logits = jnp.squeeze(flat_logits, axis=1)

    return flat_logits, new_flat_cache, conf, layer

  def predict_batch_with_aux(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jax.random.KeyArray] = None,
      decoder_params: Optional[MutableMapping[str, Any]] = None,
      return_all_decodes: bool = False,
      num_decodes: int = 1,
      prompt_with_targets: bool = False,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Predict with fast decoding beam search on a batch.

    Here we refer to "parameters" for values that can be compiled into the
    model dynamically, as opposed to static configuration settings that require
    a recompile. For example, the model weights and the decoder brevity-penalty
    are parameters and can be modified without requiring a recompile. The number
    of layers, the batch size and the decoder beam size are configuration
    options that require recompilation if changed.

    This method can be used with a customizable decoding function as long as it
    follows the signature of `DecodeFnCallable`. In order to provide a unified
    interface for the decoding functions, we use a generic names. For example, a
    beam size is a concept unique to beam search. Conceptually, it corresponds
    to the number of sequences returned by the beam search.  Therefore, the
    generic argument `num_decodes` corresponds to the beam size if
    `self._decode_fn` is a beam search. For temperature sampling, `num_decodes`
    corresponds to the number of independent sequences to be sampled. Typically
    `num_decodes = 1` is used for temperature sampling.

    If `return_all_decodes = True`, the return tuple contains the predictions
    with a shape [batch, num_decodes, max_decode_len] and the scores (i.e., log
    probability of the generated sequence) with a shape [batch, num_decodes].

    If `return_all_decodes = False`, the return tuple contains the predictions
    with a shape [batch, max_decode_len] and the scores with a shape [batch].

    `decoder_params` can be used to pass dynamic configurations to
    `self.decode_fn`. An example usage is to pass different random seed (i.e.,
    `jax.random.PRNGKey(seed)` with different `seed` value). This can be done by
    setting `decoder_params['decode_rng'] = jax.random.PRNGKey(seed)`.

    If `prompt_with_targets = True`, then `decoder_prompt_inputs` is initialized
    from the batch's `decoder_input_tokens`. The EOS is stripped to avoid
    decoding to stop after the prompt by matching to `output_vocabulary.eos_id`.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      rng: an optional RNG key to use during prediction, which is passed as
        'decode_rng' to the decoding function.
      decoder_params: additional (model-independent) parameters for the decoder.
      return_all_decodes: whether to return the entire beam or just the top-1.
      num_decodes: the number of beams to use in beam search.
      prompt_with_targets: Whether the force decode decoder_inputs.

    Returns:
      A tuple containing:
        the batch of predictions, with the entire beam if requested
        an auxiliary dictionary of decoder scores
    """
    # Prepare zeroed-out autoregressive cache.
    # [batch, input_len]
    inputs = batch['encoder_input_tokens']
    # [batch, target_len]
    target_shape = batch['decoder_input_tokens'].shape
    target_type = batch['decoder_input_tokens'].dtype
    _, variables_with_cache = self.module.apply(
        {'params': params},
        jnp.ones(inputs.shape, inputs.dtype),
        jnp.ones(target_shape, target_type),
        jnp.ones(target_shape, target_type),
        decode=True,
        enable_dropout=False,
        mutable=['cache'])

    cache = variables_with_cache['cache']

    # Prepare transformer fast-decoder call for beam search: for beam search, we
    # need to set up our decoder model to handle a batch size equal to
    # batch_size * num_decodes, where each batch item's data is expanded
    # in-place rather than tiled.
    # i.e. if we denote each batch element subtensor as el[n]:
    # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
    # [batch * num_decodes, input_len, emb_dim]
    encoded_inputs = decoding.flat_batch_beam_expand(
        self.module.apply({'params': params},
                          inputs,
                          enable_dropout=False,
                          method=self.module.encode), num_decodes)

    # [batch * num_decodes, input_len]
    raw_inputs = decoding.flat_batch_beam_expand(inputs, num_decodes)

    if self.apply_early_inference:
      tokens_ids_to_logits = functools.partial(
          self._compute_logits_from_slice_early_exit,
          params=params,
          encoded_inputs=encoded_inputs,
          raw_inputs=raw_inputs,
          max_decode_length=target_shape[1],
          conf_threshold=self.conf_threshold)
    else:
      tokens_ids_to_logits = functools.partial(
          self._compute_logits_from_slice,
          params=params,
          encoded_inputs=encoded_inputs,
          raw_inputs=raw_inputs,
          max_decode_length=target_shape[1])

    if decoder_params is None:
      decoder_params = {}
    if rng is not None:
      if decoder_params.get('decode_rng') is not None:
        raise ValueError(
            f'Got RNG both from the `rng` argument ({rng}) and '
            f"`decoder_params['decode_rng']` ({decoder_params['decode_rng']}). "
            'Please specify one or the other.')
      decoder_params['decode_rng'] = rng

    # `decoder_prompt_inputs` is initialized from the batch's
    # `decoder_input_tokens`. The EOS is stripped to avoid decoding to stop
    # after the prompt by matching to `output_vocabulary.eos_id`.
    # These inputs are ignored by the beam search decode fn.
    if prompt_with_targets:
      decoder_prompt_inputs = batch['decoder_input_tokens']
      decoder_prompt_inputs = decoder_prompt_inputs * (
          decoder_prompt_inputs != self.output_vocabulary.eos_id)
    else:
      decoder_prompt_inputs = jnp.zeros_like(batch['decoder_input_tokens'])

    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    # decodes: [batch, num_decodes, max_decode_len + 1]
    # scores: [batch, num_decodes]
    scanned = hasattr(self.module, 'scan_layers') and self.module.scan_layers

    if 'eos_id' not in decoder_params:
      decoder_params['eos_id'] = self.output_vocabulary.eos_id
    decodes, scores = self._decode_fn(
        inputs=decoder_prompt_inputs,
        cache=cache,
        tokens_to_logits=tokens_ids_to_logits,
        num_decodes=num_decodes,
        cache_offset=1 if scanned else 0,
        **decoder_params)

    # TODO(talschuster) make the decode func return a general dict.
    if self.apply_early_inference:
      scores, exits, confidences = scores
    else:
      exits, confidences = [], []

    # Beam search returns [n_batch, n_beam, n_length] with beam dimension sorted
    # in increasing order of log-probability.
    # Return the highest scoring beam sequence.
    if return_all_decodes:
      return decodes, {  # pytype: disable=bad-return-type  # jax-ndarray
          'scores': scores,
          'exits': exits,
          'confidences': confidences
      }
    else:
      return decodes[:, -1, :], {
          'scores': scores[:, -1],
          'exits': exits[:, -1, :],
          'confidences': confidences[:, -1, :]
      }
