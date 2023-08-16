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

"""Provides model subclasses with Mixture of Experts support."""

from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple, Union

import clu.metrics as clu_metrics
from flax import core as flax_core
from flax import linen as nn
from flax import traverse_util
from flax.core import scope as flax_scope
import jax
import jax.numpy as jnp
import seqio
from t5x import decoding
from t5x import losses
from t5x import metrics as metrics_lib
from t5x import models as base_models
from t5x import optimizers

AveragePerStep = metrics_lib.AveragePerStep
DecodeFnCallable = base_models.DecodeFnCallable
FrozenVariableDict = flax_scope.FrozenVariableDict
MetricsMap = metrics_lib.MetricsMap
PyTree = base_models.PyTree
Sum = metrics_lib.Sum

MOE_METRICS = (
    'auxiliary_loss',
    'router_z_loss',
    'fraction_tokens_left_behind',
    'expert_usage',
    'router_confidence',
)


class MoeEncoderDecoderModel(base_models.EncoderDecoderModel):
  """Encoder-decoder subclass which propagates MoE auxiliary loss & metrics."""

  def __init__(
      self,
      module: nn.Module,
      input_vocabulary: seqio.Vocabulary,
      output_vocabulary: seqio.Vocabulary,
      optimizer_def: optimizers.OptimizerDefType,
      decode_fn: DecodeFnCallable = decoding.beam_search,
      feature_converter_cls: Optional[
          Callable[..., seqio.FeatureConverter]
      ] = None,
      label_smoothing: float = 0.0,
      z_loss: float = 0.0,
      loss_normalizing_factor: Optional[
          Union[float, int, str, losses.SpecialLossNormalizingFactor]
      ] = None,
      aux_loss_factor: float = 0.0,
      router_z_loss_factor: float = 0.0,
  ):
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
    self._aux_loss_factor = aux_loss_factor
    self._router_z_loss_factor = router_z_loss_factor

  def loss_fn(
      self,
      params: base_models.PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray],
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Cross-entropy loss function with auxiliary MoE losses.

    Args:
      params: Model parameters.
      batch: Batch of training examples.
      dropout_rng: Random number generator key for dropout.

    Returns:
      - Model loss.
      - Metrics.
    """
    logits, state = self._compute_logits(
        params, batch, dropout_rng, mutable=['intermediates']
    )
    return _moe_loss_fn(
        batch,
        logits,
        state,
        self._label_smoothing,
        self._z_loss,
        self._loss_normalizing_factor,
        self._aux_loss_factor,
        self._router_z_loss_factor,
    )

  def predict_batch_with_aux(  # pylint: disable=useless-super-delegation
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

    This override is only included for dependency injection configurability
    (e.g. gin). See parent method docstring for detailed description.

    Args:
      params: Model parameters.
      batch: Batch of inputs.
      rng: RNG key to use during prediction.
      decoder_params: Additional (model-independent) parameters for the decoder.
      return_all_decodes: Whether to return the entire beam or just the top-1.
      num_decodes: Number of beams to use in beam search.
      prompt_with_targets: Whether to force decode decoder_inputs.

    Returns:
      - Batch of predictions, with the entire beam if requested,
      - Auxiliary dictionary of decoder scores.
    """
    return super().predict_batch_with_aux(
        params,
        batch,
        rng,
        decoder_params,
        return_all_decodes,
        num_decodes,
        prompt_with_targets,
    )


class MoeDecoderOnlyModel(base_models.DecoderOnlyModel):
  """Decoder-only subclass which propagates MoE auxiliary loss and metrics."""

  def __init__(
      self,
      module: nn.Module,
      vocabulary: seqio.Vocabulary,
      optimizer_def: optimizers.OptimizerDefType,
      decode_fn: DecodeFnCallable = decoding.temperature_sample,
      inputs_bidirectional_attention: bool = False,
      feature_converter_cls: Optional[
          Callable[..., seqio.FeatureConverter]
      ] = None,
      label_smoothing: float = 0.0,
      z_loss: float = 0.0,
      loss_normalizing_factor: Optional[
          Union[float, int, str, losses.SpecialLossNormalizingFactor]
      ] = None,
      aux_loss_factor: float = 0.0,
      router_z_loss_factor: float = 0.0,
  ):
    super().__init__(
        module=module,
        vocabulary=vocabulary,
        optimizer_def=optimizer_def,
        decode_fn=decode_fn,
        inputs_bidirectional_attention=inputs_bidirectional_attention,
        feature_converter_cls=feature_converter_cls,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
    )
    self._aux_loss_factor = aux_loss_factor
    self._router_z_loss_factor = router_z_loss_factor

  def loss_fn(
      self,
      params: base_models.PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray],
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Cross-entropy loss function with auxiliary MoE losses.

    Args:
      params: Model parameters.
      batch: Batch of training examples.
      dropout_rng: Random number generator key for dropout.

    Returns:
      - Model loss.
      - Metrics.
    """
    logits, state = self._compute_logits(
        params, batch, dropout_rng, mutable=['intermediates']
    )
    return _moe_loss_fn(
        batch,
        logits,
        state,
        self._label_smoothing,
        self._z_loss,
        self._loss_normalizing_factor,
        self._aux_loss_factor,
        self._router_z_loss_factor,
    )

  def predict_batch_with_aux(  # pylint: disable=useless-super-delegation
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jax.random.KeyArray] = None,
      *,
      return_all_decodes: bool = False,
      num_decodes: int = 1,
      decoder_params: Optional[MutableMapping[str, Any]] = None,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Predict with prefix.

    This override is only included for dependency injection configurability
    (e.g. gin). See parent method docstring for detailed description.

    Args:
      params: Model parameters.
      batch: Batch of inputs with the model features specified in
        seqio.DecoderFeatureConverter.
      rng: RNG key to use during prediction.
      return_all_decodes: Whether to return the entire beam or just the top-1.
      num_decodes: Number of decoded sequences to be returned.
      decoder_params: Additional (model-independent) parameters for the decoder.

    Returns:
      Sampled sequences of shape [batch, max_decode_length].
    """
    return super().predict_batch_with_aux(
        params,
        batch,
        rng,
        return_all_decodes=return_all_decodes,
        num_decodes=num_decodes,
        decoder_params=decoder_params,
    )


def _moe_loss_fn(
    batch: Mapping[str, jnp.ndarray],
    logits: jnp.ndarray,
    state: flax_scope.FrozenVariableDict,
    label_smoothing: float,
    z_loss: float,
    loss_normalizing_factor: Optional[float],
    aux_loss_factor: float,
    router_z_loss_factor: float,
) -> Tuple[jnp.ndarray, MetricsMap]:
  """Computes combined cross-entropy and MoE auxiliary loss."""
  loss_normalizing_factor: Optional[
      Union[float, int, str, losses.SpecialLossNormalizingFactor]
  ]
  (loss_normalizing_factor, weights) = (
      losses.get_loss_normalizing_factor_and_weights(
          loss_normalizing_factor, batch
      )
  )

  targets = batch['decoder_target_tokens']
  total_loss, z_loss, _ = losses.compute_weighted_cross_entropy(
      logits,
      targets=targets,
      weights=weights,
      label_smoothing=label_smoothing,
      z_loss=z_loss,
      loss_normalizing_factor=loss_normalizing_factor,
  )

  # Extract and add MoE losses to total loss.
  diversity_metrics = _extract_diversity_metrics(state)

  aux_loss, router_z_loss = _expert_losses(
      diversity_metrics, aux_loss_factor, router_z_loss_factor
  )
  total_loss += aux_loss + router_z_loss

  metrics = base_models.compute_base_metrics(
      logits=logits,
      targets=targets,
      mask=weights,
      loss=total_loss,
      z_loss=z_loss,
  )
  metrics.update(
      _expert_metrics(  # pytype: disable=wrong-arg-types  # jax-ndarray
          diversity_metrics,
          total_loss,
          z_loss,
          aux_loss,
          router_z_loss,
          num_tokens=targets.size,
      )
  )

  return total_loss, metrics


def _extract_diversity_metrics(
    state: flax_scope.FrozenVariableDict,
) -> Dict[str, jnp.ndarray]:
  """Extract average expert diversity metrics from sown state intermediates.

  Args:
    state: Model state holding sown intermediate metrics.

  Returns:
    Diversity metrics, averaged across MoE layers.

  Raises:
    ValueError if unable to extract diversity metrics from model state.
  """
  state_dict = traverse_util.flatten_dict(flax_core.unfreeze(state))

  avg_metrics = {}
  for metric in MOE_METRICS:
    summed_metric = 0.0
    count = 0
    for path, value in state_dict.items():
      if path[-1] == metric:
        summed_metric += jnp.asarray(value, dtype=jnp.float32)
        count += 1

    if count == 0:
      raise ValueError(
          f'Unable to find expert metric: {metric}. Please check that MoE '
          'metrics and losses are correctly sown.'
      )

    avg_metrics[metric] = summed_metric / count

  return avg_metrics


def _expert_losses(
    diversity_metrics: Mapping[str, jnp.ndarray],
    auxiliary_loss_factor: float,
    router_z_loss_factor: float,
) -> Tuple[float, float]:
  """Summarizes per-layer MoE auxiliary losses.

  For auxiliary losses, we take the mean across MoE layers.

  Args:
    diversity_metrics: Per-layer mixture of expert metrics.
    auxiliary_loss_factor: Factor by which to scale auxiliary load balancing
      loss for mixture of experts models. The raw auxiliary losses will be
      summed and then scaled by this factor.
    router_z_loss_factor: Factor by which to scale router z-loss for mixture of
      experts models.

  Returns:
    - Load balancing loss.
    - Router z-loss.
  """
  aux_loss = auxiliary_loss_factor * diversity_metrics['auxiliary_loss'].mean()
  router_z_loss = (
      router_z_loss_factor * diversity_metrics['router_z_loss'].mean()
  )
  return aux_loss, router_z_loss  # pytype: disable=bad-return-type  # jax-ndarray


def _expert_metrics(
    diversity_metrics: Mapping[str, jnp.ndarray],
    total_loss: float,
    z_loss: float,
    auxiliary_loss: float,
    router_z_loss: float,
    num_tokens: int,
) -> MetricsMap:
  """Summarizes per-layer expert metrics for the entire model.

  The return metrics map will also contain overrides for the cross entropy loss
  metrics to account for the MoE losses.

  Args:
    diversity_metrics: Per-layer mixture of expert metrics.
    total_loss: Total model loss.
    z_loss: Output logits z-loss (not MoE specific).
    auxiliary_loss: Auxiliary load balancing loss for MoE models.
    router_z_loss: Router z-loss for MoE models.
    num_tokens: Total number of target tokens.

  Returns:
    Expert diversity metrics.
  """
  cross_ent_loss = total_loss - z_loss - auxiliary_loss - router_z_loss
  return {
      'experts/auxiliary_loss': AveragePerStep.from_model_output(
          auxiliary_loss
      ),
      'experts/router_z_loss': AveragePerStep.from_model_output(router_z_loss),
      'experts/fraction_tokens_left_behind': AveragePerStep.from_model_output(
          diversity_metrics['fraction_tokens_left_behind'].mean()
      ),
      'experts/expert_usage': AveragePerStep.from_model_output(
          diversity_metrics['expert_usage'].mean()
      ),
      'experts/router_confidence': AveragePerStep.from_model_output(
          diversity_metrics['router_confidence'].mean()
      ),
      # Override vanilla T5 cross entropy loss metrics with corrected loss that
      # accounts for MoE losses.
      'cross_ent_loss': metrics_lib.AveragePerStep(total=cross_ent_loss),
      'cross_ent_loss_per_all_target_tokens': clu_metrics.Average(  # pytype: disable=wrong-arg-types  # jnp-array
          total=jnp.sum(cross_ent_loss), count=num_tokens
      ),
  }
