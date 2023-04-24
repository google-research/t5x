# Copyright 2024 The T5X Authors.
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

"""T5X Models.

This module uses layers.py to build a higher-level model structure and define
methods for the loss computation as well as a train, prediction, and evaluation
steps.
"""

import abc
import dataclasses
import functools
import inspect
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple, Union

from absl import logging
import clu.metrics as clu_metrics
from flax import core as flax_core
from flax import linen as nn
from flax.core import scope as flax_scope
from flax.linen import partitioning as flax_partitioning
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import seqio
from t5x import decoding
from t5x import losses
from t5x import metrics as metrics_lib
from t5x import optimizers
import tensorflow as tf
import typing_extensions

from t5x.te_helper import TransformerEngineHelper

# Remove _ShardedDeviceArray when users of t5x have their types updated
_ShardedDeviceArray = Any
Array = Union[np.ndarray, jnp.ndarray, _ShardedDeviceArray, tf.Tensor]
MetricsMap = metrics_lib.MetricsMap
PyTree = Any
PyTreeDef = jax.tree_util.PyTreeDef


class TokensIdsToLogitsCallable(typing_extensions.Protocol):
  """Token ids to logits mapping call signature."""

  def __call__(
      self, decoding_state: decoding.DecodingState
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Performs forward pass to convert token ids to logits.

    Args:
      decoding_state: Current decoding state, including current token ids and
        cache.

    Returns:
      a tuple of logits with a shape [batch_size, vocab_size] and an updated
      cache.
    """
    ...


class DecodeFnCallable(typing_extensions.Protocol):
  """Decoding function call signature."""

  def __call__(
      self,
      *,
      inputs: jnp.ndarray,
      cache: Mapping[str, jnp.ndarray],
      tokens_to_logits: TokensIdsToLogitsCallable,
      eos_id: int,
      num_decodes: int,
      decode_rng: Optional[jax.Array],
      cache_offset: int,
      **kwargs,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    """
    ...


class BaseModel(abc.ABC):
  """Abstract base class for models.

  Wraps a flax module to provide a basic interface for computing loss,
  evaluation metrics, prediction, and scoring.

  Subclasses must implement the abstract methods. Any additional arguments added
  to these methods must have defaults or be bound at run time to fit the
  interface expected by the standard training, inference, and evaluation
  functions.
  """

  FEATURE_CONVERTER_CLS: Callable[..., seqio.FeatureConverter]

  def __init__(self, optimizer_def: optimizers.OptimizerDefType):
    # TODO(jbulian): Move the optimizer out of the model and make it a training
    #                parameter.
    self.optimizer_def = optimizer_def

  @abc.abstractmethod
  def loss_fn(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array],
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Computes loss and metrics.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      dropout_rng: rng to use for dropout, or None for deterministic mode.

    Returns:
      loss: the loss computed for the given inputs and parameters.
      aux:
        weight_sum: sum of the per-token weights applied to the loss.
        metrics: a mapping of metrics computed for this batch.
    """
    pass

  def eval_fn(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Computes loss and metrics during the evaluation.

    Args:
      params: model parameters.
      batch: a batch of inputs.

    Returns:
      loss: the loss computed for the given inputs and parameters.
      aux:
        weight_sum: sum of the per-token weights applied to the loss.
        metrics: a mapping of metrics computed for this batch.
    """
    return self.loss_fn(
        params=params,
        batch=batch,
        dropout_rng=None,
        flax_mutables=flax_mutables,
    )

  def predict_batch(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jax.Array] = None,
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> jnp.ndarray:
    """Predicts a batch of outputs from the model.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      rng: an optional RNG to use during prediction (e.g., for decoding).

    Returns:
      The model predictions.
    """
    return self.predict_batch_with_aux(params=params, batch=batch, rng=rng, flax_mutables=flax_mutables)[0]

  @abc.abstractmethod
  def predict_batch_with_aux(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jax.Array] = None,
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Predict a batch from the modelwith auxiliary outputs.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      rng: an optional RNG key to use during prediction (e.g., for decoding).

    Returns:
      predictions: the model predictions
      aux: auxiliary data
    """
    pass

  @abc.abstractmethod
  def score_batch(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> jnp.ndarray:
    """Computes scores for batch."""
    pass

  @abc.abstractmethod
  def get_initial_variables(
      self,
      rng: jax.Array,
      input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str, jnp.dtype]] = None,
  ) -> flax_scope.FrozenVariableDict:
    """Returns the initial variables of the model."""
    pass


class BaseTransformerModel(BaseModel):
  """Abstract base class for Transformer models.

  Subclasses must implement `predict_batch_with_aux`, `score_batch`,
  `get_initial_variables` from `BaseModel` as well as `_compute_logits`.
  """

  def __init__(
      self,
      module: nn.Module,
      input_vocabulary: seqio.Vocabulary,
      output_vocabulary: seqio.Vocabulary,
      optimizer_def: optimizers.OptimizerDefType,
      decode_fn: Optional[DecodeFnCallable] = None,
      label_smoothing: float = 0.0,
      z_loss: float = 0.0,
      loss_normalizing_factor: Optional[
          Union[float, int, str, losses.SpecialLossNormalizingFactor]
      ] = None,
  ):
    self.module = module
    self._input_vocabulary = input_vocabulary
    self._output_vocabulary = output_vocabulary
    self._decode_fn = decode_fn
    self._label_smoothing = label_smoothing
    self._z_loss = z_loss
    self._loss_normalizing_factor = loss_normalizing_factor

    super().__init__(optimizer_def=optimizer_def)

  @property
  def input_vocabulary(self):
    return self._input_vocabulary

  @property
  def output_vocabulary(self):
    return self._output_vocabulary

  @property
  def decode_fn(self):
    return self._decode_fn

  @abc.abstractmethod
  def _compute_logits(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array] = None,
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> jnp.ndarray:
    """Computes logits via a forward pass of the model."""
    pass

  def loss_fn(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array],
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Loss function used for training with a cross-entropy loss."""
    logits = self._compute_logits(params, batch, dropout_rng, 
                                  flax_mutables=flax_mutables)

    loss_normalizing_factor: Optional[
        Union[float, int, str, losses.SpecialLossNormalizingFactor]
    ]
    (loss_normalizing_factor, weights) = (
        losses.get_loss_normalizing_factor_and_weights(
            self._loss_normalizing_factor, batch
        )
    )

    loss, z_loss, _ = losses.compute_weighted_cross_entropy(
        logits,
        targets=batch['decoder_target_tokens'],
        weights=weights,
        label_smoothing=self._label_smoothing,
        z_loss=self._z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
    )

    # segment ids to compute packing, padding etc.
    segment_ids = {
        k[: -len('_segment_ids')]: v
        for k, v in batch.items()
        if k.endswith('_segment_ids')
    }
    # If these don't exist then we can create only padding mask.
    if not segment_ids:
      segment_ids = {
          k: v != 0
          for k, v in batch.items()
          if k in ('encoder_input_tokens', 'decoder_target_tokens')
      }

    metrics = self._compute_metrics(
        logits=logits,
        targets=batch['decoder_target_tokens'],
        mask=weights,
        loss=loss,
        z_loss=z_loss,
        segment_ids=segment_ids,
    )
    return loss, metrics

  def eval_fn(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      flax_mutables:  Optional[PyTreeDef] = None,
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Computes loss and metrics during the evaluation.

    Args:
      params: model parameters.
      batch: a batch of inputs.

    Returns:
      loss: the loss computed for the given inputs and parameters.
      aux:
        weight_sum: sum of the per-token weights applied to the loss.
        metrics: a mapping of metrics computed for this batch.
    """
    return self.loss_fn(
        params=params,
        batch=batch,
        dropout_rng=None,
        flax_mutables=flax_mutables,
    )

  def _compute_metrics(
      self,
      logits: jnp.ndarray,
      targets: jnp.ndarray,
      mask: jnp.ndarray,
      loss: jnp.ndarray,
      z_loss: Optional[jnp.ndarray] = None,
      segment_ids: Optional[Mapping[str, jnp.ndarray]] = None,
  ) -> MetricsMap:
    return compute_base_metrics(
        logits=logits,
        targets=targets,
        mask=mask,
        loss=loss,
        z_loss=z_loss,
        segment_ids=segment_ids,
    )


@dataclasses.dataclass(frozen=True)
class DecoderParams:
  return_all_decodes: bool = False
  num_decodes: int = 1


class EncoderDecoderModel(BaseTransformerModel):
  """Wrapper class for the models.Transformer nn.module."""

  FEATURE_CONVERTER_CLS = seqio.EncDecFeatureConverter

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
      default_decoder_params: Optional[DecoderParams] = None,
  ):
    if feature_converter_cls is not None:
      self.FEATURE_CONVERTER_CLS = feature_converter_cls  # pylint: disable=invalid-name
    self._default_decoder_params = default_decoder_params or DecoderParams()
    super().__init__(
        module=module,
        input_vocabulary=input_vocabulary,
        output_vocabulary=output_vocabulary,
        optimizer_def=optimizer_def,
        decode_fn=decode_fn,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
    )

  def get_initial_variables(
      self,
      rng: jax.Array,
      input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str, jnp.dtype]] = None,
  ) -> flax_scope.FrozenVariableDict:
    """Get the initial variables for an encoder-decoder model."""
    input_types = {} if input_types is None else input_types
    encoder_shape = input_shapes['encoder_input_tokens']
    encoder_type = input_types.get('encoder_input_tokens', jnp.float32)
    decoder_shape = input_shapes['decoder_input_tokens']
    decoder_type = input_types.get('decoder_input_tokens', jnp.float32)
    if 'encoder_positions' in input_shapes:
      encoder_positions = jnp.ones(
          input_shapes['encoder_positions'],
          input_types.get('encoder_positions', jnp.int32),
      )
    else:
      encoder_positions = None
    if 'decoder_positions' in input_shapes:
      decoder_positions = jnp.ones(
          input_shapes['decoder_positions'],
          input_types.get('decoder_positions', jnp.int32),
      )
    else:
      decoder_positions = None
    if 'encoder_segment_ids' in input_shapes:
      encoder_segment_ids = jnp.ones(
          input_shapes['encoder_segment_ids'],
          input_types.get('encoder_segment_ids', jnp.int32),
      )
    else:
      encoder_segment_ids = None
    if 'decoder_segment_ids' in input_shapes:
      decoder_segment_ids = jnp.ones(
          input_shapes['decoder_segment_ids'],
          input_types.get('decoder_segment_ids', jnp.int32),
      )
    else:
      decoder_segment_ids = None
    initial_variables = flax_core.freeze(
        self.module.init(
            rng,
            jnp.ones(encoder_shape, encoder_type),
            jnp.ones(decoder_shape, decoder_type),
            jnp.ones(decoder_shape, decoder_type),
            encoder_positions=encoder_positions,
            decoder_positions=decoder_positions,
            encoder_segment_ids=encoder_segment_ids,
            decoder_segment_ids=decoder_segment_ids,
            decode=False,
            enable_dropout=False,
        )
    )
    return initial_variables

  def _compute_logits(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array] = None,
      mutable: flax_scope.CollectionFilter = False,
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
    """Computes logits via a forward pass of `self.module_cls`."""
    # Dropout is provided only for the training mode.
    rngs = {'dropout': dropout_rng} if dropout_rng is not None else None
    if flax_mutables is None:
      flax_mutables = {}
    return self.module.apply(
        {
            'params': params,
            **flax_mutables,
        },
        batch['encoder_input_tokens'],
        batch['decoder_input_tokens'],
        batch['decoder_target_tokens'],
        encoder_segment_ids=batch.get('encoder_segment_ids', None),
        decoder_segment_ids=batch.get('decoder_segment_ids', None),
        encoder_positions=batch.get('encoder_positions', None),
        decoder_positions=batch.get('decoder_positions', None),
        decode=False,
        enable_dropout=rngs is not None,
        rngs=rngs,
        mutable=mutable,
    )

  def _compute_logits_from_slice(
      self,
      decoding_state: decoding.DecodingState,
      params: PyTree,
      encoded_inputs: jnp.ndarray,
      raw_inputs: jnp.ndarray,
      max_decode_length: int,
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Token slice to logits from decoder model."""
    flat_ids = decoding_state.cur_token
    flat_cache = decoding_state.cache

    if flax_mutables is None:
      flax_mutables = {}

    # flat_ids: [batch * beam, seq_len=1]
    # cache is expanded inside beam_search to become flat_cache
    # flat_cache: [batch * beam, num_heads, depth_per_head, max_decode_len]
    # flat_logits: [batch * beam, seq_len=1, vocab]
    flat_logits, new_vars = self.module.apply(
        {
            'params': params,
            'cache': flat_cache,
            **flax_mutables,
        },
        encoded_inputs,
        raw_inputs,  # only needed for encoder padding mask
        flat_ids,
        flat_ids,
        enable_dropout=False,
        decode=True,
        max_decode_length=max_decode_length,
        mutable=['cache'],
        method=self.module.decode,
    )
    # Remove sequence length dimension since it's always 1 during decoding.
    flat_logits = jnp.squeeze(flat_logits, axis=1)
    new_flat_cache = new_vars['cache']
    return flat_logits, new_flat_cache

  def _compute_kv_cache(
      self,
      params,
      encoded_inputs: jnp.ndarray,
      encoder_input_tokens: jnp.ndarray,
      decoder_input_tokens: jnp.ndarray,
      prefill_decoder_prompt: bool = False,
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> Tuple[PyTree, Optional[jnp.ndarray]]:
    """Initialize the key/value cache, with optional prompt.

    Args:
      params: The parameters of the model.
      encoded_inputs: Output of the encoder on the inputs.
      encoder_input_tokens: Input tokens for the encoder. Only needed for
        padding mask.
      decoder_input_tokens: Input tokens for the decoder, possibly containing a
        prompt.
      prefill_decoder_prompt: Whether to prefill the cache using the decoder
        prompt.

    Returns:
      cache: The initialzed cache.
      initial_index: The index of the next position following prefill or None if
        `prefill_decoder_prompt` is False.
    """
    if flax_mutables is None:
        flax_mutables = {}
    del encoded_inputs
    _, initial_variables = self.module.apply(
        {
            'params': params,
            **flax_mutables,
        },
        encoder_input_tokens=jnp.ones_like(encoder_input_tokens),
        decoder_input_tokens=jnp.ones_like(decoder_input_tokens),
        decoder_target_tokens=jnp.ones_like(decoder_input_tokens),
        mutable=['cache'],
        decode=True,
        enable_dropout=False,
    )

    cache = initial_variables['cache']

    if not prefill_decoder_prompt:
      return cache, None

    # Prefill the cache based on an (optional) prompt.
    # We assume the only 0 tokens are a BOS=0 token at the beginning of the
    # input and PAD=0 tokens at the end.
    inputs_lengths = jnp.sum(decoder_input_tokens != 0, axis=1)

    _, variables_with_cache = self.module.apply(
        {'params': params, 'cache': cache},
        encoded=encoded_inputs,
        encoder_input_tokens=encoder_input_tokens,  # only for padding mask,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=jnp.ones_like(decoder_input_tokens),  # for shape
        mutable=['cache'],
        enable_dropout=False,
        prefill=True,
        prefill_lengths=inputs_lengths,
        method=self.module.decode,
    )

    cache = variables_with_cache['cache']
    if 'position_embedder' in cache['decoder']:
      # TODO(adarob): Instead have `module.decode` accept an index.
      cache['decoder']['position_embedder'][
          'position_embedder_index'
      ] = inputs_lengths

    return cache, inputs_lengths

  def predict_batch(self,
                    params: PyTreeDef,
                    batch: Mapping[str, jnp.ndarray],
                    rng: Optional[jax.random.KeyArray] = None,
                    flax_mutables: Optional[PyTreeDef] = None) -> jnp.ndarray:
    """Predicts a batch of outputs from the model.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      rng: an optional RNG to use during prediction (e.g., for decoding).

    Returns:
      The model predictions.
    """
    return self.predict_batch_with_aux(params=params, batch=batch, 
                                       rng=rng, flax_mutables=flax_mutables)[0]

  def predict_batch_with_aux(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jax.Array] = None,
      decoder_params: Optional[MutableMapping[str, Any]] = None,
      return_all_decodes: bool = None,
      num_decodes: int = None,  # pytype:disable=annotation-type-mismatch
      prompt_with_targets: bool = False,
      flax_mutables: Optional[PyTreeDef] = None,
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
    The beam dimension is sorted in increasing order of log-probability.

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
    if return_all_decodes is None:
      return_all_decodes = self._default_decoder_params.return_all_decodes
    if num_decodes is None:
      num_decodes = self._default_decoder_params.num_decodes
    if flax_mutables is None:
      flax_mutables = {}

    # [batch, input_len]
    encoder_input_tokens = batch['encoder_input_tokens']
    decoder_input_tokens = batch['decoder_input_tokens']

    # Prepare transformer fast-decoder call for beam search: for beam search, we
    # need to set up our decoder model to handle a batch size equal to
    # batch_size * num_decodes, where each batch item's data is expanded
    # in-place rather than tiled.
    # i.e. if we denote each batch element subtensor as el[n]:
    # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
    # [batch * num_decodes, input_len, emb_dim]
    encoded_inputs = decoding.flat_batch_beam_expand(
        self.module.apply(
            {'params': params, **flax_mutables},
            encoder_input_tokens,
            enable_dropout=False,
            method=self.module.encode,
        ),
        num_decodes,
    )

    # `decoder_prompt_inputs` is initialized from the batch's
    # `decoder_input_tokens`. The EOS is stripped to avoid decoding to stop
    # after the prompt by matching to `output_vocabulary.eos_id`.
    # These inputs are ignored by the beam search decode fn.
    if prompt_with_targets:
      decoder_prompt_inputs = decoder_input_tokens
      decoder_prompt_inputs = decoder_prompt_inputs * (
          decoder_prompt_inputs != self.output_vocabulary.eos_id
      )
    else:
      decoder_prompt_inputs = jnp.zeros_like(decoder_input_tokens)

    encoded_inputs = self.module.apply(
        {'params': params},
        encoder_input_tokens,
        enable_dropout=False,
        method=self.module.encode,
    )

    # Prepare autoregressive cache.
    if 'initial_index' not in inspect.signature(self.decode_fn).parameters:
      logging.info(
          'Disabling prompt prefilling due to incompatible decode fn: %s.',
          self.decode_fn,
      )
      prefill_decoder_prompt = False
    elif 'prefill' not in inspect.signature(self.module.decode).parameters:
      logging.info(
          'Disabling prompt prefilling due to incompatible `module.decode`.'
      )
      prefill_decoder_prompt = False
    else:
      logging.info('Enabling prompt prefilling.')
      prefill_decoder_prompt = True
    cache, initial_index = self._compute_kv_cache(
        params,
        encoded_inputs=encoded_inputs,
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_prompt_inputs,
        prefill_decoder_prompt=prefill_decoder_prompt,
        flax_mutables=flax_mutables,
    )

    # Prepare transformer fast-decoder call for beam search: for beam search, we
    # need to set up our decoder model to handle a batch size equal to
    # batch_size * num_decodes, where each batch item's data is expanded
    # in-place rather than tiled.
    # i.e. if we denote each batch element subtensor as el[n]:
    # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
    # [batch * num_decodes, input_len, emb_dim]
    tokens_ids_to_logits = functools.partial(
        self._compute_logits_from_slice,
        params=params,
        # [batch * num_decodes, input_len, emb_dim]
        encoded_inputs=decoding.flat_batch_beam_expand(
            encoded_inputs, num_decodes
        ),
        # [batch * num_decodes, input_len]
        raw_inputs=decoding.flat_batch_beam_expand(
            encoder_input_tokens, num_decodes
        ),
        max_decode_length=decoder_input_tokens.shape[1],
        flax_mutables=flax_mutables,
    )

    if decoder_params is None:
      decoder_params = {}
    if initial_index is not None:
      # We only set initial_index when it's non-None since it is not supported
      # by all decoders.
      decoder_params['initial_index'] = initial_index

    if rng is not None:
      if decoder_params.get('decode_rng') is not None:
        raise ValueError(
            f'Got RNG both from the `rng` argument ({rng}) and'
            " `decoder_params['decode_rng']`"
            f' ({decoder_params["decode_rng"]}). Please specify one or the'
            ' other.'
        )
      decoder_params['decode_rng'] = rng

    # TODO(hwchung): rename the returned value names to more generic ones.
    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    # decodes: [batch, num_decodes, max_decode_len + 1]
    # scores: [batch, num_decodes]
    scanned = hasattr(self.module, 'scan_layers') and self.module.scan_layers
    cfg = TransformerEngineHelper.get_t5x_config(self.module.config)

    if 'eos_id' not in decoder_params:
      decoder_params['eos_id'] = self.output_vocabulary.eos_id
    decodes, scores = self._decode_fn(
        inputs=decoder_prompt_inputs,
        cache=cache,
        tokens_to_logits=tokens_ids_to_logits,
        num_decodes=num_decodes,
        cache_offset=1 if (scanned or cfg.transpose_batch_sequence) else 0,
        **decoder_params
    )

    # Beam search returns [n_batch, n_beam, n_length] with beam dimension sorted
    # in increasing order of log-probability.
    # Return the highest scoring beam sequence.
    if return_all_decodes:
      return decodes, {'scores': scores}
    else:
      return decodes[:, -1, :], {'scores': scores[:, -1]}

  def score_batch(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, Any]]]:
    """Compute log likelihood score on a batch."""
    weights = batch['decoder_loss_weights']
    target_tokens = batch['decoder_target_tokens']

    if return_intermediates:
      logits, modified_variables = self._compute_logits(
          params=params, batch=batch, mutable=['intermediates'],
          flax_mutables=flax_mutables,
      )

      # Inside self.module, we called nn.Module.sow to track various
      # intermediate values. We extract them here.
      intermediates = flax_core.unfreeze(
          modified_variables.get('intermediates', {})
      )

      # Track per-token labels and loss weights as well. These are not
      # intermediate values of logit computation, so we manually add them here.
      intermediates.setdefault('decoder', {})
      intermediates['decoder']['target_tokens'] = (target_tokens,)
      intermediates['decoder']['loss_weights'] = (weights,)
      # Note that the values are singleton tuples. This is because values inside
      # `intermediates` should be tuples tracking all instantiations of a value.
      # These values each have just one instantiation, hence singletons.
    else:
      logits = self._compute_logits(params, batch, flax_mutables=flax_mutables)  # type: jnp.ndarray  # pytype: disable=annotation-type-mismatch  # jax-ndarray

    # Purposefully don't use config.z_loss because that term is for training
    # stability and shouldn't affect our reported scores.
    token_scores = (
        -losses.cross_entropy_with_logits(
            logits,
            common_utils.onehot(
                target_tokens, logits.shape[-1], on_value=1, off_value=0
            ),
            z_loss=0.0,
        )[0]
        * weights
    )
    if return_intermediates:
      intermediates['decoder']['token_scores'] = (token_scores,)

    sequence_scores = token_scores.sum(-1)

    if return_intermediates:
      return sequence_scores, intermediates

    return sequence_scores


class DecoderOnlyModel(BaseTransformerModel):
  """Model class for the decoder-only modules.

  It accepts inputs made out of only 'targets' or both 'inputs'
  and 'targets'. If both 'inputs' and 'targets' are present, the loss will
  be computed only on 'targets'.

  By default the self-attention is fully causal and a given position only
  attends to the time steps before and itself. If
  `inputs_bidirectional_attention = True`, the attention in the "inputs" region
  is bidirectional. This architecture was referred to as "Prefix LM" in Raffel
  et al. 2019 (https://arxiv.org/abs/1910.10683).
  """

  FEATURE_CONVERTER_CLS = seqio.DecoderFeatureConverter

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
  ):
    if feature_converter_cls is not None:
      self.FEATURE_CONVERTER_CLS = feature_converter_cls  # pylint: disable=invalid-name
    self._inputs_bidirectional_attention = inputs_bidirectional_attention
    super().__init__(
        module,
        input_vocabulary=vocabulary,
        output_vocabulary=vocabulary,
        optimizer_def=optimizer_def,
        decode_fn=decode_fn,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
    )

  def get_initial_variables(
      self,
      rng: jax.Array,
      input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str, jnp.dtype]] = None,
  ) -> flax_scope.FrozenVariableDict:
    """Get the initial variables."""
    input_types = {} if input_types is None else input_types
    decoder_shape = input_shapes['decoder_input_tokens']
    decoder_type = input_types.get('decoder_input_tokens', jnp.float32)
    initial_variables = self.module.init(
        rng,
        jnp.ones(decoder_shape, decoder_type),
        jnp.ones(decoder_shape, decoder_type),
        enable_dropout=False,
    )
    return flax_core.freeze(initial_variables)

  def _get_decoder_causal_attention(self, batch):
    """Returns decoder causal attention from the batch or None."""
    if self._inputs_bidirectional_attention:
      if 'decoder_causal_attention' not in batch:
        raise ValueError(
            '`inputs_bidirectional_attention` mode requires '
            '"decoder_causal_attention" feature in the batch'
        )
      decoder_causal_attention = batch['decoder_causal_attention']
    else:
      decoder_causal_attention = None

    return decoder_causal_attention

  def _compute_logits(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array] = None,
      mutable: flax_scope.CollectionFilter = False,
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> jnp.ndarray:
    """Computes logits via a forward pass of `self.module`."""
    rngs = {'dropout': dropout_rng} if dropout_rng is not None else None
    decoder_causal_attention = self._get_decoder_causal_attention(batch)
    if flax_mutables is None:
      flax_mutables = {}

    return self.module.apply(
        {
            'params': params,
            **flax_mutables,
        },
        batch['decoder_input_tokens'],
        batch['decoder_target_tokens'],
        decoder_segment_ids=batch.get('decoder_segment_ids', None),
        decoder_positions=batch.get('decoder_positions', None),
        decoder_causal_attention=decoder_causal_attention,
        rngs=rngs,
        decode=False,
        enable_dropout=rngs is not None,
        mutable=mutable,
    )

  def _compute_logits_from_slice(
      self,
      decoding_state: decoding.DecodingState,
      params: PyTree,
      max_decode_length: int,
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Token slice to logits from decoder model."""
    flat_ids = decoding_state.cur_token
    flat_cache = decoding_state.cache
    # flat_ids: [batch, seq_len=1]
    # flat_cache['cached_(keys|values)']:
    #   [batch, num_heads, depth_per_head, max_decode_length]
    # flat_cache['cache_index']: [batch]
    # flat_logits: [batch, seq_len=1, vocab]
    flat_logits, new_vars = self.module.apply(
        {
            'params': params,
            'cache': flat_cache,
            **flax_mutables,
        },
        flat_ids,
        flat_ids,
        enable_dropout=False,
        decode=True,
        max_decode_length=max_decode_length,
        mutable=['cache'],
    )
    # Remove sequence length dimension since it's always 1 during decoding.
    flat_logits = jnp.squeeze(flat_logits, axis=1)
    new_flat_cache = new_vars['cache']
    return flat_logits, new_flat_cache

  def score_batch(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> jnp.ndarray:
    """Compute log likelihood score on a batch."""

    decoder_target_tokens = batch['decoder_target_tokens']
    weights = batch['decoder_loss_weights']

    if return_intermediates:
      logits, modified_variables = self._compute_logits(
          params=params,
          batch=batch,
          dropout_rng=None,
          mutable=['intermediates'],
          flax_mutables=flax_mutables,
      )

      # Inside self.module, we called nn.Module.sow to track various
      # intermediate values. We extract them here.
      intermediates = flax_core.unfreeze(
          modified_variables.get('intermediates', {})
      )

      # Track per-token labels and loss weights as well. These are not
      # intermediate values of logit computation, so we manually add them here.
      intermediates.setdefault('decoder', {})
      intermediates['decoder']['target_tokens'] = (decoder_target_tokens,)
      intermediates['decoder']['loss_weights'] = (weights,)
      # Note that the values are singleton tuples. This is because values inside
      # `intermediates` should be tuples tracking all instantiations of a value.
      # These values each have just one instantiation, hence singletons.
    else:
      logits = self._compute_logits(
          params=params, batch=batch, dropout_rng=None, flax_mutables=flax_mutables,
      )

    token_scores = (
        -losses.cross_entropy_with_logits(
            logits,
            common_utils.onehot(
                decoder_target_tokens, logits.shape[-1], on_value=1, off_value=0
            ),
            z_loss=0.0,
        )[0]
        * weights
    )
    if return_intermediates:
      intermediates['decoder']['token_scores'] = (token_scores,)

    sequence_scores = token_scores.sum(-1)

    if return_intermediates:
      return sequence_scores, intermediates  # pytype: disable=bad-return-type  # jax-ndarray

    return sequence_scores

  def _compute_kv_cache(
      self,
      params: PyTree,
      inputs: jnp.ndarray,
      causal_attention_mask: jnp.ndarray,
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> Tuple[PyTree, jnp.ndarray]:
    """Compute the key/value cache on the input prompt.

    Args:
      params: The parameters of the model.
      inputs: Tokens to use for prompting, with 0-padding.
      causal_attention_mask: A boolean mask containing 1 at positions that are
        treated as inputs.

    Returns:
      cache: The prefilled cache.
      initial_index: The index of the next position following prefill.
    """
    if flax_mutables is None:
        flax_mutables = {}
    # The lengths of the inputs match the number of non-padding positions,
    # excluding the initial BOS.
    inputs_lengths = jnp.sum(inputs[:, 1:] != 0, axis=-1)

    _, initial_variables = self.module.apply(
        {
            'params': params,
            **flax_mutables,
        },
        jnp.ones_like(inputs),
        jnp.ones_like(inputs),
        enable_dropout=False,
        decode=True,
        mutable=['cache'],
    )
    cache = initial_variables['cache']
    if 'cache_axes' in initial_variables:
      cache_axes = initial_variables['cache_axes']

      cache = jax.tree_util.tree_map(
          flax_partitioning.with_sharding_constraint,
          cache,
          flax_partitioning.get_axis_names(cache_axes),
      )

    # Prefill our cache with all the inputs. `inputs_lengths` is the index of
    # the last input token. The cache will be filled for all the input
    # positions, save the last input token. The cache index will point to the
    # index of this last input token which is considered during prefilling but
    # not cached. This re-computation is required as the logits for this
    # position are required for selecting the first output token.
    #
    # The cache is still `[B, ..., max_decode_len]` but any position less than
    # the `inputs_length` will be non-zero, that is
    # `cached_key[b, ..., i < inputs_lengths[b]] != 0`.
    #
    # The cache index is now a vector of size [B] = input_lengths

    # If `self._inputs_bidirectional_attention = False`, we should not pass
    # batch['decoder_causal_attention'] to `module.apply` during cache prefill
    # and pass None instead.
    maybe_causal_attention_mask = self._get_decoder_causal_attention(
        {'decoder_causal_attention': causal_attention_mask}
    )

    _, variables_with_cache = self.module.apply(
        {
            'params': params,
            'cache': cache,
            **flax_mutables,
        },
        decoder_input_tokens=inputs,
        # Use the `decoder_causal_attention`, which has 1 for all input
        # positions, including the BOS token, as the targets so when the
        # decoder attention mask is built, it will correctly cover the whole
        # input, Using something like the inputs will cause the first input
        # token (the 0 for BOS) will not be included in the mask. This also
        # restricts the mask to not include any target positions like it would
        # if you used `decoder_target_tokens`.
        decoder_target_tokens=causal_attention_mask,
        decoder_causal_attention=maybe_causal_attention_mask,
        mutable=['cache'],
        enable_dropout=False,
        prefill=True,
        prefill_lengths=inputs_lengths,
    )
    return variables_with_cache['cache'], inputs_lengths

  def predict_batch_with_aux(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jax.Array] = None,
      *,
      return_all_decodes: bool = False,
      num_decodes: int = 1,
      decoder_params: Optional[MutableMapping[str, Any]] = None,
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Predict with prefix.

    `decoder_params` can be used to pass dynamic configurations to
    `self.decode_fn`. An example usage is to pass different random seed (i.e.,
    `jax.random.PRNGKey(seed)` with different `seed` value). This can be done by
    setting `decoder_params['decode_rng'] = jax.random.PRNGKey(seed)`.

    Although this method is short, there are a few subtle points that. We use a
    running example to make these points clear.

    ```
    Example
      inputs = [9, 4, 6, 1]
      targets = [3, 9, 1]

      seqio.DecoderFeatureConverter will generate these set of features

         decoder_target_tokens = [9, 4, 6, 1, 3, 9, 1, 0, 0]
          decoder_input_tokens = [0, 9, 4, 6, 1, 3, 9, 1, 0]
      decoder_causal_attention = [1, 1, 1, 1, 1, 0, 0, 0, 0]

      The output of this function is (a` through `e` are the sampled token ids):

             sampled_sequences = [9, 4, 6, 1, a, b, c, d, e].
    ```

    Given these set of features, we make a few important observation.

    1) When a decoder-only model is used for a supervised learning with "inputs"
       and "targets", one way to handle this is to concatenate the "inputs" and
       "targets". For training, we use teacher forcing for the entire
       concatenated sequence. For inference, on the other hand, we don't have
       the targets. This requires that we use teacher forcing on the "inputs"
       portion while using the generated token as the input token for the next
       decoding step. For evaluation, we do have "targets" but we only want to
       use them for computing metrics, i.e., by comparing to the sequence
       generated by the model.

       This function is currently used for evaluation mode, but by ignoring
       "targets", it can be extended for the inference mode.

    2) During evaluation mode, the targets portion is zeroed out and they are
       filled with the sampled token ids. The inputs portion is kept intact.

    3) Note that `decoder_causal_attention` has an additional 1 after the final
       "inputs" token. This is because the position where the last "inputs"
       token (in this case 1) is input and the output is the first "target"
       token (in this case 3) can be included in the non-causal attention
       region.

       This results in an alignment between `decoder_input_tokens` and
       `decoder_causal_attention` because the former is shifted to the right by
       one position. So we use `decoder_causal_attention` as a binary mask to
       zero out the target tokens in `decoder_input_tokens`.

    Note:
      In order to use a custom self._decode_fn with this model it must support:

      1) Decoding from a partially decoded state by accepting a vector of
         `initial_indices` that specify where in the input to start decoding
         from.
      2) Using a vector as the loop counter to support different examples being
         a different number of steps into their decoding loop.
      3) Be able to handle one batch element reaching `max_decode_length`
         before the others without it causing the model to prematurely stop
         decoding.

    Args:
      params: model parameters.
      batch: batch element with the model features specified in
        seqio.DecoderFeatureConverter.
      rng: an optional RNG key to use during prediction, which is passed as
        'decode_rng' to the decoding function.
      return_all_decodes: if True, will return all batch_size * num_decodes
        samples from the model as an array of shape [batch_size, num_decodes,
        sequence_length]. In this case the `num_decodes` dimension is sorted in
        increasing order of log-probability. Otherwise returns only the most
        likely samples as an array of shape [batch_size, sequence_length].
      num_decodes: number of decoded sequences to be returned.
      decoder_params: additional (model-independent) parameters for the decoder.

    Returns:
      sampled_sequences: an array of shape [batch, max_decode_length].
    """
    if 'decoder_causal_attention' not in batch:
      raise ValueError(
          'Batch does not have the right format for text generation: probably '
          'because `task_feature_lengths` passed to the feature converter does '
          'not have both `inputs` and `targets`.'
      )

    # since decoder_input_tokens is shifted to the right and
    # `decoder_causal_attention` has one more 1 than the number of inputs
    # tokens, this masks out targets portion of the decoder_input_tokens.
    inputs = batch['decoder_input_tokens'] * batch['decoder_causal_attention']

    prefilled_cache, initial_index = self._compute_kv_cache(
        params, inputs, batch['decoder_causal_attention'], flax_mutables=flax_mutables,
    )

    target_shape = batch['decoder_input_tokens'].shape
    max_decode_length = target_shape[1]

    tokens_ids_to_logits = functools.partial(
        self._compute_logits_from_slice,
        params=params,
        max_decode_length=max_decode_length,
        flax_mutables=flax_mutables,
    )

    if decoder_params is None:
      decoder_params = {}
    if rng is not None:
      if decoder_params.get('decode_rng') is not None:
        raise ValueError(
            f'Got RNG both from the `rng` argument ({rng}) and'
            " `decoder_params['decode_rng']`"
            f' ({decoder_params["decode_rng"]}). Please specify one or the'
            ' other.'
        )
      decoder_params['decode_rng'] = rng

    # Using the above-defined single-step decoder function, run temperature
    # sampling with the prefix.
    # [batch, max_decode_length]
    scanned = hasattr(self.module, 'scan_layers') and self.module.scan_layers

    if 'eos_id' not in decoder_params:
      decoder_params['eos_id'] = self.output_vocabulary.eos_id
    decoded_sequences, scores = self._decode_fn(
        inputs=inputs,
        cache=prefilled_cache,
        tokens_to_logits=tokens_ids_to_logits,
        num_decodes=num_decodes,
        initial_index=initial_index,
        cache_offset=1 if scanned else 0,
        **decoder_params,
    )

    if not return_all_decodes:
      # Search returns [n_batch, n_beam/decodes, n_length] with the beam/decode
      # dimension sorted in increasing order of log-probability.
      # `scores` is [batch, beam/decode_size]
      # We take the highest scoring sequence (-1) and its score
      decoded_sequences = decoded_sequences[:, -1, :]
      # Beam search returns []
      aux = {'scores': scores[:, -1]}
    else:
      # We return all samples and scores, rather than just the top ones.
      aux = {'scores': scores}

    return remove_prefix(decoded_sequences, initial_index), aux


@jax.vmap
def remove_prefix(
    sequence: jnp.ndarray, prefix_length: jnp.ndarray
) -> jnp.ndarray:
  """Remove the prefix portion and shift to the left by the prefix length.

  The example below uses non-decorated function definition, i.e., arrays do not
  have batch dimension. `jax.vmap` internally inserts the batch dimension at
  axis=0. The shape annotations do not include the batch dimension either.

  Example:
  ```python
  sequence = [1, 2, 3, 4, 5, 6, 7, 0]
  prefix_length = 2
  remove_prefix(sequence, prefix_length) = [3, 4, 5, 6, 7, 0, 0, 0]
  ```

  Note that this function assumes that the padding token has an id of 0.

  Args:
    sequence: [length] array.
    prefix_length: scalar, i.e., rank 0 array.

  Returns:
    [length] array with the prefix removed and the suffix shifted.
  """
  length = sequence.shape[-1]
  # A binary mask with 1 at inputs.
  inputs_mask = jnp.arange(length) < prefix_length
  # A binary mask with 1 at the targets and padding positions.
  targets_and_padding_mask = jnp.logical_not(inputs_mask).astype(sequence.dtype)
  # Since padding id = 0, the padding mask is zeroed out.
  targets = sequence * targets_and_padding_mask
  # Shift to the left by prefix length. Wrapped elements are already zeroed.
  return jnp.roll(targets, -prefix_length, axis=-1)


# TODO(cpgaffney) Remove this method when dependencies no longer use - rely on
# WeightedAccuracy Metric instead.
def compute_weighted_accuracy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array of categories.
   weights: None or array of shape [batch, length]

  Returns:
    Scalar accuracy.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s targets'
        % (str(logits.shape), str(targets.shape))
    )
  accuracy = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  if weights is not None:
    accuracy = accuracy * weights

  return jnp.sum(accuracy)  # pytype: disable=bad-return-type  # jnp-type


# TODO(cpgaffney) remove when users rely on compute_base_metrics
def compute_metrics(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: jnp.ndarray,
    loss: jnp.ndarray,
    weight_sum: jnp.ndarray,
    additional_metrics: MetricsMap,
) -> MetricsMap:
  """Compute summary metrics."""
  accuracy = compute_weighted_accuracy(logits, targets, weights)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
      'weight_sum': weight_sum,
      'num_examples': targets.shape[0],
      'num_tokens': targets.size,
  }
  metrics = metrics_lib.create_metrics_dict(metrics)
  metrics.update(additional_metrics)
  return metrics


def count_packed_examples(segment_ids: jnp.ndarray) -> int:
  """Return the number of packed examples.

  After packing, each row of segment_ids contains the ids of packed examples.
  For some model inputs, some features could have some examples but not others.
  For example, two tasks in a multimodal setup could be: (1). text -> text, and
  (2). image -> text. Examples from (1) will be missing image input feature and
  examples from (2) will be missing text input feature.

  To count the packed examples, we count the unique ids in segment_ids excluding
  0s (because of padding). It can be implemented by counting the number of
  non-zero values in the first discrete difference along axis=1, plus the number
  of rows in segment_ids, and minus the number of padded examples.

  Example:
    [[1, 1, 3, 3, 0, 0],
     [2, 2, 2, 2, 2, 2],
     [2, 7, 7, 7, 7, 0]] has 5 packed examples.

  Args:
    segment_ids: [B, L] array.

  Returns:
    Scalar count.
  """

  # If there is padding, it's at the end and the id is always 0.
  num_padded_examples = jnp.sum(segment_ids[:, -1] == 0)
  # Get the first discrete different along axis=1.
  first_diff = jnp.diff(segment_ids, n=1, axis=1)
  # count = #(non-0 diff) + #(row) - #(padded ex).
  return jnp.sum(first_diff != 0) + segment_ids.shape[0] - num_padded_examples  # pytype: disable=bad-return-type  # jnp-type


def compute_base_metrics(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    loss: jnp.ndarray,
    z_loss: Optional[jnp.ndarray] = None,
    segment_ids: Optional[Mapping[str, jnp.ndarray]] = None,
) -> MetricsMap:
  """Compute summary metrics.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array of categories.
   mask: None or array of shape [batch, length]. Note: must consist of boolean
     values (float-valued weights not supported).
   loss: loss (float)
   z_loss: z_loss (float)
   segment_ids: Optional dictionary of feature and value is the segment ids used
     for packing, i.e. [batch, length] arrays.

  Returns:
    Dict of metrics.
  """
  num_examples = jnp.array(targets.shape[0])
  num_tokens = jnp.array(targets.size)
  num_devices = jax.device_count()
  assert num_devices, 'JAX is reporting no devices, but it should.'
  # Note: apply mask again even though mask has already been applied to loss.
  # This is needed to divide by mask sum, but should not affect correctness of
  # the numerator.
  nonpadding_tokens = jnp.sum(mask) if mask is not None else targets.size
  metrics = {
      'accuracy': clu_metrics.Accuracy.from_model_output(
          logits=logits, labels=targets.astype(jnp.int32), mask=mask
      ),
      'loss': metrics_lib.AveragePerStep(total=loss),
      'loss_per_nonpadding_target_token': clu_metrics.Average(
          total=loss, count=nonpadding_tokens
      ),
      'loss_per_all_target_tokens': clu_metrics.Average(
          total=loss, count=num_tokens
      ),
      'timing/seqs_per_second': metrics_lib.TimeRate.from_model_output(  # pytype: disable=wrong-arg-types  # jnp-type
          numerator=num_examples
      ),
      'timing/steps_per_second': metrics_lib.StepsPerTime.from_model_output(),
      'timing/seconds': metrics_lib.Time(),
      'timing/seqs': metrics_lib.Sum(num_examples),
      'timing/seqs_per_second_per_core': metrics_lib.TimeRate.from_model_output(  # pytype: disable=wrong-arg-types  # jnp-type
          numerator=num_examples / num_devices
      ),
      'timing/target_tokens_per_second': metrics_lib.TimeRate.from_model_output(  # pytype: disable=wrong-arg-types  # jnp-type
          numerator=num_tokens
      ),
      'timing/target_tokens_per_second_per_core': (
          metrics_lib.TimeRate.from_model_output(  # pytype: disable=wrong-arg-types  # jnp-type
              numerator=num_tokens / num_devices
          )
      ),
      'non_padding_fraction/loss_weights': clu_metrics.Average(
          total=nonpadding_tokens, count=num_tokens
      ),
  }
  if z_loss is not None:
    metrics.update({
        'z_loss': metrics_lib.AveragePerStep(total=z_loss),
        'z_loss_per_all_target_tokens': clu_metrics.Average(
            total=z_loss, count=num_tokens
        ),
        'cross_ent_loss': metrics_lib.AveragePerStep(total=loss - z_loss),
        'cross_ent_loss_per_all_target_tokens': clu_metrics.Average(
            total=jnp.sum(loss - z_loss), count=num_tokens
        ),
    })

  if segment_ids is not None:
    total_tokens = jnp.array(0)
    total_non_padding_tokens = jnp.array(0)
    for feature, feature_segment_ids in segment_ids.items():
      if feature_segment_ids is None or feature_segment_ids.shape[1] == 0:
        continue
      # Since this is [B, L] with the segment ids in axis = 1.
      num_examples = count_packed_examples(feature_segment_ids)
      metrics[f'effective_batch_size/{feature}'] = metrics_lib.AveragePerStep(
          total=num_examples
      )
      # 0s is padding
      feature_non_padding = jnp.sum(feature_segment_ids != 0)
      feature_size = jnp.array(feature_segment_ids.size)
      total_tokens += feature_size
      total_non_padding_tokens += feature_non_padding
      metrics[f'non_padding_fraction/{feature}'] = clu_metrics.Average(
          total=feature_non_padding, count=feature_size
      )
    metrics['non_padding_fraction/overall'] = clu_metrics.Average(
        total=total_non_padding_tokens, count=total_tokens
    )

  return metrics


def get_input_vocabulary(model: BaseTransformerModel) -> seqio.Vocabulary:
  return model.input_vocabulary


def get_output_vocabulary(model: BaseTransformerModel) -> seqio.Vocabulary:
  return model.output_vocabulary
