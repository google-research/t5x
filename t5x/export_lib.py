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

"""Functions for exporting a T5X model."""

import dataclasses
import functools
import inspect
import itertools
import json
import os
import os.path
import typing
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

from absl import logging
from flax.core import frozen_dict
import flax.traverse_util
import jax
from jax.experimental import jax2tf  # type: ignore[import]
import jax.numpy as jnp
import ml_collections
import numpy as np
import seqio
from t5x import checkpoints
from t5x import decoding
from t5x import models
from t5x import partitioning
from t5x import utils
import tensorflow as tf  # type: ignore
import typing_extensions

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2



PyTree = Any
ConfigDict = ml_collections.ConfigDict
DecoderParamsSpec = Sequence[Tuple[str, tf.DType, Sequence[int]]]
PreprocessorFn = Callable[..., Mapping[str, tf.Tensor]]
WarmupExample = Union[Union[str, bytes], List[int]]
WarmupExamples = List[WarmupExample]
PostprocessorFn = Callable[
    [Tuple[Any, Any]], Union[Tuple[Any, Any], Mapping[str, Any]]
]
InferenceFn = Callable[[Mapping[str, Any], Any], PyTree]


class CreatePreprocessorFnNew(typing_extensions.Protocol):

  def __call__(
      self,
      batch_size: Optional[int],
      output_features: Mapping[str, seqio.Feature],
      task_feature_lengths: Mapping[str, int],
      tokenized_inputs: bool,
  ) -> Tuple[PreprocessorFn, Sequence[tf.TensorSpec]]:
    ...


# Old signature, for backwards-compatibility.
# TODO(marcrasi): Delete this after migrating clients.
CreatePreprocessorFnOld = Callable[
    [Mapping[str, seqio.Feature], Mapping[str, int], bool], PreprocessorFn
]
CreatePreprocessorFn = Union[CreatePreprocessorFnOld, CreatePreprocessorFnNew]


@dataclasses.dataclass
class CustomInferenceMode:
  # The name of the model function which can be fetched from
  # getattr(model, model_fn_name).
  model_fn_name: str
  # Fetch useful output from the raw output of the model function.
  fetch_output: Optional[Callable[[PyTree], PyTree]] = None
  # Constant keyword aguments to append when calling the model function.
  model_fn_const_kwargs: None | Mapping[str, Any] = None


class CreatePostprocessorFn(typing_extensions.Protocol):

  def __call__(
      self,
      vocab: seqio.Vocabulary,
      inference_mode: Union[str, CustomInferenceMode],
      decode_outputs: bool = True,
      output_feature_names: Optional[List[str]] = None,
  ) -> PostprocessorFn:
    ...


class CreateDecodingStateCallbackFn(typing_extensions.Protocol):

  def __call__(
      self,
      vocab: seqio.Vocabulary,
      num_decodes: int = 1,
      output_feature_names: Optional[List[str]] = None,
      call_tf_graph: bool = False,
  ) -> decoding.StateCallbackFn:
    ...


class ExportableModule(tf.Module):
  """Wrapper for TF function + parameters to be exported."""

  def __init__(
      self,
      preproc_tf_fn,
      model_tf_fn,
      postproc_tf_fn,
      params: Mapping[str, Any],
      batch_size: Optional[int],
      num_batch_threads: int = 8,
      max_enqueued_batches: int = 64,
      batch_timeout_micros: int = 1000_000,
      max_batch_size: Optional[int] = None,
      allowed_batch_sizes: Optional[Sequence[int]] = None,
      jit_compile: bool = True,
      use_batch_function: bool = False,
      use_gpu: bool = False,
      enable_large_batch_splitting: bool = True,
  ):
    super().__init__()

    def flat_params(params):
      flat_param_vars = {}
      for k, v in flax.traverse_util.flatten_dict(params).items():
        flat_param_vars[k] = tf.Variable(
            np.asarray(v), trainable=False, name='__'.join(k)
        )
      return flat_param_vars

    if use_gpu:
      tf_device = tf.config.list_logical_devices('GPU')[0]
      with tf.device(tf_device):
        flat_param_vars = flat_params(params)
    else:
      flat_param_vars = flat_params(params)
    self._variables = list(flat_param_vars.values())
    param_vars = frozen_dict.freeze(
        flax.traverse_util.unflatten_dict(flat_param_vars)
    )
    self._preproc_tf_fn = preproc_tf_fn
    self._postproc_tf_fn = postproc_tf_fn

    # TF trackable resources must be assigned to an attribute of the module.
    # TODO(dinghua): We should have a more formal API for getting the
    #                trackable members from pre/post-processing functions.
    self._other_trackables = []
    for fn in (self._preproc_tf_fn, self._postproc_tf_fn):
      if hasattr(fn, 'trackable_resources'):
        self._other_trackables.append(fn.trackable_resources)

    # Note: jit_compile=True also instructs the TPU inference converter v2 to
    # wrap this function with `TPUPartitionedCall`.
    self._model_tf_fn = tf.function(
        lambda x: model_tf_fn(param_vars, x),
        autograph=False,
        jit_compile=jit_compile,
    )
    self._batch_size = batch_size
    self._num_batch_threads = num_batch_threads
    self._max_enqueued_batches = max_enqueued_batches
    self._batch_timeout_micros = batch_timeout_micros
    self._allowed_batch_sizes = allowed_batch_sizes
    self._use_batch_function = use_batch_function
    self._max_batch_size = max_batch_size
    self._enable_large_batch_splitting = enable_large_batch_splitting

  @functools.partial(tf.function, autograph=False, jit_compile=False)
  def __call__(self, *input_batches) -> Tuple[Any, Any]:
    if not self._use_batch_function:
      return self._call(*input_batches)

    if self._allowed_batch_sizes:
      if self._batch_size is not None:
        raise ValueError('allowed_batch_sizes requires polymorphic batch size')
      max_batch_size = self._max_batch_size or max(self._allowed_batch_sizes)
      allowed_batch_sizes = self._allowed_batch_sizes
    elif self._batch_size is not None:
      max_batch_size = self._max_batch_size or self._batch_size
      allowed_batch_sizes = [self._batch_size]
    else:
      raise ValueError(
          'Need to set either batch_size or allowed_batch_sizes when '
          'using batch_function.'
      )
    batch_wrapper = tf.nondifferentiable_batch_function(
        num_batch_threads=self._num_batch_threads,
        max_enqueued_batches=self._max_enqueued_batches,
        max_batch_size=max_batch_size,
        batch_timeout_micros=self._batch_timeout_micros,
        allowed_batch_sizes=allowed_batch_sizes,
        enable_large_batch_splitting=self._enable_large_batch_splitting,
    )
    flattended, tree_def = jax.tree_util.tree_flatten(input_batches)
    return batch_wrapper(functools.partial(self._call, tree_def=tree_def))(
        *flattended
    )

  def _call(self, *args, tree_def=None):
    if tree_def is not None:
      input_batches = jax.tree_util.tree_unflatten(tree_def, args)
    else:
      input_batches = args
    features = self._preproc_tf_fn(*input_batches)
    model_output = self._model_tf_fn(features)
    return self._postproc_tf_fn(model_output)

  @property
  def tpu_func(self):
    return self._model_tf_fn

  @property
  def export_batch_sizes(self):
    return self._allowed_batch_sizes or [self._batch_size]

  @tf.function(autograph=False, jit_compile=False)
  def preproc_func(self, *args):
    return self._preproc_tf_fn(*args)


def get_train_state_initializer(
    model: models.BaseTransformerModel,
    partitioner: partitioning.BasePartitioner,
    task_feature_lengths: Mapping[str, int],
    batch_size: Optional[int],
    trailing_shapes: Optional[Mapping[str, Tuple[int, ...]]] = None,
) -> Optional[utils.TrainStateInitializer]:
  """Creates an TrainStateInitializer based on the model and partitioning."""
  if not partitioner:
    return None

  data_layout = partitioner.get_data_layout(batch_size)
  p_batch_size = data_layout.batch_size
  feature_converter = model.FEATURE_CONVERTER_CLS(pack=False)
  model_feature_lengths = feature_converter.get_model_feature_lengths(
      task_feature_lengths
  )
  input_shapes = {}
  for k, l in model_feature_lengths.items():
    input_shapes[k] = (p_batch_size, l)
    if feature_converter.MODEL_FEATURES[k].rank > 1:
      if trailing_shapes is None or k not in trailing_shapes:
        raise ValueError(
            'Must set the trailing shape--`...?` in '
            '`(batch_size, seqlen, ...?)`--for higher rank '
            f'feature {k}'
        )
      input_shapes[k] += trailing_shapes[k]
  train_state_initializer = utils.TrainStateInitializer(
      optimizer_def=None,
      init_fn=model.get_initial_variables,
      input_shapes=input_shapes,
      partitioner=partitioner,
  )
  utils.log_model_info(
      None, train_state_initializer.global_train_state_shape, partitioner
  )
  return train_state_initializer


def flatten(
    compute_outputs: PyTree, assert_output_len=None
) -> Tuple[jnp.ndarray, ...]:
  values, _ = jax.tree_util.tree_flatten(compute_outputs)
  if assert_output_len is not None:
    assert len(values) == assert_output_len
  return tuple(values)


_BUILTIN_INFERENCE_MODE_PARAMS = {
    'predict': {
        'model_fn_name': 'predict_batch_with_aux',
        'default_output_len': 2,
    },
    'score': {'model_fn_name': 'score_batch', 'default_output_len': 1},
}


def create_inference_function(
    *,
    model: models.BaseTransformerModel,
    inference_mode: Union[str, CustomInferenceMode],
    partitioner: Optional[partitioning.BasePartitioner],
    train_state_initializer: Optional[utils.TrainStateInitializer],
    decoding_state_callback_fn: Optional[decoding.StateCallbackFn] = None,
    enable_jax2tf: bool,
    enable_xla: bool = True,
    polymorphic_shapes_inputs: Optional[Any] = None,
    native_lowering: bool = True,
    native_lowering_platforms: Optional[Sequence[str]] = None,
    model_fn_extra_kwargs: Optional[Mapping[str, Any]] = None,
    jax2tf_disable_platform_checks: bool = False,
    output_len: Optional[int] = None,
) -> Callable[[Mapping[str, Any], Any], PyTree]:
  """Fetches a model and returns the inference function based on inference_mode."""
  if partitioner and train_state_initializer:
    maybe_partition = lambda fn: partitioner.partition(  # pylint:disable=g-long-lambda
        fn,
        # TODO(b/121310741): Re-enable pytype.
        # pytype:disable=wrong-arg-types
        in_axis_resources=(
            train_state_initializer.train_state_axes.params,
            partitioning.PartitionSpec(
                'data',
            ),
        ),
        out_axis_resources=partitioning.PartitionSpec(
            'data',
        ),
        # pytype:enable=wrong-arg-types
    )

  else:
    maybe_partition = lambda fn: fn

  if not isinstance(inference_mode, CustomInferenceMode):
    if inference_mode not in _BUILTIN_INFERENCE_MODE_PARAMS:
      raise ValueError(
          '`inference_mode` must be a string in '
          f'{list(_BUILTIN_INFERENCE_MODE_PARAMS.keys())} or a '
          f'`CustomInferenceMode`. Got inference_mode={inference_mode}.'
      )
    default_mode_params = _BUILTIN_INFERENCE_MODE_PARAMS[inference_mode]
    assert_output_len = (
        default_mode_params['default_output_len']
        if output_len is None
        else output_len
    )
    inference_mode = CustomInferenceMode(
        default_mode_params['model_fn_name'],
        fetch_output=functools.partial(
            flatten, assert_output_len=assert_output_len
        ),
    )

  inference_mode = typing.cast(CustomInferenceMode, inference_mode)

  if inference_mode.model_fn_name == 'predict_batch_with_aux':
    # Extract `decoder_params` passed by the preprocessor. Decoder params are
    # supported only for `predict_batch_with_aux`.
    #
    # TODO(b/256173604): Make the following Gin-configurable.

    def model_fn(
        params: Mapping[str, Any], inputs: Mapping[str, jnp.ndarray]
    ) -> Tuple[Any, Any]:
      batch = dict(inputs)

      decoder_params = batch.pop('decoder_params', {})
      if decoding_state_callback_fn is not None:
        decoder_params['state_callback_fn'] = decoding_state_callback_fn

      kwargs = {}
      if decoder_params:
        kwargs['decoder_params'] = decoder_params
      if model_fn_extra_kwargs:
        kwargs.update(model_fn_extra_kwargs)
      # pytype: disable=wrong-keyword-args
      return model.predict_batch_with_aux(params, batch, **kwargs)
      # pytype: enable=wrong-keyword-args

  else:
    model_fn = getattr(model, inference_mode.model_fn_name)

  if inference_mode.model_fn_const_kwargs:
    model_fn = functools.partial(
        model_fn, **inference_mode.model_fn_const_kwargs
    )

  model_fn = maybe_partition(model_fn)
  if enable_jax2tf:
    disabled_checks = (
        [jax2tf.DisabledSafetyCheck.platform()]
        if jax2tf_disable_platform_checks
        else []
    )
    if native_lowering and (not native_lowering_platforms):
      # Change default value to make the exported cpu model still work.
      native_lowering_platforms = ['cpu', 'tpu']
    model_fn = jax2tf.convert(
        model_fn,
        polymorphic_shapes=[None, polymorphic_shapes_inputs],
        native_serialization=native_lowering,
        native_serialization_platforms=native_lowering_platforms,
        native_serialization_disabled_checks=disabled_checks,
        enable_xla=enable_xla,
    )

  def inference_fn(
      params: Mapping[str, Any], batch: Mapping[str, jnp.ndarray]
  ) -> PyTree:
    outputs = model_fn(params, batch)
    if inference_mode.fetch_output:
      outputs = inference_mode.fetch_output(outputs)
    return outputs

  return inference_fn


def load_params_from_checkpoint(
    restore_checkpoint_cfg: utils.RestoreCheckpointConfig,
    train_state_initializer: Optional[utils.TrainStateInitializer],
    partitioner: partitioning.BasePartitioner,
) -> frozen_dict.FrozenDict:
  """Loads the checkpoint and casts the variable."""
  if train_state_initializer is not None:
    restore_cfg, ckpt_paths = utils.get_first_valid_restore_config_and_paths(
        [restore_checkpoint_cfg]
    )
    if len(ckpt_paths) != 1:
      raise ValueError(
          f'Expected only 1 checkpoint but got {len(ckpt_paths)} for '
          f'config(s): {restore_cfg}'
      )
    train_state, _ = utils.create_checkpoint_manager_and_restore(
        train_state_initializer=train_state_initializer,
        partitioner=partitioner,
        restore_checkpoint_cfg=restore_cfg,
        restore_path=ckpt_paths[0],
        fallback_init_rng=jax.random.PRNGKey(0),
        save_checkpoint_cfg=None,
        model_dir=None,
        ds_iter=None,
        use_orbax=True,
    )
    return train_state.params  # pytype:disable=attribute-error
  else:
    if restore_checkpoint_cfg.mode != 'specific':
      raise NotImplementedError("Only mode='specific' is currently supported")
    if not isinstance(restore_checkpoint_cfg.path, str):
      raise NotImplementedError('Only string paths are currently supported')
    variables = checkpoints.load_t5x_checkpoint(
        path=restore_checkpoint_cfg.path,
        state_transformation_fns=(
            restore_checkpoint_cfg.state_transformation_fns
        ),
        restore_dtype=jnp.dtype(restore_checkpoint_cfg.dtype),
    )
    return frozen_dict.freeze(variables['target'])


def create_single_tensor_input_signature(
    batch_size: Optional[int],
    task_feature_lengths: Mapping[str, int],
    tokenized_inputs: bool = False,
    name='text_batch',
) -> Sequence[tf.TensorSpec]:
  """Returns an input signature for a model that takes a single input tensor.

  Args:
    batch_size: Batch size for model to process. If None, then batch
      polymorphism is invoked.
    task_feature_lengths: Mapping from 'inputs' and 'targets' to sequence
      lengths.
    tokenized_inputs: specifies whether the input is expected to be
      pre-tokenized. If so, the preprocessor expects an int32 tensor of shape
      [B, N] rather than a string tensor of shape [B].
    name: the name of the single `tf.TensorSpec` in the input signature.
  """
  if tokenized_inputs:
    inputs_length = task_feature_lengths['inputs']
    return (tf.TensorSpec([batch_size, inputs_length], tf.int32, name=name),)
  else:
    return (tf.TensorSpec([batch_size], tf.string, name=name),)


def bucketize_tokenized_input(
    input_tensor: tf.Tensor, bucket_keys: List[int]
) -> tf.Tensor:
  """Returns a scalar tf.Tensor indicates the bucketized length of the input_tensor.

  Args:
    input_tensor: A tokenized tf.Tensor.
    bucket_keys: a bucket of sequence lengths, sorted ascendingly.
  """
  index = tf.searchsorted(bucket_keys, [tf.shape(input_tensor)[0]])
  index = tf.math.minimum(index, len(bucket_keys) - 1)
  index = tf.squeeze(index)
  return tf.constant(bucket_keys)[index]


def truncate_and_pad_tokenized_input(
    input_tensor: tf.Tensor,
    max_length: int,
    allowed_lengths: Optional[list[int]] = None,
) -> tf.Tensor:
  """Truncate the examples to the maximum feature length, and pad to bucketed sequence lengths."""
  if allowed_lengths:
    allowed_lengths.sort()
    if max_length != allowed_lengths[-1]:
      raise ValueError(
          'Expected the largest allowed length to be the same as the task'
          f' feature length {max_length}, but got {allowed_lengths[-1]}.'
      )
    max_length = bucketize_tokenized_input(input_tensor, allowed_lengths)
  input_tensor = input_tensor[:max_length]
  input_tensor = tf.pad(
      input_tensor, [[0, max_length - tf.shape(input_tensor)[0]]]
  )
  if not allowed_lengths:
    input_tensor.set_shape([max_length])
  return input_tensor


def _build_ragged_feature_length_spec(
    feature_key: str,
    bucket_keys: Optional[Mapping[str, list[int]]],
    task_feature_lengths: Mapping[str, int],
) -> tf.RaggedTensorSpec:
  """Build a spec of the feature tensor's sequence length after preprocessing.

  The sequence length will be dynamic when polymorphic sequence length buckets
  are configured for the given feature.
  Args:
    feature_key: The name of an input feature tensor.
    bucket_keys: A mapping from feature key to allowed sequence lengths.
    task_feature_lengths: A mapping from feature key to fixed sequence length.

  Returns:
    A RaggedTensorSpec representing the sequence length dimension of the
    feature tensor.
  """
  if bucket_keys and feature_key in bucket_keys:
    # Polymorphic sequence length.
    feature_length = None
  else:
    feature_length = task_feature_lengths[feature_key]
  return tf.RaggedTensorSpec(
      shape=[feature_length],
      dtype=tf.int32,
      ragged_rank=0,
  )


# TODO(danielandor): More principled score-mode input format.
def create_preprocessor(
    batch_size: Optional[int],
    output_features: Mapping[str, seqio.Feature],
    task_feature_lengths: Mapping[str, int],
    tokenized_inputs: bool = False,
    *,
    input_tensor_name: str = 'text_batch',
    split_separator: Optional[str] = None,
    bucket_keys: Optional[Mapping[str, List[int]]] = None,
) -> Tuple[PreprocessorFn, Sequence[tf.TensorSpec]]:
  """Builds a function based on the config task to tokenize and batch the input text.

  Args:
    batch_size: Batch size for model to process. If None, then batch
      polymorphism is invoked.
    output_features: Mapping from 'inputs' and 'targets' to seqio.Feature.
    task_feature_lengths: Mapping from 'inputs' and 'targets' to sequence
      lengths.
    tokenized_inputs: specifies whether the input is expected to be
      pre-tokenized. If so, the preprocessor expects an int32 tensor of shape
      [B, N] rather than a string tensor of shape [B].
    input_tensor_name: the name of the input tensor.
    split_separator: If given, splits the input text at the first separator, and
      sets the target text for scoring to the second element. If None, the
      target is set to the empty string. The latter is appropriate for predict
      mode.
    bucket_keys: If given, bucketizes the tensors according to bucket_keys.

  Returns:
    The preprocessor function.
  """

  def preprocess(input_texts: tf.Tensor) -> Mapping[str, tf.Tensor]:
    """TF-based preprocessor that takes a batch of text and converts it to model features."""
    if tokenized_inputs:
      inputs = input_texts  # actually an int32 tensor of shape [B, N].
      targets = tf.broadcast_to(
          tf.constant(0, dtype=tf.int32), tf.shape(input_texts)
      )
    elif split_separator is None:
      inputs = input_texts
      targets = tf.broadcast_to(tf.constant(''), tf.shape(input_texts))
    else:
      ragged_split = tf.strings.split(
          input_texts, sep=split_separator, maxsplit=1
      )
      split = ragged_split.to_tensor(shape=[tf.shape(input_texts)[0], 2])
      inputs, targets = split[:, 0], split[:, 1]

    # TODO(b/188656799): Generalize this code to work with arbitrary models.
    def featurize(text, k):
      """Replicates what tokenization + seqio.EncDecFeatureConverter does, without Dataset."""
      vocab = output_features[k].vocabulary  # type: seqio.Vocabulary
      length = task_feature_lengths[k]
      if not tokenized_inputs:  # if inputs are tokenized, we don't re-tokenize.
        t = vocab.encode_tf(text)
      else:
        t = text
      if output_features[k].add_eos:
        # The following matches the default behavior of the prediction server,
        # which uses seqio.preprocessors.append_eos_after_trim, implemented at:
        # https://github.com/google/seqio/tree/main/seqio/preprocessors.py;l=250;rcl=480228505
        t = tf.concat([t[: length - 1], [vocab.eos_id]], axis=0)
      allowed_lengths = bucket_keys.get(k) if bucket_keys else None
      t = truncate_and_pad_tokenized_input(
          t, max_length=length, allowed_lengths=allowed_lengths
      )
      ar_inputs = seqio.feature_converters.autoregressive_inputs(t)
      loss_weights = seqio.feature_converters.non_padding_position(t)
      return t, ar_inputs, loss_weights

    encoder_output_signature = _build_ragged_feature_length_spec(
        'inputs', bucket_keys, task_feature_lengths
    )
    encoder_input_tokens, _, _ = tf.map_fn(
        functools.partial(featurize, k='inputs'),
        inputs,
        fn_output_signature=(encoder_output_signature,) * 3,
    )
    encoder_input_tokens = encoder_input_tokens.to_tensor()

    decoder_output_signature = _build_ragged_feature_length_spec(
        'targets', bucket_keys, task_feature_lengths
    )
    decoder_target_tokens, decoder_input_tokens, loss_weights = tf.map_fn(
        functools.partial(featurize, k='targets'),
        targets,
        fn_output_signature=(decoder_output_signature,) * 3,
    )
    decoder_target_tokens = decoder_target_tokens.to_tensor()
    decoder_input_tokens = decoder_input_tokens.to_tensor()
    loss_weights = loss_weights.to_tensor()

    return dict(
        encoder_input_tokens=encoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_loss_weights=loss_weights,
    )

  input_signature = create_single_tensor_input_signature(
      batch_size, task_feature_lengths, tokenized_inputs, input_tensor_name
  )
  return preprocess, input_signature


def create_dual_encoder_preprocessor(
    batch_size: Optional[int],
    output_features: Mapping[str, seqio.Feature],
    task_feature_lengths: Mapping[str, int],
    tokenized_inputs: bool = False,
    input_tensor_name: str = 'text_batch',
    bucket_keys: Optional[Mapping[str, List[int]]] = None,
    split_separator: Optional[str] = None,
) -> Tuple[PreprocessorFn, Sequence[tf.TensorSpec]]:
  """Builds a function based on the config task to tokenize and batch the input text."""

  def preprocess(input_texts: tf.Tensor) -> Mapping[str, tf.Tensor]:
    """TF-based preprocessor that takes a batch of text and converts it to model features."""
    if tokenized_inputs:
      inputs = input_texts
      targets = tf.broadcast_to(
          tf.constant(0, dtype=tf.int32), tf.shape(input_texts)
      )
    elif split_separator is None:
      inputs = input_texts
      targets = tf.broadcast_to(tf.constant(''), tf.shape(input_texts))
    else:
      ragged_split = tf.strings.split(
          input_texts, sep=split_separator, maxsplit=1
      )
      split = ragged_split.to_tensor(shape=[tf.shape(input_texts)[0], 2])
      inputs, targets = split[:, 0], split[:, 1]

    # TODO(b/188656799): Generalize this code to work with arbitrary models.
    def featurize(text, k):
      """Replicates what tokenization + nlp.nlx.t5x_retrieval.DualEncoderFeatureConverter does, without Dataset."""
      vocab = output_features[k].vocabulary  # type: seqio.Vocabulary
      length = task_feature_lengths[k]
      if not tokenized_inputs:  # if inputs are tokenized, we don't re-tokenize.
        t = vocab.encode_tf(text)
      else:
        t = text
      if output_features[k].add_eos:
        t = tf.concat([t[: length - 1], [vocab.eos_id]], axis=0)
      allowed_lengths = bucket_keys.get(k) if bucket_keys else None
      t = truncate_and_pad_tokenized_input(
          t, max_length=length, allowed_lengths=allowed_lengths
      )
      return t

    left_encoder_input_tokens = tf.map_fn(
        functools.partial(featurize, k='inputs'),
        inputs,
        fn_output_signature=_build_ragged_feature_length_spec(
            'inputs', bucket_keys, task_feature_lengths
        ),
    ).to_tensor()

    right_encoder_input_tokens = tf.map_fn(
        functools.partial(featurize, k='targets'),
        targets,
        fn_output_signature=_build_ragged_feature_length_spec(
            'targets', bucket_keys, task_feature_lengths
        ),
    ).to_tensor()

    return dict(
        left_encoder_input_tokens=left_encoder_input_tokens,
        right_encoder_input_tokens=right_encoder_input_tokens,
    )

  input_signature = create_single_tensor_input_signature(
      batch_size, task_feature_lengths, tokenized_inputs, input_tensor_name
  )
  return preprocess, input_signature


def create_decoder_preprocessor(
    output_features: Mapping[str, seqio.Feature],
    task_feature_lengths: Mapping[str, int],
    tokenized_inputs: bool = False,
    input_feature: str = 'inputs',
) -> PreprocessorFn:
  """Returns a function to tokenize and featurize inputs for decoder only models.

  Args:
    output_features: Mapping from 'inputs' and 'targets' to seqio.Feature.
    task_feature_lengths: Mapping from 'inputs' and 'targets' to sequence
      lengths.
    tokenized_inputs: specifies whether the input is expected to be
      pre-tokenized. If so, the preprocessor expects an int32 tensor padded with
      0s to shape of [B, N] rather than a string tensor of shape [B].
    input_feature: Name of the feature provided by `input_texts`, e.g., 'inputs'
      or 'targets'.
  """

  def preprocess(input_texts: tf.Tensor) -> Mapping[str, tf.Tensor]:
    """TF-based preprocessor that takes a batch of text and converts it to model features."""

    def tokenize(text):
      feature = output_features[input_feature]
      vocab = feature.vocabulary  # type: seqio.Vocabulary
      if not tokenized_inputs:  # if inputs are tokenized, we don't re-tokenize.
        t = vocab.encode_tf(text)
      else:
        t = text
      if feature.add_eos:
        t = tf.concat([t, [vocab.eos_id]], axis=-1)
      return t

    decoder_tokens = tf.map_fn(
        tokenize,
        input_texts,
        fn_output_signature=(tf.int32),
    )

    if input_feature == 'inputs':
      # 'inputs_width' is the length of 'inputs' (excluding padding 0).
      ragged_input_tokens = tf.RaggedTensor.from_tensor(
          decoder_tokens, padding=0
      )
      inputs_length = tf.cast(ragged_input_tokens.row_lengths(), dtype=tf.int32)
      inputs_width = tf.expand_dims(inputs_length, -1)
      inputs_width_add_pos = inputs_width + 1
    else:
      inputs_width = tf.zeros(tf.shape(decoder_tokens)[0], dtype=tf.int32)
      inputs_width_add_pos = inputs_width

    def featurize(text, length):
      text = text[:length]
      text = tf.pad(text, [[0, length - tf.shape(text)[0]]])
      text.set_shape([length])
      ar_inputs = seqio.feature_converters.autoregressive_inputs(text)
      loss_weights = seqio.feature_converters.non_padding_position(text)

      return text, ar_inputs, loss_weights

    targets_length = sum(task_feature_lengths.values())
    decoder_target_tokens, decoder_input_tokens, decoder_loss_weights = (
        tf.map_fn(
            functools.partial(featurize, length=targets_length),
            decoder_tokens,
            fn_output_signature=(tf.int32, tf.int32, tf.int32),
        )
    )
    positions = tf.range(tf.shape(decoder_target_tokens)[-1])
    positions = tf.repeat(
        [positions], tf.shape(decoder_target_tokens)[0], axis=0
    )

    decoder_causal_attention = tf.cast(
        positions < inputs_width_add_pos, dtype=decoder_target_tokens.dtype
    )

    inputs = positions < inputs_width
    padding_mask = tf.cast(decoder_loss_weights, dtype=tf.bool)

    decoder_loss_weights = tf.cast(
        tf.math.logical_xor(inputs, padding_mask),
        dtype=decoder_target_tokens.dtype,
    )

    return dict(
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        decoder_loss_weights=decoder_loss_weights,
        decoder_causal_attention=decoder_causal_attention,
    )

  return preprocess


def _default_value_for_spec(v):
  return tf.zeros(v.shape, v.dtype).numpy()


def _feature_description_from_element_spec(element_spec):
  """Feature description from element spec."""
  feature_description = {}
  for k, v in element_spec.items():
    if isinstance(v, tf.SparseTensorSpec):
      feature_description[k] = tf.io.VarLenFeature(dtype=v.dtype)
    elif isinstance(v, tf.TensorSpec):
      if v.shape.is_fully_defined():
        feature_description[k] = tf.io.FixedLenFeature(
            shape=v.shape,
            dtype=v.dtype,
            default_value=_default_value_for_spec(v),
        )
      else:
        if v.shape[0] is None and v.shape[1:].is_fully_defined():
          # We only parse single examples (not batches) so the
          # FixeLenSequenceFeature will never need to add padding through
          # `default_value`.
          feature_description[k] = tf.io.FixedLenSequenceFeature(
              shape=v.shape[1:], dtype=v.dtype, allow_missing=True
          )
        else:
          raise ValueError(
              'Except for the first dimension, all dimentions of shape for '
              f'feature {k} need to be known but received {v.shape!s}.'
          )
    else:
      raise ValueError(
          f'Cannot generate feature description for feature "{k}" with '
          f'element spec type {type(v)}; '
          'supported types: tf.SparseTensorSpec, tf.TensorSpec.'
      )
  return feature_description


class PreprocessorFnFromTask(object):
  """A PreprocessorFn based on seqio.Task."""

  def __init__(
      self,
      batch_size: Optional[int],
      model: models.BaseTransformerModel,
      task_feature_lengths: Mapping[str, int],
      task_name: str = '',
      serialized_examples: bool = True,
      run_precache: bool = False,
  ):
    self.task = seqio.TaskRegistry.get(task_name)
    if serialized_examples:
      ds = self.task.source.get_dataset(self.task.splits[0])
      feature_description = _feature_description_from_element_spec(
          ds.element_spec
      )
      self.parse_example = functools.partial(
          tf.io.parse_single_example, features=feature_description
      )
    else:
      self.parse_example = lambda x: x

    self.feature_converter = model.FEATURE_CONVERTER_CLS(pack=False)
    self.task_feature_lengths = task_feature_lengths
    self.batch_size = batch_size
    self.run_precache = run_precache

    def is_trackable_resource(x):
      return isinstance(x, tf.saved_model.experimental.TrackableResource)

    self.trackable_resources = list()
    for p in self.task.preprocessors:
      # TODO(dinghua): We should have a more formal API for getting the
      #                trackable members from a seqio preprocessor.
      for _, tr in inspect.getmembers(p, is_trackable_resource):
        self.trackable_resources.append(tr)

  def process_fn(self, examples: tf.Tensor) -> Mapping[str, tf.Tensor]:
    """Converts serialized tf.Examples to batched model features.

    Args:
      examples: batch examples. If `self.batch_size` is not None,
        `examples.shape[0]` must be the same as `self.batch_size`.

    Returns:
      A Mapping from feature names to batch features.
    """
    ds = tf.data.Dataset.from_tensor_slices(examples)
    # Dataset of parsed tf Examples.
    ds = ds.map(self.parse_example)
    if self.run_precache:
      ds = self.task.preprocess_precache(ds)
    ds = self.task.preprocess_postcache(ds, self.task_feature_lengths)
    # Dataset of batched model features.
    ds = self.feature_converter(
        ds, task_feature_lengths=self.task_feature_lengths
    )
    # Assume all batch size are the same.
    single_feature = jax.tree_util.tree_leaves(examples)[0]
    if self.batch_size is not None:
      single_feature.shape[:1].assert_is_compatible_with([self.batch_size])
      ds = ds.batch(self.batch_size, drop_remainder=True)
    else:
      batch_size = tf.cast(tf.shape(single_feature)[0], dtype=tf.int64)
      ds = ds.batch(batch_size, drop_remainder=True)
    # As we process one batch at a time, the dataset ds has a single batch.
    return ds.get_single_element()

  def __call__(self, examples: tf.Tensor) -> Mapping[str, tf.Tensor]:
    return self.process_fn(examples)


def create_preprocessor_from_task(
    batch_size: Optional[int],
    output_features: Mapping[str, seqio.Feature],  # unused
    task_feature_lengths: Mapping[str, int],
    tokenized_inputs: bool,
    *,
    model: models.BaseTransformerModel,
    task_name: str = '',
    serialized_examples: bool = True,
    run_precache: bool = False,
    input_tensor_name: str = 'text_batch',
) -> Tuple[PreprocessorFn, Sequence[tf.TensorSpec]]:
  """Create a preprocessor based on a seqio task."""
  del output_features
  return PreprocessorFnFromTask(
      batch_size,
      model,
      task_feature_lengths,
      task_name,
      serialized_examples,
      run_precache,
  ), create_single_tensor_input_signature(
      batch_size, task_feature_lengths, tokenized_inputs, input_tensor_name
  )


def create_preprocessor_with_decoder_params(
    batch_size: Optional[int],
    output_features: Mapping[str, seqio.Feature],  # unused
    task_feature_lengths: Mapping[str, int],
    tokenized_inputs: bool,
    *,
    create_preprocessor_fn: CreatePreprocessorFn,
    decoder_params_spec: DecoderParamsSpec,
) -> Tuple[PreprocessorFn, Sequence[tf.TensorSpec]]:
  """Creates a preprocessor and adds decoder params as inputs.

  Args:
    batch_size: See `save`.
    output_features: See `save`.
    task_feature_lengths: See `save`.
    tokenized_inputs: See `save`.
    create_preprocessor_fn: A function that creates a preprocessor to be
      wrapped.
    decoder_params_spec: A sequence of `(name, dtype, per_example_shape)` for
      decoder params to be exposed as inputs. The decoder must be able to accept
      the listed decoder params on a per-example basis, i.e., the shape of each
      decoder param will be [batch_size, *per_example_shape]. Decoder params are
      appended to the inputs in the specified order.

  Returns:
    A preprocessor that calls `create_preprocessor_fn(...)` with additional
    inputs representing decoder params and adds the specified `decoder_params`
    as a new feature.
  """

  # TODO(marcrasi): Delete after migrating clients.
  if 'batch_size' in inspect.signature(create_preprocessor_fn).parameters:
    # New signature.
    preprocessor, input_signature = create_preprocessor_fn(
        batch_size, output_features, task_feature_lengths, tokenized_inputs
    )  # type: ignore
  else:
    # Old signature.
    preprocessor = create_preprocessor_fn(
        output_features, task_feature_lengths, tokenized_inputs
    )  # type: ignore
    input_signature = create_single_tensor_input_signature(
        batch_size, task_feature_lengths, tokenized_inputs
    )

  def wrapped(*args: tf.Tensor) -> Mapping[str, tf.Tensor]:
    # Splice the args into inputs and decoder params.
    num_decoder_params = len(decoder_params_spec)
    decoder_params_values = args[-num_decoder_params:]
    inputs = args[:-num_decoder_params]

    features = dict(preprocessor(*inputs))

    # Add decoder params as additional features. They are removed from the
    # features dict in `create_inference_function`.
    decoder_params = {}
    for (name, _, _), value in zip(decoder_params_spec, decoder_params_values):
      decoder_params[name] = value
    features['decoder_params'] = decoder_params

    return features

  input_signature = tuple(input_signature) + tuple(
      tf.TensorSpec((batch_size,) + tuple(per_example_shape), dtype, name=name)
      for name, dtype, per_example_shape in decoder_params_spec
  )
  return wrapped, input_signature


def _maybe_name_outputs(
    feature_values: Tuple[Any, ...], feature_names: Optional[List[str]]
) -> Union[Tuple[Any, ...], Mapping[str, Any]]:
  """Names the output features if feature_names are specified."""
  if feature_names is None:
    # Even in single arg case, the returned sequence is going to make sure that
    # we have consistent behaviors.
    return feature_values
  if len(feature_values) != len(feature_names):
    raise ValueError(
        f'Output feature names {feature_names} must match '
        f'number of outputs {len(feature_values)}'
    )
  return dict(zip(feature_names, feature_values))


def create_postprocessor(
    vocab: seqio.Vocabulary,
    inference_mode: Union[str, CustomInferenceMode],
    decode_outputs: bool = True,
    output_feature_names: Optional[List[str]] = None,
    add_token_length_to_output: Optional[bool] = False,
) -> PostprocessorFn:
  """Creates a TF postprocessor function.

  Args:
    vocab: The vocab to use to decode.
    inference_mode: 'predict', 'score' or a CustomInferenceMode instance.
    decode_outputs: whether to decode output tokens.
    output_feature_names: A list of names to name the output for the savedmodel.
      e.g., ['output_a', 'output_b'] will tag the savedmodel output to obtain
      two entries with 'output_a' and 'output_b'. The order must match the
      outputs from the module.
    add_token_length_to_output: A boolean to indicate whether to include token
      length infomation in the output. If True, will append 'num_tokens' and
      'num_tokens_before_eos' to the output. 'num_tokens' indicates the length
      of the decoder's output.  'num_tokens_before_eos' will count the tokens
      until the first 'vocab.eos_id' (excluding the 'eos_id'), or the same as
      'num_tokens' if 'vocab.eos_id' is None.

  Returns:
    A function that that post processing on inference outputs.
  """
  if inference_mode == 'predict':

    def postprocessor(
        values: Tuple[Any, Any]
    ) -> Union[Tuple[Any, Any], Mapping[str, Any]]:
      tokens, scores = values
      if decode_outputs:
        decoded = vocab.decode_tf(tokens)
        if add_token_length_to_output:
          num_tokens = tf.reduce_sum(
              tf.ones_like(tokens, dtype=tf.int32), axis=-1
          )
          if vocab.eos_id:
            after_eos = tf.cumsum(
                tf.cast(tf.equal(tokens, vocab.eos_id), tf.int32),
                axis=-1,
            )
            before_eos = tf.cast(
                tf.logical_not(tf.cast(after_eos, tf.bool)), tf.int32
            )
            num_tokens_before_eos = tf.reduce_sum(before_eos, axis=-1)
          else:
            num_tokens_before_eos = num_tokens
          if isinstance(decoded, tf.RaggedTensor):
            decoded = decoded.to_tensor()
          return _maybe_name_outputs(
              feature_values=(
                  decoded,
                  scores,
                  num_tokens,
                  num_tokens_before_eos,
              ),
              feature_names=output_feature_names,
          )
        # If add_eos=False, vocab.decode_tf returns a tf.Tensor rather than
        # a tf.RaggedTensor.
        if isinstance(decoded, tf.RaggedTensor):
          decoded = decoded.to_tensor()
        return _maybe_name_outputs(
            feature_values=(decoded, scores), feature_names=output_feature_names
        )
      else:
        return _maybe_name_outputs(
            feature_values=(tokens, scores), feature_names=output_feature_names
        )

    return postprocessor
  else:
    return functools.partial(
        _maybe_name_outputs, feature_names=output_feature_names
    )




def _request_for_batch(
    text_batch: WarmupExamples,
    model_name: str,
    input_tensor_name: str,
    signature_name: str,
    batch_size: Optional[int],
    decoder_params_spec: Optional[DecoderParamsSpec] = None,
    input_tensor_dtype: Optional[tf.DType] = None,
) -> predict_pb2.PredictRequest:
  """Adds a single batch of Predict warmup data."""
  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  request.model_spec.signature_name = signature_name
  if input_tensor_dtype is not None:
    dtype = input_tensor_dtype
  elif text_batch and isinstance(text_batch[0], (str, bytes)):
    dtype = tf.string
  else:
    dtype = tf.int32
  # Truncate/Pad the request to have batch_size.
  adjusted_batch = text_batch
  if batch_size is not None:
    adjusted_batch = list(
        itertools.islice(itertools.cycle(text_batch), batch_size)
    )
  request.inputs[input_tensor_name].CopyFrom(
      tf.make_tensor_proto(adjusted_batch, dtype=dtype)
  )
  if decoder_params_spec is not None:
    for name, dtype, per_example_shape in decoder_params_spec:
      request.inputs[name].CopyFrom(
          tf.make_tensor_proto(
              tf.zeros((len(adjusted_batch),) + tuple(per_example_shape), dtype)
          )
      )
  return request


def _request_to_prediction_log(
    request: predict_pb2.PredictRequest,
) -> prediction_log_pb2.PredictionLog:
  """Creates a PredictionLog for the Predict method."""
  return prediction_log_pb2.PredictionLog(
      predict_log=prediction_log_pb2.PredictLog(request=request)
  )


def generate_examples_with_sequence_lengths(
    sequence_lengths: list[int], single_token_example: WarmupExample = 'Q'
) -> list[WarmupExamples]:
  """Creates synthetic sequences of specified sizes by repeating a single token.

  Args:
    sequence_lengths: The sequence lengths to generate examples for.
    single_token_example: An example such that `N*ex` is always `N` tokens long.
      This is used to build sequences of a specified size. Defaults to `'Q'`,
      which satisfies this property for the tokenizer used by pretrained T5X
      models.

  Returns:
    A list of WarmupExamples batches with lengths in tokens equal to
    `sequence_lengths`.
  """
  return [[single_token_example * l] for l in sequence_lengths]


def write_warmup_examples(
    text_batch: WarmupExamples,
    output_dir: str,
    model_name: str,
    signature_name: str,
    *,
    generate_examples_fn: Optional[Callable[[], list[WarmupExamples]]] = None,
    batch_sizes: List[Optional[int]],
    input_tensor_name: str = 'text_batch',
    decoder_params_spec: Optional[DecoderParamsSpec] = None,
    request_to_prediction_log: Callable[
        [predict_pb2.PredictRequest], prediction_log_pb2.PredictionLog
    ] = _request_to_prediction_log,
    input_tensor_dtype: Optional[tf.DType] = None,
):
  """Writes warmup examples for all batch_sizes requested.

  The text_batch is either filled to batch_size or truncated based on the
  different batch_sizes.
  For example, if text_batch has length 2 while requested batch_size is 4, it is
  repeated two times. If text_batch has length 2 while requested batch_size is
  1, it is truncated to length 1.

  Args:
    text_batch: A batch of texts used as warmup examples.
    output_dir: The directory for writing the warmup examples to.
    model_name: The name of the savedmodel spec.
    signature_name: Optional name of the exported function.
    generate_examples_fn: An optional no-arg function that generates a list of
      synthetic warmup batches. If provided, `text_batch` will be ignored, and
      all synthetic batches will be written out, for all batch sizes. NOTE: This
      differs from `text_batch`, which only supplies a single batch of examples
      that are written together, instead of many batches that are written
      separately.
    batch_sizes: A list of batch sizes to warmup with. The written number of
      tfrecords will be equal to the size of batch_sizes. The list might contain
      None entries, and the warmup examples for the None entry won't be padded
      or truncated.
    input_tensor_name: The entry name of the PredictRequest inputs dict.
    decoder_params_spec: The parameter specifciations on decoding. If present,
      dummy data (0s) with specified shape/dtype will be written into warmup
      examples.
    request_to_prediction_log: A function that creates a PredictionLog from a
      given request.
    input_tensor_dtype: The dtype of the input tensor.
  """
  if generate_examples_fn:
    logging.warning(
        'Ignoring provided warmup batch. Using `generate_examples_fn` to'
        ' generate warmup examples instead.'
    )
    warmup_examples = generate_examples_fn()
  else:
    warmup_examples = [text_batch]

  assets_extra = os.path.join(output_dir, 'assets.extra')
  tf.io.gfile.makedirs(assets_extra)
  warmup_output = os.path.join(assets_extra, 'tf_serving_warmup_requests')
  with tf.io.TFRecordWriter(warmup_output) as writer:
    for warmup_example in warmup_examples:
      for batch_size in batch_sizes:
        logging.info('Writing warmup data for batch size: %s ...', batch_size)
        request = _request_for_batch(
            warmup_example,
            model_name,
            input_tensor_name,
            signature_name,
            batch_size,
            decoder_params_spec,
            input_tensor_dtype,
        )
        log = request_to_prediction_log(request)
        writer.write(log.SerializeToString())




def _standardize_output_features(
    mixture_or_task_name: Optional[str],
    output_features: Optional[Mapping[str, seqio.Feature]],
):
  """Standarize the output_features from user inputs."""
  new_output_features = output_features
  if mixture_or_task_name is not None and output_features is not None:
    raise ValueError(
        'Only one of mixture_or_task_name and output_features may be non empty.'
    )
  if mixture_or_task_name is not None:
    logging.info('Fetching output features from task %s', mixture_or_task_name)
    new_output_features = seqio.get_mixture_or_task(
        mixture_or_task_name
    ).output_features
  return new_output_features


def _standardize_output_dirs(output_dir: Union[str, Mapping[str, str]]):
  """Standardize the format of output_dirs from user input."""
  logging.info('Standardizing the output_dir: %s', output_dir)
  if output_dir is None:
    raise ValueError('output_dir is mandatory')
  if isinstance(output_dir, str):
    output_dirs = {'tpu': output_dir}
  else:
    output_dirs = dict(output_dir)
  if 'cpu' not in output_dirs:
    if 'tpu' not in output_dirs:
      raise ValueError('output_dir["cpu"] or output_dir["tpu"] is mandatory')
    export_version = os.path.basename(output_dirs['tpu'])
    if not export_version.isdigit():
      raise ValueError(
          'output_dir must be in the form ${BASE}/${VERSION}, '
          'where  ${VERSION} is an integer. Got a non-numeric '
          f'version {export_version}.'
      )
    output_dirs['cpu'] = os.path.join(
        os.path.dirname(output_dirs['tpu']) + '_cpu', export_version
    )
  logging.info('Result standardized output_dirs: %s', output_dirs)
  return output_dirs


def create_fake_input(signature: Dict[str, tf.TensorSpec]) -> Any:
  """Create all zeros fake inputs according to signature spec.

  Args:
    signature: A dictionary of tensor specs that will used for serving.

  Returns:
    A pytree with the same structure as signature with all zeros tf.Tensor.
  """

  def _gen_dummy_tensor(ts: tf.TensorSpec):
    shape = ts.shape.as_list()
    if not all(shape[1:]):
      raise ValueError(
          'Only supports polymorphic batch size at leading dimension, got '
          f'{ts} in the input signature.'
      )
    if shape and shape[0] is None:
      shape[0] = 1
    return tf.zeros(shape, ts.dtype)

  return jax.tree_util.tree_map(_gen_dummy_tensor, signature)


def create_batch_polymorphic_shapes(
    input_signature,
    preprocessor,
    *,
    create_fake_input_fn=create_fake_input,
    overrides=None,
):
  """Creates batch polymorhic shapes for jax2tf, and override specific shapes."""
  is_poly_batch = not all(
      s.shape.is_fully_defined()
      for s in jax.tree_util.tree_leaves(input_signature)
  )

  # All shapes are static.
  if not is_poly_batch and not overrides:
    return None

  fake_inputs = create_fake_input_fn(input_signature)
  features = preprocessor(*fake_inputs)

  # All the features have a leading batch dimension.
  shapes = jax.tree_util.tree_map(
      lambda _: 'b, ...' if is_poly_batch else None, features
  )
  if overrides:
    shapes.update(overrides)
  return shapes


def save(
    *,
    model: models.BaseTransformerModel,
    inference_mode: str | CustomInferenceMode,
    restore_checkpoint_cfg: utils.RestoreCheckpointConfig,
    exportable_module_cls: Type[ExportableModule],
    create_preprocessor_fn: CreatePreprocessorFn = create_preprocessor,
    create_inference_function_fn: Callable[
        ..., InferenceFn
    ] = create_inference_function,
    create_postprocessor_fn: CreatePostprocessorFn = create_postprocessor,
    partitioner: Optional[partitioning.BasePartitioner],
    create_decoding_state_callback_fn: Optional[
        CreateDecodingStateCallbackFn
    ] = None,
    output_features: Optional[Mapping[str, seqio.Feature]],
    task_feature_lengths: Mapping[str, int],
    batch_size: Optional[int],
    output_dir: Union[str, Mapping[str, str]],
    model_name: str,
    warmup_examples: Optional[WarmupExamples] = None,
    tokenized_inputs: bool = False,
    write_warmup_example_fn: Callable[..., None] = write_warmup_examples,
    mixture_or_task_name: Optional[str] = None,
    validation_examples: Optional[List[Any]] = None,
    native_lowering: bool = True,
    native_lowering_platforms: Optional[Sequence[str]] = None,
    enable_xla: bool = True,
    decode_outputs: Optional[bool] = None,
    trailing_shapes: Optional[Mapping[str, Tuple[int, ...]]] = None,
    output_vocab_feature_name: Optional[str] = 'targets',
    signature_name: str = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    create_polymorphic_shapes_fn: Any = create_batch_polymorphic_shapes,
):
  """Saves the passed EncoderDecoderModel as a TPU-enabled TF SavedModel.

  Args:
    model:
    inference_mode: "predict", "score" or a CustomInferenceMode instance.
    restore_checkpoint_cfg: Configuration for restoring model from checkpoint.
    exportable_module_cls: A configured implementation of ExportableModule.
    create_preprocessor_fn: Configurable func. to create the PreprocessorFn.
    create_inference_function_fn: Configurable func. to create the InferenceFn.
    create_postprocessor_fn: Configurable func. to create the PostprocessorFn.
    partitioner: Partitioner, usually for Pjit.
    create_decoding_state_callback_fn: Configurable func. to create an optional
      decoding.StateCallbackFn.
    output_features: Output Features of the task.
    task_feature_lengths: Input and target lengths.
    batch_size: Batch size for model to process. If None, then batch
      polymorphism is invoked.
    output_dir: This is either: (a) A path in ${BASE}/${VERSION} format output
      the final TPU-converted saved model. The CPU saved model will be saved to
      ${BASE}_cpu/${VERSION}, such that "_cpu" is appended to the base path but
      the numeric version is preserved. (b) A dict with key 'cpu' and as value
      the path to write the CPU model to.
    model_name: Name of model, like "/ml/user/half_plus_two".
    warmup_examples: Optional list of warmup examples. If proveded, they will be
      written in Predict mode to assets.extra.
    tokenized_inputs: if True, inputs are expected to be pre-tokenized before
      being passed to the Jax2TF converted model, e.g. an int32 tensor of type
      [B, L]. If False, inputs is expected to be a string tensor of shape [B].
      We typically set tokenized_inputs to True if tokenization is handled by an
      external service. This will disable tokenization in the preprocessor and
      postprocessor.
    write_warmup_example_fn: a callable which writes a set of warmup examples to
      a pbtxt file for use validating a converted model.
    mixture_or_task_name: Optional SeqIO task name used to get output features.
      In order to set this output_features must be None.
    validation_examples: Optional list of validation examples. If proveded, they
      will be used to validate the latency and numeric accuracy of the TPU saved
      model.
    native_lowering: for experimental purposes only -- if True, don't convert
      Jax fns to TF fns.
    native_lowering_platforms: In conjunction with `native_lowering`, specify
      the platform(s) for which to lower the code. Must be a tuple of strings,
      including a subset of: 'cpu', 'cuda', 'rocm', 'tpu'. The default
      (None), specifies the JAX default backend on the machine where the
      lowering is done.
    enable_xla: Defaults to true. If false, jax2tf conversion only emits non-XLA
      ops.
    decode_outputs: Optional bool. If provided, determines whether to decode the
      output with the tokenizer, or to leave the output as is.
    trailing_shapes: Optional mapping of model feature name to trailing shape,
      the `...?` in `(batch_size, seqlen, ...?)`, which is needed to initialize
      the model correctly.
    output_vocab_feature_name: The vocabulary feature which maps decoded ids to
      plain text. For standard T5X models this will always be 'targets', but may
      be different or empty for other models.
    signature_name: Name of the exported function.
    create_polymorphic_shapes_fn: Optional function to create polymorphic shapes
      for input tensors to the JAX model function.
  """  # fmt: skip
  jax.monitoring.record_event('/jax/t5x/export/beacon')
  output_dirs = _standardize_output_dirs(output_dir)
  del output_dir


  logging.info('jax.process_count: %s', jax.process_count())
  logging.info('jax.local_devices: %s', jax.local_devices())  # Seems necessary.
  logging.info('Creating inference function...')
  train_state_initializer = get_train_state_initializer(
      model, partitioner, task_feature_lengths, batch_size, trailing_shapes
  )

  output_features = _standardize_output_features(
      mixture_or_task_name, output_features
  )
  # Get the preprocessor and postprocessor.

  # Non-vanilla seq-to-seq/decoder-only models can have a different
  # vocabulary feature or not use a vocabulary feature at all.
  output_vocab = None
  if output_vocab_feature_name:
    output_vocab = output_features[output_vocab_feature_name].vocabulary

  # Handle the new and old create_preprocessor_fn signatures, for backwards
  # compatibility.
  # TODO(marcrasi): Delete after migrating clients.
  if 'batch_size' in inspect.signature(create_preprocessor_fn).parameters:
    # New signature.
    preprocessor, input_signature = create_preprocessor_fn(
        batch_size, output_features, task_feature_lengths, tokenized_inputs
    )  # type: ignore
  else:
    # Old signature.
    preprocessor = create_preprocessor_fn(
        output_features, task_feature_lengths, tokenized_inputs
    )  # type: ignore
    input_signature = create_single_tensor_input_signature(
        batch_size, task_feature_lengths, tokenized_inputs
    )

  logging.info('Converting inference function...')

  decoding_state_callback_fn = None
  if create_decoding_state_callback_fn is not None:
    decoding_state_callback_fn = create_decoding_state_callback_fn(
        vocab=output_vocab,
        call_tf_graph=native_lowering,
    )

  model_tf_fn = create_inference_function_fn(
      model=model,
      train_state_initializer=train_state_initializer,
      decoding_state_callback_fn=decoding_state_callback_fn,
      partitioner=partitioner,
      inference_mode=inference_mode,
      enable_jax2tf=True,
      enable_xla=enable_xla,
      polymorphic_shapes_inputs=create_polymorphic_shapes_fn(
          input_signature, preprocessor
      ),
      native_lowering=native_lowering,
      native_lowering_platforms=native_lowering_platforms,
  )

  logging.info('Loading parameters from checkpoint...')
  params = load_params_from_checkpoint(
      restore_checkpoint_cfg=restore_checkpoint_cfg,
      train_state_initializer=train_state_initializer,
      partitioner=partitioner,
  )

  logging.info('Preparing Module to save...')
  if decode_outputs is None:
    decode_outputs = not tokenized_inputs
  postprocessor = create_postprocessor_fn(
      output_vocab, inference_mode, decode_outputs
  )
  module = exportable_module_cls(
      preproc_tf_fn=preprocessor,
      model_tf_fn=model_tf_fn,
      postproc_tf_fn=postprocessor,
      params=params,
      batch_size=batch_size,
  )
  module.preproc_func.get_concrete_function(*input_signature)
  signatures = {
      signature_name: module.__call__.get_concrete_function(*input_signature)
  }
  logging.info('Saving the CPU model...')
  # TODO(b/196260374): Figure out how to set experimental_custom_gradients=True.
  options = tf.saved_model.SaveOptions(
      experimental_custom_gradients=False,
      function_aliases={
          'tpu_func': module.tpu_func,
      },
  )
  tf.saved_model.save(
      module,
      output_dirs['cpu'],
      signatures=signatures,
      options=options,
  )


  if warmup_examples:
    if batch_size:
      warmup_examples = warmup_examples[:batch_size]
      while len(warmup_examples) < batch_size:
        if tokenized_inputs:
          warmup_examples.append(np.zeros_like(warmup_examples[0]).tolist())
        else:
          warmup_examples.append('')

    write_warmup_example_fn(
        warmup_examples,
        output_dir=output_dirs['cpu'],
        model_name=model_name,
        batch_sizes=module.export_batch_sizes,
        signature_name=signature_name,
    )



  # TODO(danielandor): Save the graph.pbtxt for debugging purposes.
