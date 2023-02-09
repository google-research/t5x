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

"""Functions for validating the exported T5X model."""
from typing import Any, List, Mapping, Optional, Tuple, Union

import gin
import jax
import seqio
from t5x import export_lib
from t5x import models
from t5x import partitioning
from t5x import utils
from t5x.export_lib import _standardize_output_dirs
from t5x.export_lib import CreateDecodingStateCallbackFn
from t5x.export_lib import CreatePostprocessorFn
from t5x.export_lib import CreatePreprocessorFn
from t5x.google.export import validate
import tensorflow as tf


@gin.configurable
def model_validate(
    *,
    model: models.BaseTransformerModel,
    inference_mode: str,
    restore_checkpoint_cfg: utils.RestoreCheckpointConfig,
    create_preprocessor_fn: CreatePreprocessorFn,
    create_postprocessor_fn: CreatePostprocessorFn,
    partitioner: Optional[partitioning.BasePartitioner],
    create_decoding_state_callback_fn: Optional[
        CreateDecodingStateCallbackFn
    ] = None,
    output_features: Optional[Mapping[str, seqio.Feature]],
    task_feature_lengths: Mapping[str, int],
    batch_size: Optional[int],
    output_dir: Union[str, Mapping[str, str]],
    tokenized_inputs: bool = False,
    mixture_or_task_name: Optional[str] = None,
    validation_examples: Optional[List[Any]] = None,
    decode_outputs: Optional[bool] = None,
    trailing_shapes: Optional[Mapping[str, Tuple[int, ...]]] = None,
    output_vocab_feature_name: Optional[str] = 'targets',
    signature_name: Optional[
        str
    ] = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
):
  """Validate the JAX model and tf.SavedModel result consistency.

  Args:
    model:
    inference_mode: "predict", "score" or a CustomInferenceMode instance.
    restore_checkpoint_cfg: Configuration for restoring model from checkpoint.
    create_preprocessor_fn: Configurable func. to create the PreprocessorFn.
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
    export_tpu: Deprecated, please pass output_dir={'cpu': ...} without a 'tpu'
      key instead. This controls whether to export the TPU saved model in
      addition to the CPU model. The CPU model is always exported, and if
      `export_tpu` is True, then the CPU model is converted and saved as a TPU
    tokenized_inputs: if True, inputs are expected to be pre-tokenized before
      being passed to the Jax2TF converted model, e.g. an int32 tensor of type
      [B, L]. If False, inputs is expected to be a string tensor of shape [B].
      We typically set tokenized_inputs to True if tokenization is handled by an
      external service. This will disable tokenization in the preprocessor and
      postprocessor.
    mixture_or_task_name: Optioanl SeqIO task name used to get output features.
      In order to set this output_features must be None.
    validation_examples: Optional list of validation examples. If proveded, they
      will be used to validate the latency and numeric accuracy of the TPU saved
  del output_dir


  jax_module, serving_configs, _, _, _ = (
      export_lib.create_orbax_jax_module_and_serve_config(
          model=model,
          inference_mode=inference_mode,
          restore_checkpoint_cfg=restore_checkpoint_cfg,
          create_preprocessor_fn=create_preprocessor_fn,
          create_postprocessor_fn=create_postprocessor_fn,
          partitioner=partitioner,
          create_decoding_state_callback_fn=create_decoding_state_callback_fn,
          output_features=output_features,
          task_feature_lengths=task_feature_lengths,
          batch_size=batch_size,
          tokenized_inputs=tokenized_inputs,
          mixture_or_task_name=mixture_or_task_name,
          decode_outputs=decode_outputs,
          trailing_shapes=trailing_shapes,
          output_vocab_feature_name=output_vocab_feature_name,
          signature_name=signature_name,
      )
  )

  validate.compare_jax_and_savedmodel(
      examples=validation_examples,
      jax_module=jax_module,
      serving_configs=serving_configs,
      batch_size=batch_size,
      tpu_savedmodel_path=output_dirs['tpu'],
      is_tpu_model=True,
  )
