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

# pylint:disable=line-too-long
# pyformat: disable
r"""This script runs inference-evaluation on a T5X-compatible model.

"""
# pyformat: enable
# pylint:enable=line-too-long

import functools
import os
from typing import Optional, Sequence, Type

from absl import logging
# Set Linen to add profiling information when constructing Modules.
# Must be set before flax imports.
# pylint:disable=g-import-not-at-top
os.environ['FLAX_PROFILE'] = 'true'
import jax
import seqio
from t5x import models
from t5x import multihost_utils
from t5x import partitioning
from t5x import utils

# Automatically search for gin files relative to the T5X package.
_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]


def evaluate(
    *,
    model: models.BaseTransformerModel,
    dataset_cfg: utils.DatasetConfig,
    restore_checkpoint_cfg: utils.RestoreCheckpointConfig,
    partitioner: partitioning.BasePartitioner,
    output_dir: str,
    feature_converter_cls: Optional[Type[seqio.FeatureConverter]] = None):
  """Evaluation function.

  Args:
    model: The model object to use for inference.
    dataset_cfg: Specification for the dataset to infer based on.
    restore_checkpoint_cfg: Specification for the model parameter checkpoint to
      load.
    partitioner: Partitioner for the model parameters and data across devices.
    output_dir: Path to directory to write temporary files and final results.
    feature_converter_cls: Optional SeqIO feature converters to apply to the
      dataset. If `None`, overrides `model.FEATURE_CONVERTER_CLS`.
  """
  if dataset_cfg.module:
    utils.import_module(dataset_cfg.module)
  batch_size = dataset_cfg.batch_size

  ds_vocabs = utils.get_vocabulary(dataset_cfg)
  if (ds_vocabs[0] != model.input_vocabulary or
      ds_vocabs[1] != model.output_vocabulary):
    raise ValueError(f'Model and Task vocabularies do not match:\n'
                     f'  task={dataset_cfg.mixture_or_task_name}\n'
                     f'  ds_vocabs=({ds_vocabs[0]}, {ds_vocabs[1]})\n'
                     f'  model.input_vocabulary={model.input_vocabulary}\n'
                     f'  model.output_vocabulary={model.output_vocabulary}\n')

  # ----------------------------------------------------------------------------
  # SeqIO (inference-based) evaluation setup
  # ----------------------------------------------------------------------------
  if feature_converter_cls is not None:
    feature_converter = feature_converter_cls(pack=False)  # pytype:disable=not-instantiable
  else:
    feature_converter = model.FEATURE_CONVERTER_CLS(pack=False)  # pytype:disable=not-instantiable

  # Init evaluator to set up cached datasets
  evaluator = seqio.Evaluator(
      mixture_or_task_name=dataset_cfg.mixture_or_task_name,
      feature_converter=feature_converter,  # pytype:disable=not-instantiable
      eval_split=dataset_cfg.split,
      use_cached=dataset_cfg.use_cached,
      seed=dataset_cfg.seed,
      sequence_length=dataset_cfg.task_feature_lengths,
      logger_cls=[
          seqio.PyLoggingLogger, seqio.TensorBoardLogger, seqio.JSONLogger
      ],
      log_dir=os.path.join(output_dir, 'inference_eval'))
  if not evaluator.eval_tasks:
    raise ValueError(
        f"'{dataset_cfg.mixture_or_task_name}' has no metrics for evaluation.")

  # ----------------------------------------------------------------------------
  # T5X model loading.
  # ----------------------------------------------------------------------------

  # Initialize optimizer from the existing checkpoint.
  input_shapes = {
      k: (batch_size, l) for k, l in evaluator.model_feature_lengths.items()
  }

  train_state_initializer = utils.TrainStateInitializer(
      optimizer_def=model.optimizer_def,
      init_fn=model.get_initial_variables,
      input_shapes=input_shapes,
      partitioner=partitioner)
  train_state_axes = train_state_initializer.train_state_axes

  predict_fn = None
  score_fn = None

  for train_state in train_state_initializer.from_checkpoints(
      [restore_checkpoint_cfg]):

    # Compile the model only once.
    if not predict_fn:
      predict_fn = utils.get_infer_fn(
          infer_step=model.predict_batch,
          batch_size=batch_size,
          train_state_axes=train_state_axes,
          partitioner=partitioner)

      score_fn = utils.get_infer_fn(
          infer_step=model.score_batch,
          batch_size=batch_size,
          train_state_axes=train_state_axes,
          partitioner=partitioner)

    # ----------------------------------------------------------------------------
    # Main training loop
    # ----------------------------------------------------------------------------

    # Run final evaluation (with decoding) on the full eval dataset.
    all_metrics, _, _ = evaluator.evaluate(
        compute_metrics=jax.host_id() == 0,
        step=int(train_state.step),
        predict_fn=functools.partial(predict_fn, train_state=train_state),
        score_fn=functools.partial(score_fn, train_state=train_state))
    all_metrics.result()  # Ensure metrics are finished being computed.
    # Wait until computations are done before continuing.
    multihost_utils.sync_devices(f'step_{train_state.step}:complete')

  logging.info('Finished.')


if __name__ == '__main__':
  from absl import app
  from absl import flags
  import gin
  from t5x import gin_utils

  FLAGS = flags.FLAGS

  jax.config.parse_flags_with_absl()

  flags.DEFINE_multi_string(
      'gin_file',
      default=None,
      help='Path to gin configuration file. Multiple paths may be passed and '
      'will be imported in the given order, with later configurations  '
      'overriding earlier ones.')

  flags.DEFINE_multi_string(
      'gin_bindings', default=[], help='Individual gin bindings.')

  flags.DEFINE_list(
      'gin_search_paths',
      default=['.'],
      help='Comma-separated list of gin config path prefixes to be prepended '
      'to suffixes given via `--gin_file`. If a file appears in. Only the '
      'first prefix that produces a valid path for each suffix will be '
      'used.')

  flags.DEFINE_string(
      'tfds_data_dir', None,
      'If set, this directory will be used to store datasets prepared by '
      'TensorFlow Datasets that are not available in the public TFDS GCS '
      'bucket. Note that this flag overrides the `tfds_data_dir` attribute of '
      'all `Task`s.')


  def main(argv: Sequence[str]):
    """Wrapper for pdb post mortems."""
    _main(argv)

  def _main(argv: Sequence[str]):
    """True main function."""
    if len(argv) > 1:
      raise app.UsageError('Too many command-line arguments.')

    if FLAGS.tfds_data_dir:
      seqio.set_tfds_data_dir_override(FLAGS.tfds_data_dir)

    # Create gin-configurable version of `eval`.
    evaluate_using_gin = gin.configurable(evaluate)

    gin_utils.parse_gin_flags(
        # User-provided gin paths take precedence if relative paths conflict.
        FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
        FLAGS.gin_file,
        FLAGS.gin_bindings)
    evaluate_using_gin()


  gin_utils.run(main)
