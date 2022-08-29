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

# pylint:disable=line-too-long
# pyformat: disable
r"""This script runs inference-evaluation on a T5X-compatible model.

"""
# pyformat: enable
# pylint:enable=line-too-long

import functools
import os
from typing import Optional, Sequence, Type

# pylint:disable=g-import-not-at-top
# TODO(adarob): Re-enable once users are notified and tests are updated.
os.environ['FLAX_LAZY_RNG'] = 'no'
from absl import logging
from clu import metric_writers
import jax
from jax.experimental import multihost_utils
import seqio
from t5x import gin_utils
from t5x import models
from t5x import partitioning
from t5x import utils
from typing_extensions import Protocol

# Automatically search for gin files relative to the T5X package.
_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]


class SummarizeConfigFn(Protocol):

  def __call__(self, model_dir: str,
               summary_writer: Optional[metric_writers.SummaryWriter],
               step: int) -> None:
    ...


def evaluate(
    *,
    model: models.BaseTransformerModel,
    dataset_cfg: utils.DatasetConfig,
    restore_checkpoint_cfg: utils.RestoreCheckpointConfig,
    partitioner: partitioning.BasePartitioner,
    output_dir: str,
    inference_evaluator_cls: utils.EvaluatorConstructor = seqio.Evaluator,
    summarize_config_fn: SummarizeConfigFn = gin_utils.summarize_gin_config,
    train_state_initializer_cls: Type[
        utils.TrainStateInitializer] = utils.TrainStateInitializer,
    fallback_init_rng: Optional[int] = None):
  """Evaluation function.

  Args:
    model: The model object to use for inference.
    dataset_cfg: Specification for the dataset to infer based on.
    restore_checkpoint_cfg: Specification for the model parameter checkpoint to
      load.
    partitioner: Partitioner for the model parameters and data across devices.
    output_dir: Path to directory to write temporary files and final results.
    inference_evaluator_cls: seqio.Evaluator class to use for inference
      evaluation, potentially with bound configuration args.
    summarize_config_fn: A function that takes in the model directory, an
      optional SummaryWriter, and the step number, and writes a summary of the
      configuration. SummaryWriter will be None in most cases.
    train_state_initializer_cls: t5x.utils.TrainStateInitializer class
      for initializing partitioned TrainState from checkpoints or scratch.
    fallback_init_rng: A random seed used for parameter initialization during
      model re-loading when utils.RestoreCheckpointConfig.fallback_to_scratch is
      set to True. If None, parameter initialization is not allowed during model
      loading and having fallback_to_scratch enabled will result in an error.
  """
  logging.info('Process ID: %d', jax.process_index())
  if dataset_cfg.module:
    utils.import_module(dataset_cfg.module)
  batch_size = dataset_cfg.batch_size

  # TODO(b/234480674): GDA not supported for eval.
  restore_checkpoint_cfg.use_gda = False

  summarize_config_fn(model_dir=output_dir, summary_writer=None, step=0)

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
  # Init evaluator to set up cached datasets
  evaluator = inference_evaluator_cls(
      mixture_or_task_name=dataset_cfg.mixture_or_task_name,
      feature_converter=model.FEATURE_CONVERTER_CLS(pack=False),
      eval_split=dataset_cfg.split,
      use_cached=dataset_cfg.use_cached,
      seed=dataset_cfg.seed,
      sequence_length=dataset_cfg.task_feature_lengths,
      log_dir=os.path.join(output_dir, 'inference_eval'),
      use_memory_cache=dataset_cfg.use_memory_cache)
  if not evaluator.eval_tasks:
    raise ValueError(
        f"'{dataset_cfg.mixture_or_task_name}' has no metrics for evaluation.")

  # ----------------------------------------------------------------------------
  # T5X model loading.
  # ----------------------------------------------------------------------------

  # Initialize optimizer from the existing checkpoint.
  input_shapes = {
      k: (batch_size,) + s for k, s in evaluator.model_feature_shapes.items()
  }

  train_state_initializer = train_state_initializer_cls(
      optimizer_def=None,  # Do not load optimizer state.
      init_fn=model.get_initial_variables,
      input_shapes=input_shapes,
      partitioner=partitioner)
  train_state_axes = train_state_initializer.train_state_axes
  # Log the variable shapes information and write to a file.
  log_file = os.path.join(output_dir, 'model-info.txt')
  utils.log_model_info(log_file,
                       train_state_initializer.global_train_state_shape,
                       partitioner)

  predict_fn = None
  score_fn = None

  # Disable strictness since we are dropping the optimizer state.
  restore_checkpoint_cfg.strict = False

  if fallback_init_rng is not None:
    fallback_init_rng = jax.random.PRNGKey(fallback_init_rng)
  for train_state in train_state_initializer.from_checkpoints(
      [restore_checkpoint_cfg], init_rng=fallback_init_rng):

    # Compile the model only once.
    if not predict_fn:
      predict_fn = utils.get_infer_fn(
          infer_step=model.predict_batch,
          batch_size=batch_size,
          train_state_axes=train_state_axes,
          partitioner=partitioner)

      predict_with_aux_fn = utils.get_infer_fn(
          infer_step=model.predict_batch_with_aux,
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
    host_step = int(utils.get_local_data(train_state.step))
    all_metrics, _, _ = evaluator.evaluate(
        compute_metrics=jax.process_index() == 0,
        step=host_step,
        predict_fn=functools.partial(
            predict_fn, train_state=train_state, rng=jax.random.PRNGKey(0)),
        score_fn=functools.partial(score_fn, train_state=train_state),
        predict_with_aux_fn=functools.partial(
            predict_with_aux_fn,
            train_state=train_state,
            rng=jax.random.PRNGKey(0)))
    all_metrics.result()  # Ensure metrics are finished being computed.
    # Wait until computations are done before continuing.
    multihost_utils.sync_global_devices(f'step_{host_step}:complete')

  logging.info('Finished.')


if __name__ == '__main__':
  from absl import app
  from absl import flags
  import gin

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
