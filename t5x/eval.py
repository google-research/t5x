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

# pylint:disable=line-too-long
# pyformat: disable
r"""Runs training- and inference-evaluation on a T5X-compatible model.

"""
# pyformat: enable
# pylint:enable=line-too-long
import functools
import os
import re
from typing import Callable, Collection, Mapping, Optional, Sequence, Set, Tuple, Type

# pylint:disable=g-import-not-at-top
# TODO(adarob): Re-enable once users are notified and tests are updated.
os.environ['FLAX_LAZY_RNG'] = 'no'
from absl import logging
from clu import metric_writers
import jax
import seqio
from t5x import checkpoints
from t5x import gin_utils
from t5x import models
from t5x import partitioning
from t5x import train_state as train_state_lib
from t5x import trainer as trainer_lib
from t5x import utils
from tensorflow.io import gfile
from typing_extensions import Protocol
# pylint:enable=g-import-not-at-top

# Automatically search for gin files relative to the T5X package.
_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]


class SummarizeConfigFn(Protocol):

  def __call__(self, model_dir: str,
               summary_writer: Optional[metric_writers.SummaryWriter],
               step: int) -> None:
    ...


class InferenceEvaluator:
  """Runs evaluation of the model against a given SeqIo task."""

  def __init__(
      self,
      infer_eval_dataset_cfg: utils.DatasetConfig,
      inference_evaluator_cls: utils.EvaluatorConstructor,
      model: models.BaseModel,
      partitioner: partitioning.BasePartitioner,
      log_dir: Optional[str] = None,
      verify_matching_vocabs_fn: Optional[
          Callable[[utils.DatasetConfig, models.BaseModel],
                   None]] = utils.verify_matching_vocabs,
  ):
    """Constructs inference evaluator.

    Args:
      infer_eval_dataset_cfg: Specification for the dataset to evaluate with
        using the inference metrics (e.g., uses sampled decoding). If None,
        inference eval is disabled.
      inference_evaluator_cls: seqio.Evaluator class to use for inference
        evaluation, potentially with bound configuration args.
      model: Model to be evaluated.
      partitioner: the partitioner to use.
      log_dir: Parent directory to log evaluation results.
      verify_matching_vocabs_fn: Function to validate whether the task
        vocabulary matches the model vocabulary. Should raise an exception on
        error.
    """
    if verify_matching_vocabs_fn is not None:
      verify_matching_vocabs_fn(infer_eval_dataset_cfg, model)

    self._model = model
    self._partitioner = partitioner
    self._infer_eval_dataset_cfg = infer_eval_dataset_cfg
    kwargs = {}
    if log_dir:
      kwargs['log_dir'] = os.path.join(log_dir, 'inference_eval')
    else:
      # Disable loggers if log dir is not provided.
      kwargs['logger_cls'] = ()
    self._seqio_evaluator = inference_evaluator_cls(
        mixture_or_task_name=infer_eval_dataset_cfg.mixture_or_task_name,
        feature_converter=model.FEATURE_CONVERTER_CLS(pack=False),
        eval_split=infer_eval_dataset_cfg.split,
        use_cached=infer_eval_dataset_cfg.use_cached,
        seed=infer_eval_dataset_cfg.seed,
        sequence_length=infer_eval_dataset_cfg.task_feature_lengths,
        use_memory_cache=infer_eval_dataset_cfg.use_memory_cache,
        **kwargs)
    # Lazily initialized upon the first `evaluate` call.
    self._predict_fn = None
    self._predict_with_aux_fn = None
    self._score_fn = None

  @property
  def model_feature_shapes(self) -> Mapping[str, Tuple[int, ...]]:
    return self._seqio_evaluator.model_feature_shapes

  @property
  def eval_tasks(self) -> Sequence[seqio.Task]:
    return self._seqio_evaluator.eval_tasks

  def close(self):
    self._seqio_evaluator.close()

  def evaluate(
      self,
      train_state: train_state_lib.TrainState,
      train_state_axes: train_state_lib.TrainState,
  ) -> seqio.evaluation.AllMetricsFuture:
    """Runs the prediction based inference eval.

    Args:
      train_state: Training state to run evaluation of.
      train_state_axes: partitioning info for the train state to be used.

    Returns:
      A dictionary of training eval metrics.
    """
    if not self._predict_fn:
      self._predict_fn = utils.get_infer_fn(
          infer_step=self._model.predict_batch,
          batch_size=self._infer_eval_dataset_cfg.batch_size,
          train_state_axes=train_state_axes,
          partitioner=self._partitioner)

      self._predict_with_aux_fn = utils.get_infer_fn(
          infer_step=self._model.predict_batch_with_aux,
          batch_size=self._infer_eval_dataset_cfg.batch_size,
          train_state_axes=train_state_axes,
          partitioner=self._partitioner)

      self._score_fn = utils.get_infer_fn(
          infer_step=self._model.score_batch,
          batch_size=self._infer_eval_dataset_cfg.batch_size,
          train_state_axes=train_state_axes,
          partitioner=self._partitioner)

    all_metrics, _ = self._seqio_evaluator.evaluate(
        compute_metrics=jax.process_index() == 0,
        step=int(utils.get_local_data(train_state.step)),
        predict_fn=functools.partial(
            self._predict_fn,
            train_state=train_state,
            rng=jax.random.PRNGKey(0)),
        score_fn=functools.partial(self._score_fn, train_state=train_state),
        predict_with_aux_fn=functools.partial(
            self._predict_with_aux_fn,
            train_state=train_state,
            rng=jax.random.PRNGKey(0)),
    )
    return all_metrics


def _sorted_ckpt_paths(ckpt_paths: Collection[str]) -> Sequence[str]:
  def _extract_ckpt_step(ckpt_path: str) -> int:
    # Steps may be prefixed with "checkpoint_", "model.ckpt-" or nothing.
    match = re.search(r'(checkpoint_|model.ckpt-)?(\d+)\/?$', ckpt_path)
    if match is None:
      raise ValueError(f'Invalid checkpoint path: {ckpt_path}')
    assert match is not None
    return int(match.group(2))

  return sorted(ckpt_paths, key=_extract_ckpt_step)


def _load_evaluated_ckpt_paths(eval_ckpt_path: str) -> Set[str]:
  if not gfile.exists(eval_ckpt_path):
    return set()
  with gfile.GFile(eval_ckpt_path, 'r') as f:
    return set(f.read().split())


def evaluate(
    *,
    model: models.BaseTransformerModel,
    dataset_cfg: utils.DatasetConfig,
    restore_checkpoint_cfg: utils.RestoreCheckpointConfig,
    partitioner: partitioning.BasePartitioner,
    output_dir: str,
    inference_evaluator_cls: Optional[
        utils.EvaluatorConstructor
    ] = seqio.Evaluator,
    training_evaluator_cls: Optional[Type[trainer_lib.Trainer]] = None,
    summarize_config_fn: SummarizeConfigFn = gin_utils.summarize_gin_config,
    train_state_initializer_cls: Type[
        utils.TrainStateInitializer
    ] = utils.TrainStateInitializer,
    train_eval_get_dataset_fn: utils.GetEvalDatasetCallable = utils.get_training_eval_datasets,
    fallback_init_rng: Optional[int] = None,
    use_orbax: bool = False,
):
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
    training_evaluator_cls: an optional Trainer class to use for training
      evaluation, potentially with bound configuration args.
    summarize_config_fn: A function that takes in the model directory, an
      optional SummaryWriter, and the step number, and writes a summary of the
      configuration. SummaryWriter will be None in most cases.
    train_state_initializer_cls: t5x.utils.TrainStateInitializer class for
      initializing partitioned TrainState from checkpoints or scratch.
    train_eval_get_dataset_fn: Optional callable use to get the train-eval
      datasets based on the DatasetConfig and shard information. If missing, it
      defaults to `utils.get_training_eval_datasets`.
    fallback_init_rng: A random seed used for parameter initialization during
      model re-loading when utils.RestoreCheckpointConfig.fallback_to_scratch is
      set to True. If None, parameter initialization is not allowed during model
      loading and having fallback_to_scratch enabled will result in an error.
    use_orbax: if True, uses Orbax for checkpointing. Experimental feature.
  """
  jax.monitoring.record_event('/jax/t5x/evaluate/beacon')
  logging.info('Process ID: %d', jax.process_index())
  if dataset_cfg.module:
    utils.import_module(dataset_cfg.module)
  batch_size = dataset_cfg.batch_size

  summarize_config_fn(model_dir=output_dir, summary_writer=None, step=0)

  evaluator = InferenceEvaluator(
      dataset_cfg,
      inference_evaluator_cls,
      model,
      partitioner,
      log_dir=output_dir)
  if not evaluator.eval_tasks:
    raise ValueError(
        f"'{dataset_cfg.mixture_or_task_name}' has no metrics for evaluation, "
        "or this mixture/task doesn't have provided split.")

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

  if training_evaluator_cls:
    data_layout = partitioner.get_data_layout(dataset_cfg.batch_size)
    train_eval_datasets = train_eval_get_dataset_fn(  # pytype:disable=missing-parameter
        dataset_cfg,
        data_layout.shard_id,
        data_layout.num_shards,
        feature_converter_cls=model.FEATURE_CONVERTER_CLS,
    )

    train_evaluator = training_evaluator_cls(  # pytype:disable=wrong-arg-types
        model=model,
        train_state=None,  # Will replace later.
        partitioner=partitioner,
        train_state_axes=train_state_axes,
        eval_names=train_eval_datasets.keys(),
        summary_dir=output_dir,
        rng=jax.random.PRNGKey(0),  # unused
        learning_rate_fn=None,  # unused
        num_microbatches=None,  # unused
    )

  def _maybe_run_train_eval(train_state: train_state_lib.TrainState):
    if training_evaluator_cls:
      train_evaluator.train_state = train_state
      train_evaluator.eval(
          {
              task: ds.as_numpy_iterator()
              for task, ds in train_eval_datasets.items()
          }
      )

  # Disable strictness since we are dropping the optimizer state.
  restore_checkpoint_cfg.strict = False

  # Skip checkpoints that have already been evaluated.
  eval_ckpt_path = os.path.join(
      output_dir, f'eval.{dataset_cfg.mixture_or_task_name}.ckpt')
  if restore_checkpoint_cfg.mode == 'all' and gfile.exists(eval_ckpt_path):
    logging.info('Found evaluation checkpoint: %s', eval_ckpt_path)

    ckpt_dirs = ([restore_checkpoint_cfg.path] if isinstance(
        restore_checkpoint_cfg.path, str) else restore_checkpoint_cfg.path)
    ckpt_paths = set()
    for ckpt_dir in ckpt_dirs:
      if not gfile.isdir(ckpt_dir):
        raise ValueError(
            f"Checkpoint path '{ckpt_dir}' must be a valid directory when "
            "using restore mode 'all'."
        )
      ckpt_paths.update(
          checkpoints.get_checkpoint_dir(ckpt_dir, step)
          for step in checkpoints.all_steps(ckpt_dir))

    evaluated_ckpt_paths = _load_evaluated_ckpt_paths(eval_ckpt_path)

    logging.info(
        'Skipping evaluated checkpoints:\n %s',
        '\n '.join(_sorted_ckpt_paths(ckpt_paths & evaluated_ckpt_paths)),
    )
    restore_checkpoint_cfg.mode = 'specific'
    restore_checkpoint_cfg.path = _sorted_ckpt_paths(
        ckpt_paths - evaluated_ckpt_paths
    )

  if fallback_init_rng is not None:
    fallback_init_rng = jax.random.PRNGKey(fallback_init_rng)
  restore_cfg, ckpt_paths = utils.get_first_valid_restore_config_and_paths(
      [restore_checkpoint_cfg]
  )
  for ckpt_path in ckpt_paths:
    train_state, _ = utils.create_checkpoint_manager_and_restore(
        train_state_initializer,
        partitioner,
        restore_cfg,
        ckpt_path,
        fallback_init_rng,
        use_orbax=use_orbax,
    )
    if train_state is None:
      raise ValueError('Failed to restore checkpoint.')

    # ----------------------------------------------------------------------------
    # Main evaluation loop
    # ----------------------------------------------------------------------------

    # Run final evaluation (with decoding) on the full eval dataset.
    host_step = int(utils.get_local_data(train_state.step))
    _maybe_run_train_eval(train_state)
    all_metrics = evaluator.evaluate(train_state, train_state_axes)
    all_metrics.result()  # Ensure metrics are finished being computed.
    # Wait until computations are done before continuing.
    utils.sync_global_devices(f'step_{host_step}:complete')
    if jax.process_index() == 0:
      # Read/write/replace rather than append to avoid filesystem issue.
      evaluated_ckpt_paths = _load_evaluated_ckpt_paths(eval_ckpt_path)
      evaluated_ckpt_paths.add(ckpt_path)
      with gfile.GFile(eval_ckpt_path, 'w') as f:
        f.write('\n'.join(_sorted_ckpt_paths(evaluated_ckpt_paths)))

  logging.info('Finished.')


if __name__ == '__main__':
  # pylint:disable=g-import-not-at-top
  from absl import app
  from absl import flags
  import fiddle as fdl
  import gin
  from t5x import config_utils

  FLAGS = flags.FLAGS

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

    if config_utils.using_fdl():
      config = config_utils.config_with_fiddle(evaluate)
      evaluate_using_fiddle = fdl.build(config)
      evaluate_using_fiddle()
    else:
      # Create gin-configurable version of `eval`.
      evaluate_using_gin = gin.configurable(evaluate)

      gin_utils.parse_gin_flags(
          # User-provided gin paths take precedence if relative paths conflict.
          FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
          FLAGS.gin_file,
          FLAGS.gin_bindings,
      )
      evaluate_using_gin()

  config_utils.run(main)
