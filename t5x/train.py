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

r"""Script to pretrain or finetune in JAX using a SeqIO pipeline.

"""
import functools
import itertools
import math
import os
import time
from typing import Callable, Iterator, Sequence, Mapping, Tuple, Type, Optional

# Set Linen to add profiling information when constructing Modules.
# Must be set before flax imports.
# pylint:disable=g-import-not-at-top
os.environ['FLAX_PROFILE'] = 'true'
# TODO(adarob): Re-enable once users are notified and tests are updated.
os.environ['FLAX_LAZY_RNG'] = 'no'
from absl import logging
from clu import metric_writers
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import seqio
from t5x import models
from t5x import multihost_utils
from t5x import partitioning
from t5x import train_state as train_state_lib
from t5x import trainer as trainer_lib
from t5x import utils
import tensorflow as tf


# Automatically search for gin files relative to the T5X package.
_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]
PyTreeDef = type(jax.tree_structure(None))
P = partitioning.PartitionSpec
# Special key that used to distinguish train metrics.
TRAIN_METRIC_KEY = 'train'
# String keys that is acceptable from config.
_ACTION_KEYS = frozenset(trainer_lib.ActionMode.__members__.keys())


def run_actions(
    mode: trainer_lib.ActionMode,
    actions: Mapping[trainer_lib.ActionMode, Sequence[trainer_lib.BaseAction]],
    train_state: train_state_lib.TrainState,
    metrics_by_task: Mapping[str, trainer_lib.MetricValueMapType]) -> bool:
  """Invokes all actions on the given mode on host 0, then broadcasts to all.

  Args:
    mode: The mode to run the actions. e.g., if mode is `train`, only actions
      configured to run with `train` mode will be invoked.
    actions: A mapping of actions that runs after train, eval or infer_eval, to
      inspect the model and perform useful operations, e.g., early stopping.
    train_state: The current train_state of the trainer.
    metrics_by_task: A map of metrics keyed by task name.

  Returns:
    A bool indicating whether training should be halted.

  Raises:
    RuntimeError: When the metrics processed on host 0 is None.
  """
  stop_training = False
  if jax.process_index() == 0:
    if not metrics_by_task:
      raise RuntimeError('Metric is unexpectedly empty on process 0')
    for action in actions.get(mode, []):
      stop_training |= action.run(train_state, metrics_by_task=metrics_by_task)
  # Broadcast result from host 0 to others.
  return bool(multihost_utils.broadcast_one_to_all(jnp.array(stop_training)))


def train(
    *,
    model: models.BaseTransformerModel,
    train_dataset_cfg: utils.DatasetConfig,
    train_eval_dataset_cfg: Optional[utils.DatasetConfig],
    infer_eval_dataset_cfg: Optional[utils.DatasetConfig],
    checkpoint_cfg: utils.CheckpointConfig,
    partitioner: partitioning.BasePartitioner,
    trainer_cls: Type[trainer_lib.BaseTrainer],
    model_dir: str,
    total_steps: int,
    eval_steps: int,
    eval_period: int,
    stats_period: Optional[int] = None,
    random_seed: Optional[int],
    use_hardware_rng: bool = False,
    summarize_config_fn: Callable[[str, metric_writers.MetricWriter, int],
                                  None],
    inference_evaluator_cls: Type[seqio.Evaluator] = seqio.Evaluator,
    get_dataset_fn: utils.GetDatasetCallable = utils.get_dataset,
    concurrent_metrics: bool = True,
    actions: Optional[Mapping[str, Sequence[trainer_lib.BaseAction]]] = None,
    train_eval_get_dataset_fn: Optional[utils.GetDatasetCallable] = None
) -> Tuple[int, train_state_lib.TrainState]:
  """Train function.

  Args:
    model: The model object to use for training.
    train_dataset_cfg: Specification for the dataset to train with.
    train_eval_dataset_cfg: Specification for the dataset to evaluate with using
      the train metrics and no inference (e.g., uses teacher forcing). If None,
      train eval is disabled.
    infer_eval_dataset_cfg: Specification for the dataset to evaluate with using
      the inference metrics (e.g., uses sampled decoding). If None, inference
      eval is disabled.
    checkpoint_cfg: Specification for saving and restoring model parameters and
      dataset state to/from checkpoints.
    partitioner: Partitioner for model parameters and data across devices.
    trainer_cls: An implementation of BaseTrainer.
    model_dir: Path of directory to store checkpoints and metric summaries.
    total_steps: The step number to stop training after. The number of actual
      steps trained in this run will be this number minus the starting step from
      the checkpoint.
    eval_steps: The number of batches to process for each train-eval loop.
    eval_period: The number of train steps between each evaluation (both
      train-eval and infer-eval).
    stats_period: The number of train steps between writing scalar stats. If
      None, defaults to eval_period.
    random_seed: A random seed to use for dropout and initialization. If None, a
      fast, non-deterministic hardware-based RNG is used.
    use_hardware_rng: Whether to force using the RngBitGenerator based hardware
      rng, which takes seeds and acts similarly to software PRNG in that it
      should be seed-deterministic. The new RngBitGenerator custom PRNG system
      should be reproducible for a given sharding, but the numbers will change
      for different shardings of the same model.
    summarize_config_fn: A function that takes in the model directory, a
      SummaryWriter, and the step number, and writes a summary of the
      configuration.
    inference_evaluator_cls: seqio.Evaluator class to use for inference
      evaluation, potentially with bound configuration args.
    get_dataset_fn: The callable use to get the train and train-eval datasets
      based on the DatasetConfig and shard information.
    concurrent_metrics: If True, allow metrics computation and logging to
      overlap with training. Will likely result in additional TPU memory usage.
    actions: A mapping of actions that runs after train, eval or infer_eval, to
      inspect the model and perform useful operations, e.g., early stopping. The
      key must have a 1:1 mapping to ActionMode enum. For EVAL actions to
        actually work, this requires `concurrent_metrics` to be turned off,
        since chaining futures and mutating states concurrently might be
        error-prone.
    train_eval_get_dataset_fn: Optional callable use to get the train-eval
      datasets based on the DatasetConfig and shard information. If missing, it
      defaults to `get_dataset_fn`.

  Returns:
    The tuple of (last_step, last_train_state).
  """
  logging.info('Process ID: %d', jax.process_index())
  tf.io.gfile.makedirs(model_dir)

  # Each "epoch" of the training loop should be the min of the eval period,
  # checkpoint period or the full training.
  # We compute here to ensure that the eval period and checkpoint period are
  # divisible by this number, otherwise we fail.
  eval_enabled = (train_eval_dataset_cfg or infer_eval_dataset_cfg)
  eval_period = eval_period if eval_enabled else 0
  checkpoint_period = checkpoint_cfg.save.period if checkpoint_cfg.save else 0
  if eval_period or checkpoint_period:
    steps_per_epoch = min(eval_period or np.inf, checkpoint_period or np.inf)
  else:
    steps_per_epoch = total_steps
  stats_period = stats_period or steps_per_epoch
  if (eval_period and eval_period % steps_per_epoch or
      checkpoint_period and checkpoint_period % steps_per_epoch):
    raise ValueError(
        f'Checkpoint period ({checkpoint_period}) must evenly divide eval '
        f'period ({eval_period}), or vice-versa.')

  if use_hardware_rng or random_seed is None:
    logging.info(
        'Using fast RngBitGenerator PRNG for initialization and dropout.')

    if random_seed is None:
      random_seed = multihost_utils.broadcast_one_to_all(np.int32(time.time()))
      logging.info('Random seed not provided, using RNG seed %s', random_seed)
    else:
      logging.warning(
          'When using hardware RNG with a fixed seed, repeatability is only '
          'guaranteed for fixed hardware and partitioning schemes and for a '
          'fixed version of this code and its dependencies.')
    utils.set_hardware_rng_ops()
    rng = random.PRNGKey(random_seed)
  else:
    logging.info('Using seed for initialization and dropout RNG: %d',
                 random_seed)
    rng = random.PRNGKey(random_seed)

  init_rng, trainer_rng = random.split(rng, 2)

  # ---------------------------------------------------------------------------
  # Initialize datasets
  # ---------------------------------------------------------------------------

  if (train_dataset_cfg.seed and
      not (checkpoint_cfg.save or checkpoint_cfg.save.save_dataset)):
    logging.warning(
        'Providing a random seed for the train dataset with '
        '`checkpoint_train_ds=False` is dangerous since each '
        'preemption/restart will cause the dataset to deterministically replay '
        'from the beginning.')

  data_layout = partitioner.get_data_layout(train_dataset_cfg.batch_size)
  ds_shard_id = data_layout.shard_id
  num_ds_shards = data_layout.num_shards

  def _verify_matching_vocabs(cfg: utils.DatasetConfig):
    ds_vocabs = utils.get_vocabulary(cfg)
    if (ds_vocabs[0] != model.input_vocabulary or
        ds_vocabs[1] != model.output_vocabulary):
      raise ValueError(f'Model and Task vocabularies do not match:\n'
                       f'  task={cfg.mixture_or_task_name}\n'
                       f'  ds_vocabs=({ds_vocabs[0]}, {ds_vocabs[1]})\n'
                       f'  model.input_vocabulary={model.input_vocabulary}\n'
                       f'  model.output_vocabulary={model.output_vocabulary}\n')

  _verify_matching_vocabs(train_dataset_cfg)

  train_ds = get_dataset_fn(train_dataset_cfg, ds_shard_id, num_ds_shards,
                            model.FEATURE_CONVERTER_CLS)

  if train_eval_dataset_cfg:
    _verify_matching_vocabs(train_eval_dataset_cfg)
    train_eval_datasets = utils.get_training_eval_datasets(
        train_eval_dataset_cfg,
        ds_shard_id,
        num_ds_shards,
        eval_steps,
        model.FEATURE_CONVERTER_CLS,
        get_dataset_fn=train_eval_get_dataset_fn if train_eval_get_dataset_fn
        is not None else get_dataset_fn)  # type: Mapping[str, tf.data.Dataset]
    if not train_eval_datasets:
      logging.warning(
          'No train_eval datasets loaded from config `train_eval_dataset_cfg`: '
          '%s', train_eval_dataset_cfg)
  else:
    train_eval_datasets = {}

  # Initialize optimizer, maybe from an existing checkpoint.
  checkpointable_train_iter: tf.data.Iterator = iter(train_ds)  # pytype:disable=annotation-type-mismatch
  train_iter: Iterator[trainer_lib.BatchType] = map(
      lambda x: jax.tree_map(np.array, x), checkpointable_train_iter)

  # The manner in which parameters are initialized follows this order of
  # preference:
  #  1. From a T5X checkpoint in `model_dir`, if one exists.
  #  2. From a T5X or TF checkpoint specified by `cfg.path`, if set.
  #  3. From scratch using `init_fn`.

  # 1. From a T5X checkpoint in `model_dir`, if one exists.
  if checkpoint_cfg.restore is not None:
    state_transforms_for_restore = [
        functools.partial(fn, is_resuming=True)
        for fn in checkpoint_cfg.restore.state_transformation_fns
    ]
  else:
    state_transforms_for_restore = []
  restore_cfgs = [
      utils.RestoreCheckpointConfig(
          path=model_dir,
          mode='latest',
          dtype=checkpoint_cfg.save.dtype,
          checkpointer_cls=checkpoint_cfg.save.checkpointer_cls,
          # Restore dataset state if it is being saved.
          restore_dataset=(checkpoint_cfg.save and
                           checkpoint_cfg.save.save_dataset),
          state_transformation_fns=state_transforms_for_restore)
  ]
  # 2. From a checkpoint specified by `checkpoint_cfg.restore.path`, if set.
  if checkpoint_cfg.restore:
    if checkpoint_cfg.restore.mode == 'all':
      raise ValueError(
          "Restore checkpoint mode 'all' is not supported in training.")

    # TODO(dhgarrette): Split "restore" behavior into separate configurations
    #     for the initial restoration for a new run, vs resuming a stopped run.
    if isinstance(checkpoint_cfg.restore.path, str):
      restore_cfgs.append(checkpoint_cfg.restore)
    elif not checkpoint_cfg.restore.path:
      # `path` is an empty (non-`str`) sequence, so there is nothing to restore.
      pass
    else:
      raise ValueError(
          'Restore checkpoint config may only have a single path in training.')

  # Need to use full batch size.
  input_shapes = {
      k: (data_layout.batch_size, *v.shape[1:])
      for k, v in train_ds.element_spec.items()
  }
  input_types = {
      k: v.dtype.as_numpy_dtype() for k, v in train_ds.element_spec.items()
  }
  init_or_restore_tick = time.time()
  train_state_initializer = utils.TrainStateInitializer(
      optimizer_def=model.optimizer_def,
      init_fn=model.get_initial_variables,
      input_shapes=input_shapes,
      input_types=input_types,
      partitioner=partitioner)
  #  3. From scratch using `init_fn`.
  train_state = train_state_initializer.from_checkpoint_or_scratch(
      restore_cfgs, init_rng=init_rng, ds_iter=checkpointable_train_iter)
  train_state_axes = train_state_initializer.train_state_axes
  init_or_restore_secs = time.time() - init_or_restore_tick
  logging.info('Initialize/restore complete (%.2f seconds).',
               init_or_restore_secs)

  # Log the variable shapes information and write to a file.
  log_file = os.path.join(model_dir, 'model-info.txt')
  utils.log_model_info(log_file,
                       train_state_initializer.global_train_state_shape,
                       partitioner)

  if checkpoint_period:
    checkpointer = checkpoint_cfg.save.checkpointer_cls(
        train_state=train_state_initializer.global_train_state_shape,
        partitioner=partitioner,
        checkpoints_dir=model_dir,
        dataset_iterator=(checkpointable_train_iter
                          if checkpoint_cfg.save.save_dataset else None),
        save_dtype=checkpoint_cfg.save.dtype,
        keep=checkpoint_cfg.save.keep)

  # Restore step from last checkpoint or set to 0 if training from scratch.
  host_step = int(train_state.step)

  # ---------------------------------------------------------------------------
  # Trainer
  # ---------------------------------------------------------------------------

  trainer: trainer_lib.BaseTrainer = trainer_cls(
      model=model,
      train_state=train_state,
      partitioner=partitioner,
      train_state_axes=train_state_axes,
      eval_names=train_eval_datasets.keys(),
      summary_dir=model_dir,
      rng=trainer_rng)
  del train_state

  train_metrics = trainer.train_metrics_manager
  summarize_config_fn(model_dir, train_metrics.summary_writer, host_step)

  train_metrics.write_scalar('timing/init_or_restore_seconds',
                             init_or_restore_secs, host_step)

  # ----------------------------------------------------------------------------
  # SeqIO (inference-based) evaluation setup
  # ----------------------------------------------------------------------------
  # Init evaluator to set up cached datasets
  evaluator = None
  if infer_eval_dataset_cfg is not None:
    _verify_matching_vocabs(infer_eval_dataset_cfg)
    evaluator = inference_evaluator_cls(
        log_dir=os.path.join(model_dir, 'inference_eval'),
        mixture_or_task_name=infer_eval_dataset_cfg.mixture_or_task_name,
        feature_converter=model.FEATURE_CONVERTER_CLS(pack=False),
        eval_split=infer_eval_dataset_cfg.split,
        use_cached=infer_eval_dataset_cfg.use_cached,
        seed=infer_eval_dataset_cfg.seed,
        sequence_length=infer_eval_dataset_cfg.task_feature_lengths,
        use_memory_cache=infer_eval_dataset_cfg.use_memory_cache)
    if not evaluator.eval_tasks:
      # Skip evaluaton.
      evaluator = None

  if evaluator is not None:
    predict_fn = utils.get_infer_fn(
        infer_step=model.predict_batch,
        batch_size=infer_eval_dataset_cfg.batch_size,
        train_state_axes=train_state_axes,
        partitioner=partitioner)

    score_fn = utils.get_infer_fn(
        infer_step=model.score_batch,
        batch_size=infer_eval_dataset_cfg.batch_size,
        train_state_axes=train_state_axes,
        partitioner=partitioner)

  if actions is None:
    actions = {}

  if set(actions.keys()).difference(_ACTION_KEYS):
    raise ValueError(f'actions keys must be one of {_ACTION_KEYS}, but got : '
                     f'{actions.keys()}')

  # Transform the string key into proper ActionMode enum.
  actions = {trainer_lib.ActionMode[k]: v for k, v in actions.items()}

  if concurrent_metrics and actions.get(trainer_lib.ActionMode.INFER_EVAL,
                                        None) is not None:
    logging.warning('Actions for INFER_EVAL will not be triggered when async '
                    'metrics computation is enabled')
  if concurrent_metrics and actions.get(trainer_lib.ActionMode.TRAIN,
                                        None) is not None:
    logging.warning('Actions for TRAIN will not be triggered when async '
                    'metrics computation is enabled')

  # ----------------------------------------------------------------------------
  # Main training loop
  # ----------------------------------------------------------------------------
  logging.info('Starting training loop.')

  first_step = host_step

  if total_steps < first_step:
    raise ValueError(
        f'Unexpected total_steps ({total_steps}) < checkpoint step '
        f' ({first_step}).')
  logging.info('Starting main loop over steps %d-%d', first_step, total_steps)

  steps_per_epoch = min(steps_per_epoch, total_steps)
  first_epoch = first_step // steps_per_epoch
  num_epochs = first_epoch + math.ceil(
      (total_steps - first_step) / steps_per_epoch)
  logging.info('Training with artificial "epochs" of %d steps.',
               steps_per_epoch)

  # Kickstart training dataset and compile train loop.
  logging.info('Kickstarting train dataset prefetch.')
  logging.flush()

  ds_tick = time.time()
  # Get first batch to warm up the dataset pipeline.
  first_batch = next(train_iter)
  # Prepend first batch back to iterator to be used by trainer.
  train_iter = itertools.chain([first_batch], train_iter)
  train_metrics.write_scalar('timing/dataset_warmup_seconds',
                             time.time() - ds_tick, host_step)
  logging.info('Compiling train loop.')
  logging.flush()
  trainer.compile_train(first_batch)

  # Main Loop over "epochs".
  for epoch in range(first_epoch, num_epochs):
    final_epoch = epoch == num_epochs - 1
    logging.info('Epoch %d of %d', epoch, num_epochs)

    # `stop_training` is requested, break out the main loop immediately.
    if trainer.stop_training:
      break

    logging.info('BEGIN Train loop.')
    try:
      # Until the last epoch, `num_steps = steps_per_epoch`
      num_steps = min(total_steps - host_step, steps_per_epoch)
      epoch_end_step = host_step + num_steps
      logging.info('Training for %d steps.', num_steps)
      while host_step < epoch_end_step:
        if trainer.stop_training:
          logging.info('Saving a checkpoint before early stopping...')
          checkpointer.save(trainer.train_state,
                            checkpoint_cfg.save.state_transformation_fns)
          logging.info('Stopping training loop early since `stop_training` is '
                       'requested.')
          break

        inner_num_steps = min(epoch_end_step - host_step, stats_period)
        train_summary = trainer.train(
            train_iter, inner_num_steps, start_step=host_step)
        if not concurrent_metrics:
          # Note that we always pass the dictionary of `tasks` -> summary so
          # that the actions can be performed without special casing. The only
          # caveat is that train would need its own special `key` given no
          # `task` will be applied.
          trainer.stop_training = run_actions(
              trainer_lib.ActionMode.TRAIN, actions, trainer.train_state,
              {TRAIN_METRIC_KEY: train_summary.result()})

        host_step += inner_num_steps
      logging.info('END Train loop.')
    except trainer_lib.PreemptionError as e:
      logging.info('Saving emergency checkpoint.')
      checkpointer.save(trainer.train_state,
                        checkpoint_cfg.save.state_transformation_fns)
      logging.info('Saving emergency checkpoint done.')
      raise e

    step_offset = host_step - first_step

    # Maybe save a checkpoint.
    if checkpoint_period and (final_epoch or
                              step_offset % checkpoint_period == 0):
      # Make sure last train step has completed before starting the clock.
      train_summary.result()
      logging.info('Saving checkpoint.')
      checkpoint_tick = time.time()
      checkpointer.save(trainer.train_state,
                        checkpoint_cfg.save.state_transformation_fns)
      checkpoint_tock = time.time()
      train_metrics.write_scalar('timing/checkpoint_seconds',
                                 checkpoint_tock - checkpoint_tick, host_step)

    is_eval_epoch = eval_period and (final_epoch or
                                     step_offset % eval_period == 0)

    # Training Evaluation (i.e., with teacher forcing).
    if is_eval_epoch and train_eval_datasets:
      if step_offset // eval_period <= 1:  # Maybe less if final step < period.
        # Compile before the first run.
        logging.info('Compiling training eval loop.')
        trainer.compile_eval({
            task: utils.get_zeros_batch_like_dataset(ds)
            for task, ds in train_eval_datasets.items()
        })
      logging.info('Computing training evaluation metrics.')
      eval_batch_iters = {
          task: ds.as_numpy_iterator()
          for task, ds in train_eval_datasets.items()
      }
      eval_summaries = trainer.eval(eval_batch_iters)
      trainer.stop_training = run_actions(trainer_lib.ActionMode.TRAIN_EVAL,
                                          actions, trainer.train_state,
                                          eval_summaries)

    # Inference Evaluation (i.e., with decoding or scoring).
    if evaluator is not None:
      logging.info('Running inference evaluation.')
      evaluate_tick = time.time()
      all_metrics, _, _ = evaluator.evaluate(
          compute_metrics=jax.process_index() == 0,
          step=host_step,
          predict_fn=functools.partial(
              predict_fn, train_state=trainer.train_state),
          score_fn=functools.partial(score_fn, train_state=trainer.train_state))
      if not concurrent_metrics:
        # Ensure metrics are finished being computed.
        all_metrics_done = all_metrics.result() or {}
        trainer.stop_training = run_actions(trainer_lib.ActionMode.INFER_EVAL,
                                            actions, trainer.train_state,
                                            all_metrics_done)
      train_metrics.write_scalar('timing/evaluate_seconds',
                                 time.time() - evaluate_tick, host_step)
      # Make sure the metrics are flushed before exiting.
      if final_epoch:
        all_metrics.result()

  # Wait until computations are done before exiting
  logging.info('Finished.')
  multihost_utils.sync_devices('complete')

  return host_step, trainer.train_state


if __name__ == '__main__':
  # pylint: disable=g-import-not-at-top
  from absl import app
  from absl import flags
  import gin
  from t5x import gin_utils
  # pylint: enable=g-import-not-at-top

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

    # Create gin-configurable version of `train`.
    train_using_gin = gin.configurable(train)

    gin_utils.parse_gin_flags(
        # User-provided gin paths take precedence if relative paths conflict.
        FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
        FLAGS.gin_file,
        FLAGS.gin_bindings)
    train_using_gin()


  gin_utils.run(main)
