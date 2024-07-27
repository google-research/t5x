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

r"""Script to pretrain or finetune in JAX using a SeqIO pipeline.

"""


# pylint: disable=g-import-not-at-top

import functools
import gc
import math
import os
import time
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Type

# Set Linen to add profiling information when constructing Modules.
# Must be set before flax imports.
os.environ['FLAX_PROFILE'] = 'true'
# TODO(adarob): Re-enable once users are notified and tests are updated.
os.environ['FLAX_LAZY_RNG'] = 'no'
from absl import logging
from clu import metric_writers
import jax
from jax import random
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
import seqio
from t5x import checkpoints
from t5x import eval as eval_lib
from t5x import models
from t5x import partitioning
from t5x import train_state as train_state_lib
from t5x import trainer as trainer_lib
from t5x import utils
import tensorflow as tf
# pylint:enable=g-import-not-at-top

# pylint:enable=g-import-not-at-top

# Automatically search for gin files relative to the T5X package.
_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]
P = partitioning.PartitionSpec
# Special key that used to distinguish train metrics.
TRAIN_METRIC_KEY = 'train'
# String keys that is acceptable from config.
_ACTION_KEYS = frozenset(trainer_lib.ActionMode.__members__.keys())
_IMPORT_TIME = time.time()


def run_actions(
    mode: trainer_lib.ActionMode,
    actions: trainer_lib.ActionMapType,
    train_state: train_state_lib.TrainState,
    metrics_by_task: Mapping[str, trainer_lib.MetricValueMapType],
) -> bool:
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
    model: models.BaseModel,
    train_dataset_cfg: utils.DatasetConfig,
    train_eval_dataset_cfg: Optional[utils.DatasetConfig],
    infer_eval_dataset_cfg: Optional[utils.DatasetConfig],
    checkpoint_cfg: utils.CheckpointConfig,
    partitioner: partitioning.BasePartitioner,
    trainer_cls: trainer_lib.BaseTrainerConstructor,
    model_dir: str,
    total_steps: int,
    eval_steps: int,
    eval_period: int,
    relative_steps: Optional[int] = None,
    stats_period: Optional[int] = None,
    random_seed: Optional[int],
    use_hardware_rng: bool = False,
    summarize_config_fn: Callable[
        [str, metric_writers.MetricWriter, int], None
    ],
    inference_evaluator_cls: utils.EvaluatorConstructor = seqio.Evaluator,
    get_dataset_fn: utils.GetDatasetCallable = utils.get_dataset,
    concurrent_metrics: bool = True,
    actions: Optional[Mapping[str, Sequence[trainer_lib.BaseAction]]] = None,
    train_eval_get_dataset_fn: utils.GetEvalDatasetCallable = utils.get_training_eval_datasets,
    run_eval_before_training: bool = False,
    train_state_initializer_cls: Type[
        utils.TrainStateInitializer
    ] = utils.TrainStateInitializer,
    use_orbax: bool = True,
    verify_matching_vocabs_fn: Optional[
        Callable[[utils.DatasetConfig, models.BaseTransformerModel], None]
    ] = utils.verify_matching_vocabs,
    gc_period: int = 0,
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
      the checkpoint. If this is set to the starting step from the checkpoint,
      the model will not be compiled for training and training will not be run.
      This can be used in conjunction with `run_eval_before_training` to only
      evaluate a model.
    eval_steps: The number of batches to process for each train-eval loop.
    eval_period: The number of train steps between each evaluation (both
      train-eval and infer-eval).
    relative_steps: The number of train steps to take relative to the current
      step loaded from the checkpoint. If this is set, total_steps is ignored.
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
    inference_evaluator_cls: seqio.Evaluator class to use for inference
      evaluation, potentially with bound configuration args.
    get_dataset_fn: The callable use to get the train and train-eval datasets
      based on the DatasetConfig and shard information.
    concurrent_metrics: If True, allow metrics computation and logging to
      overlap with training. Will likely result in additional TPU memory usage.
    actions: A mapping of actions that runs after train, eval or infer_eval, to
      inspect the model and perform useful operations, e.g., early stopping. The
      key must have a 1:1 mapping to ActionMode enum. For EVAL actions to
      actually work, this requires `concurrent_metrics` to be turned off, since
      chaining futures and mutating states concurrently might be error-prone.
    train_eval_get_dataset_fn: Optional callable use to get the train-eval
      datasets based on the DatasetConfig and shard information. If missing, it
      defaults to `utils.get_training_eval_datasets`.
    run_eval_before_training: If True, calculate training eval and inference
      eval metrics before training begins.
    train_state_initializer_cls: t5x.utils.TrainStateInitializer class for
      initializing partitioned TrainState from checkpoints or scratch.
    use_orbax: if True, uses Orbax for checkpointing. Experimental feature.
    verify_matching_vocabs_fn: Function to validate whether the task vocabulary
      matches the model vocabulary, if the model is a BaseTransformerModel
      instance. Should raise an exception on error.
    gc_period: The number of train steps between runs of the garbage collector.
      If 0, the garbage collector will run at the normal frequency. # BEGIN
    training_eval_average_metrics: Averages the metric over the list of tasks
      for training_eval (e.g., {'task_average/accuracy': ['task_a', 'task_b']}).
    infer_eval_average_metrics: Averages the metric over the list of tasks for
      infer_eval (e.g., {'task_average/accuracy': ['task_a', 'task_b']}). # END

  Returns:
    The tuple of (last_step, last_train_state).
  """
  logging.info('Process ID: %d', jax.process_index())
  tf.io.gfile.makedirs(model_dir)

  if use_orbax:
    logging.info('Checkpointing with Orbax enabled.')

  # Each "epoch" of the training loop should be the min of the eval period,
  # checkpoint period or the full training.
  # We compute here to ensure that the eval period and checkpoint period are
  # divisible by this number, otherwise we fail.
  eval_enabled = train_eval_dataset_cfg or infer_eval_dataset_cfg
  eval_period = eval_period if eval_enabled else 0
  checkpoint_period = checkpoint_cfg.save.period if checkpoint_cfg.save else 0
  checkpoint_steps = (
      checkpoint_cfg.save.checkpoint_steps if checkpoint_cfg.save else []
  )

  if use_hardware_rng or random_seed is None:
    logging.info(
        'Using fast RngBitGenerator PRNG for initialization and dropout.'
    )

    if random_seed is None:
      random_seed = multihost_utils.broadcast_one_to_all(np.int32(time.time()))
      logging.info('Random seed not provided, using RNG seed %s', random_seed)
    else:
      logging.warning(
          'When using hardware RNG with a fixed seed, repeatability is only '
          'guaranteed for fixed hardware and partitioning schemes and for a '
          'fixed version of this code and its dependencies.'
      )
    utils.set_hardware_rng_ops()
    rng = random.PRNGKey(random_seed)
  else:
    logging.info(
        'Using seed for initialization and dropout RNG: %d', random_seed
    )
    rng = random.PRNGKey(random_seed)

  init_rng, trainer_rng = random.split(rng, 2)

  # ---------------------------------------------------------------------------
  # Initialize datasets
  # ---------------------------------------------------------------------------

  if train_dataset_cfg.seed and not (
      checkpoint_cfg.save and checkpoint_cfg.save.save_dataset
  ):
    logging.warning(
        'Providing a random seed for the train dataset with '
        '`checkpoint_train_ds=False` is dangerous since each '
        'preemption/restart will cause the dataset to deterministically replay '
        'from the beginning.'
    )

  data_layout = partitioner.get_data_layout(train_dataset_cfg.batch_size)
  ds_shard_id = data_layout.shard_id
  num_ds_shards = data_layout.num_shards

  def _verify_matching_vocabs(cfg: utils.DatasetConfig):
    if verify_matching_vocabs_fn and isinstance(
        model, models.BaseTransformerModel
    ):
      verify_matching_vocabs_fn(cfg, model)

  _verify_matching_vocabs(train_dataset_cfg)

  train_iter = get_dataset_fn(
      train_dataset_cfg, ds_shard_id, num_ds_shards, model.FEATURE_CONVERTER_CLS
  )
  train_iter = utils.prepare_train_iter(
      train_iter,
      checkpoint_cfg=checkpoint_cfg,
      partitioner=partitioner,
      data_layout=data_layout,
  )
  input_shapes = jax.tree.map(
      lambda x: (data_layout.batch_size, *x.shape[1:]),
      train_iter.element_spec,
  )
  input_types = jax.tree.map(lambda x: x.dtype, train_iter.element_spec)

  if train_eval_dataset_cfg:
    _verify_matching_vocabs(train_eval_dataset_cfg)
    train_eval_datasets = train_eval_get_dataset_fn(
        train_eval_dataset_cfg,
        ds_shard_id,
        num_ds_shards,
        eval_steps,
        model.FEATURE_CONVERTER_CLS,
    )  # type: Mapping[str, tf.data.Dataset]
    if not train_eval_datasets:
      logging.warning(
          'No train_eval datasets loaded from config `train_eval_dataset_cfg`: '
          '%s',
          train_eval_dataset_cfg,
      )
  else:
    train_eval_datasets = {}

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
          dtype=checkpoint_cfg.save.dtype if checkpoint_cfg.save else 'float32',
          checkpointer_cls=checkpoint_cfg.save.checkpointer_cls
          if checkpoint_cfg.save
          else checkpoints.Checkpointer,
          # Restore dataset state if it is being saved.
          restore_dataset=(
              checkpoint_cfg.save and checkpoint_cfg.save.save_dataset
          ),
          state_transformation_fns=state_transforms_for_restore,
      )
  ]
  # 2. From a checkpoint specified by `checkpoint_cfg.restore.path`, if set.
  if checkpoint_cfg.restore:
    if checkpoint_cfg.restore.mode == 'all':
      raise ValueError(
          "Restore checkpoint mode 'all' is not supported in training."
      )

    # TODO(dhgarrette): Split "restore" behavior into separate configurations
    #     for the initial restoration for a new run, vs resuming a stopped run.
    if isinstance(checkpoint_cfg.restore.path, str):
      restore_cfgs.append(checkpoint_cfg.restore)
    elif not checkpoint_cfg.restore.path:
      # `path` is an empty (non-`str`) sequence, so there is nothing to restore.
      pass
    else:
      raise ValueError(
          'Restore checkpoint config may only have a single path in training.'
      )

  init_or_restore_tick = time.time()
  train_state_initializer = train_state_initializer_cls(
      optimizer_def=model.optimizer_def,
      init_fn=model.get_initial_variables,
      input_shapes=input_shapes,
      input_types=input_types,
      partitioner=partitioner,
  )

  # May be None, empty
  valid_restore_cfg, restore_paths = (
      utils.get_first_valid_restore_config_and_paths(restore_cfgs)
  )
  if len(restore_paths) > 1:
    raise ValueError('Multiple restore paths not permitted in training.')

  # Skip initialization if neither save nor restore is requested.
  train_state = None
  checkpoint_manager = None
  if valid_restore_cfg or checkpoint_period or checkpoint_steps:
    train_state, checkpoint_manager = (
        utils.create_checkpoint_manager_and_restore(
            train_state_initializer,
            partitioner,
            valid_restore_cfg,
            restore_paths[0] if restore_paths else None,
            init_rng,
            save_checkpoint_cfg=checkpoint_cfg.save,
            model_dir=model_dir,
            ds_iter=train_iter,
            use_orbax=use_orbax,
        )
    )

  # Start warming up the input pipeline in the background. This must happen
  # after input pipeline checkpoints were restored.
  first_batch_ready = train_iter.peek_async()

  # 3. If no checkpoint to restore, init from scratch.
  train_state = train_state or train_state_initializer.from_scratch(init_rng)
  train_state_axes = train_state_initializer.train_state_axes
  init_or_restore_secs = time.time() - init_or_restore_tick
  logging.info(
      'Initialize/restore complete (%.2f seconds).', init_or_restore_secs
  )

  # Log the variable shapes information and write to a file.
  log_file = os.path.join(model_dir, 'model-info.txt')
  utils.log_model_info(
      log_file, train_state_initializer.global_train_state_shape, partitioner
  )

  # Restore step from last checkpoint or set to 0 if training from scratch.
  host_step = int(utils.get_local_data(train_state.step))  # pytype: disable=attribute-error

  if relative_steps:
    total_steps = host_step + relative_steps

  if eval_period or checkpoint_period or gc_period:
    steps_per_epoch = min(
        eval_period or np.inf, checkpoint_period or np.inf, gc_period or np.inf
    )
  else:
    steps_per_epoch = total_steps
  stats_period = stats_period or steps_per_epoch
  if (
      eval_period
      and eval_period % steps_per_epoch
      or checkpoint_period
      and checkpoint_period % steps_per_epoch
      or gc_period
      and gc_period % steps_per_epoch
  ):
    raise ValueError(
        f'Checkpoint period ({checkpoint_period}), eval '
        f'period ({eval_period}), and GC period ({gc_period}) must all be '
        'multiples of each other.'
    )

  # ---------------------------------------------------------------------------
  # Trainer
  # ---------------------------------------------------------------------------

  trainer: trainer_lib.BaseTrainer = trainer_cls(  # pytype: disable=wrong-arg-types
      model=model,
      train_state=train_state,
      partitioner=partitioner,
      train_state_axes=train_state_axes,
      eval_names=train_eval_datasets.keys(),
      summary_dir=model_dir,
      rng=trainer_rng,
  )
  del train_state

  train_metrics = trainer.train_metrics_manager
  summarize_config_fn(model_dir, train_metrics.summary_writer, host_step)

  train_metrics.write_scalar(
      'timing/init_or_restore_seconds', init_or_restore_secs, host_step
  )

  # ----------------------------------------------------------------------------
  # SeqIO (inference-based) evaluation setup
  # ----------------------------------------------------------------------------
  # Init evaluator to set up cached datasets
  evaluator = None
  if infer_eval_dataset_cfg is not None:
    evaluator = eval_lib.InferenceEvaluator(
        infer_eval_dataset_cfg=infer_eval_dataset_cfg,
        inference_evaluator_cls=inference_evaluator_cls,
        model=model,
        partitioner=partitioner,
        log_dir=model_dir,
        verify_matching_vocabs_fn=verify_matching_vocabs_fn,
    )
    if not evaluator.eval_tasks:
      # Skip evaluation.
      evaluator = None

  if actions is None:
    actions = {}

  if set(actions.keys()).difference(_ACTION_KEYS):
    raise ValueError(
        f'actions keys must be one of {_ACTION_KEYS}, but got : '
        f'{actions.keys()}'
    )

  # Transform the string key into proper ActionMode enum.
  actions = {trainer_lib.ActionMode[k]: v for k, v in actions.items()}

  if (
      concurrent_metrics
      and actions.get(trainer_lib.ActionMode.INFER_EVAL, None) is not None
  ):
    logging.warning(
        'Actions for INFER_EVAL will not be triggered when async '
        'metrics computation is enabled'
    )
  if (
      concurrent_metrics
      and actions.get(trainer_lib.ActionMode.TRAIN, None) is not None
  ):
    logging.warning(
        'Actions for TRAIN will not be triggered when async '
        'metrics computation is enabled'
    )

  # ----------------------------------------------------------------------------
  # Setup Eval Utility Functions
  # ----------------------------------------------------------------------------

  def _run_training_eval(first_run: bool = False):
    if first_run:
      logging.info('Compiling training eval loop.')
      trainer.compile_eval({  # pytype: disable=wrong-arg-types  # jax-ndarray
          task: utils.get_zeros_batch_like_dataset(ds)
          for task, ds in train_eval_datasets.items()
      })
    logging.info('Computing training evaluation metrics.')
    eval_batch_iters = {}
    for task, ds in train_eval_datasets.items():
      if isinstance(ds, tf.data.Dataset):
        eval_batch_iters[task] = ds.as_numpy_iterator()
      else:
        eval_batch_iters[task] = ds

    eval_summaries = trainer.eval(eval_batch_iters)
    trainer.stop_training = run_actions(
        trainer_lib.ActionMode.TRAIN_EVAL,  # pytype: disable=wrong-arg-types  # jax-ndarray
        actions,
        trainer.train_state,
        eval_summaries,
    )

  def _run_inference_eval():
    """Run prediction based inference eval."""
    if evaluator is None:
      return
    logging.info('Running inference evaluation.')
    evaluate_tick = time.time()
    all_metrics = evaluator.evaluate(trainer.train_state, train_state_axes)
    if not concurrent_metrics:
      # Ensure metrics are finished being computed.
      all_metrics_done = all_metrics.result() or {}
      trainer.stop_training = run_actions(
          trainer_lib.ActionMode.INFER_EVAL,
          actions,
          trainer.train_state,
          all_metrics_done,
      )
    train_metrics.write_scalar(
        'timing/evaluate_seconds', time.time() - evaluate_tick, host_step
    )

  # Optionally run teacher-forcing training eval and SeqIO inference-base eval
  # before training. Useful for testing how much a model knows before any
  # finetuning.
  if run_eval_before_training:
    if train_eval_datasets:
      logging.info('Running training eval before training.')
      _run_training_eval(first_run=True)
    if evaluator is not None:
      logging.info('Running inference eval before training.')
      _run_inference_eval()

  # Save checkpoints before the training loop starts.
  if checkpoint_period and checkpoint_manager:
    # If not using Orbax, always save checkpoint, otherwise, only save a
    # checkpoint if a checkpoint does not already exist for that step. This is
    # because Orbax will error out if we try to save a checkpoint that already
    # exists.
    if not use_orbax or (
        use_orbax
        and utils.get_local_data(trainer.train_state.step)
        not in checkpoint_manager.all_steps()
    ):
      logging.info('Saving checkpoint before the training loop starts.')
      checkpoint_manager.save(
          trainer.train_state,
          checkpoint_cfg.save.state_transformation_fns,  # pytype: disable=attribute-error
      )

  # If we take manual control of the garbage collector, we need to disable it
  # before starting training.
  if gc_period:
    gc.disable()

  # ----------------------------------------------------------------------------
  # Main training loop
  # ----------------------------------------------------------------------------
  logging.info('Starting training loop.')

  def _cleanup() -> None:
    """Ensures everything has been closed upon completion."""
    trainer.close()
    if evaluator:
      evaluator.close()
    utils.sync_global_devices('complete')
    logging.info('Finished.')

  first_step = host_step

  if total_steps < first_step:
    raise ValueError(
        f'Unexpected total_steps ({total_steps}) < checkpoint step '
        f' ({first_step}).'
    )
  elif total_steps == first_step:
    logging.warning(
        'Total training steps and checkpoint step were both %d, so no training '
        'will be done. If you are only doing evaluation, this is expected. '
        'Stopping now.',
        total_steps,
    )
    _cleanup()
    return host_step, trainer.train_state

  logging.info('Starting main loop over steps %d-%d', first_step, total_steps)

  steps_per_epoch = min(steps_per_epoch, total_steps)
  first_epoch = first_step // steps_per_epoch
  num_epochs = first_epoch + math.ceil(
      (total_steps - first_step) / steps_per_epoch
  )
  logging.info(
      'Training with artificial "epochs" of %d steps.', steps_per_epoch
  )

  logging.info('Compiling train loop.')
  logging.flush()

  def _as_gda(spec):
    dummy = np.ones((data_layout.batch_size, *spec.shape[1:]), spec.dtype)
    return jax.make_array_from_callback(
        dummy.shape,
        jax.sharding.NamedSharding(
            partitioner.mesh, partitioner.data_partition_spec
        ),
        lambda idx: dummy[idx],
    )

  # Construct dummy batch for compiling the model.
  dummy_batch = jax.tree.map(_as_gda, train_iter.element_spec)
  if not isinstance(dummy_batch, Mapping):
    raise ValueError(
        'Training loop expects batches to have type '
        f'Mapping[str, np.ndarray] but got {type(dummy_batch)}.'
    )

  assert isinstance(dummy_batch, Mapping)
  trainer.compile_train(dummy_batch)

  # ----------------------------------------------------------------------------
  # Warmup input pipeline.
  # ----------------------------------------------------------------------------
  train_iter_warmup_tick = time.time()
  # We are cheating here. The input pipeline already started warmup when
  # first_batch_ready was created. The warmup was then interleaved with the
  # model compilation above. We just measure the additional time needed.
  first_batch_ready.result()
  train_iter_warmup_tock = time.time()
  train_metrics.write_scalar(
      'timing/train_iter_warmup',
      train_iter_warmup_tock - train_iter_warmup_tick,
      host_step,
  )

  jax.monitoring.record_event_duration_secs(
      '/jax/t5x/train/time_before_first_step_secs', time.time() - _IMPORT_TIME
  )

  # Current index within checkpoint_steps list for faster lookup runtime and
  # for creating a checkpoint if needed between stats_period iterations.
  checkpoint_steps_index = 0

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
      epoch_end_step = first_step + steps_per_epoch * (epoch - first_epoch + 1)
      epoch_end_step = min(total_steps, epoch_end_step)
      logging.info('Training for %d steps.', epoch_end_step - host_step)
      while host_step < epoch_end_step:
        if trainer.stop_training:
          if checkpoint_period and checkpoint_manager:
            logging.info('Saving a checkpoint before early stopping...')
            checkpoint_manager.save(
                trainer.train_state,
                checkpoint_cfg.save.state_transformation_fns,  # pytype: disable=attribute-error
            )
          logging.info(
              'Stopping training loop early since `stop_training` is requested.'
          )
          break
        inner_num_steps = min(epoch_end_step - host_step, stats_period)

        # first index in checkpoint_steps list will not always be 0 (in cases
        # where first_step is non-zero, for example), so we must iterate to the
        # first un-trained step in checkpoint_steps list to not re-train /
        # save old steps
        checkpoint_steps_index = utils.find_first_checkpoint_step(
            checkpoint_steps_index, checkpoint_steps, first_step, host_step
        )
        # check if inner_num_steps will skip a checkpoint_step that must be
        # saved, if so, then iterate only to that step and save a checkpoint
        # at that step and then continue with further iterations
        is_checkpoint_step = False
        (inner_num_steps, is_checkpoint_step) = utils.find_next_checkpoint_step(
            checkpoint_steps_index,
            inner_num_steps,
            is_checkpoint_step,
            host_step,
            checkpoint_steps,
            epoch_end_step,
            checkpoint_period,
            first_step,
        )
        # Handled separately if this is the overall last step.
        if host_step + inner_num_steps == total_steps:
          is_checkpoint_step = False

        train_summary = trainer.train(
            train_iter, inner_num_steps, start_step=host_step
        )
        if not concurrent_metrics:
          # Note that we always pass the dictionary of `tasks` -> summary so
          # that the actions can be performed without special casing. The
          # only caveat is that train would need its own special `key`
          # given no `task` will be applied.
          trainer.stop_training = run_actions(  # pytype: disable=wrong-arg-types  # jax-ndarray
              trainer_lib.ActionMode.TRAIN,
              actions,
              trainer.train_state,
              {TRAIN_METRIC_KEY: train_summary.result()},
          )

        if is_checkpoint_step and checkpoint_manager:
          logging.info('Saving a checkpoint at specified checkpoint step')
          checkpoint_manager.save(
              trainer.train_state,
              checkpoint_cfg.save.state_transformation_fns,  # pytype: disable=attribute-error
          )
          # Increment the checkpoint_step_index only if a checkpoint was saved.
          if (
              checkpoint_steps
              and checkpoint_steps_index < len(checkpoint_steps) - 1
          ):
            checkpoint_steps_index += 1
        host_step += inner_num_steps
      logging.info('END Train loop.')
    except trainer_lib.PreemptionError as e:
      if checkpoint_period and checkpoint_manager:
        logging.info('Saving emergency checkpoint.')
        checkpoint_manager.save(
            trainer.train_state,
            checkpoint_cfg.save.state_transformation_fns,  # pytype: disable=attribute-error
        )
        checkpoint_manager.wait_until_finished()
        logging.info('Saving emergency checkpoint done.')
      raise e

    step_offset = host_step - first_step

    if gc_period and (final_epoch or step_offset % gc_period == 0):
      gc.collect()

    # Maybe save a checkpoint if step is at period.
    if (
        checkpoint_period
        and (final_epoch or step_offset % checkpoint_period == 0)
        and checkpoint_manager
    ):
      train_summary.result()
      logging.info('Saving checkpoint.')
      checkpoint_tick = time.time()
      # Make sure last train step has completed before starting the clock.
      checkpoint_manager.save(
          trainer.train_state,
          checkpoint_cfg.save.state_transformation_fns,  # pytype: disable=attribute-error
      )
      # `_run_training_eval`` depends upon the result of the checkpoint,
      # thus calling `wait_until_finished()`` here.
      checkpoint_manager.wait_until_finished()
      checkpoint_tock = time.time()
      train_metrics.write_scalar(
          'timing/checkpoint_seconds',
          checkpoint_tock - checkpoint_tick,
          host_step,
      )

    is_eval_epoch = eval_period and (
        final_epoch or step_offset % eval_period == 0
    )

    # Training Evaluation (i.e., with teacher forcing).
    if is_eval_epoch and train_eval_datasets:
      # Maybe less if final step < period.
      first_run = step_offset // eval_period <= 1
      _run_training_eval(first_run and not run_eval_before_training)

    # Inference Evaluation (i.e., with decoding or scoring).
    if is_eval_epoch and evaluator is not None:
      _run_inference_eval()
  if checkpoint_manager:
    checkpoint_manager.close()

  # Wait until computations are done before exiting
  _cleanup()

  if gc_period:
    # Reenable garbage collection to avoid affecting future code executed in
    # the same interpreter.
    gc.enable()

  return host_step, trainer.train_state


if __name__ == '__main__':
  # pylint: disable=g-import-not-at-top
  from absl import app
  from absl import flags
  import fiddle as fdl
  import gin
  from t5x import config_utils
  from t5x import gin_utils
  # pylint: enable=g-import-not-at-top

  FLAGS = flags.FLAGS

  flags.DEFINE_multi_string(
      'gin_file',
      default=None,
      help=(
          'Path to gin configuration file. Multiple paths may be passed and '
          'will be imported in the given order, with later configurations  '
          'overriding earlier ones.'
      ),
  )

  flags.DEFINE_multi_string(
      'gin_bindings', default=[], help='Individual gin bindings.'
  )

  flags.DEFINE_list(
      'gin_search_paths',
      default=['.'],
      help=(
          'Comma-separated list of gin config path prefixes to be prepended '
          'to suffixes given via `--gin_file`. If a file appears in. Only the '
          'first prefix that produces a valid path for each suffix will be '
          'used.'
      ),
  )

  flags.DEFINE_string(
      'tfds_data_dir',
      None,
      'If set, this directory will be used to store datasets prepared by '
      'TensorFlow Datasets that are not available in the public TFDS GCS '
      'bucket. Note that this flag overrides the `tfds_data_dir` attribute of '
      'all `Task`s.',
  )

  flags.DEFINE_list(
      'seqio_additional_cache_dirs',
      [],
      'Directories to search for cached Tasks in addition to defaults.',
  )

  flags.DEFINE_boolean(
      'multiprocess_gpu',
      False,
      help=(
          'Initialize JAX distributed system for multi-host GPU, using '
          '`coordinator_address`, `process_count`, and `process_index`.'
      ),
  )

  flags.DEFINE_string(
      'coordinator_address',
      None,
      help='IP address:port for multi-host GPU coordinator.',
  )

  flags.DEFINE_integer(
      'process_count', None, help='Number of processes for multi-host GPU.'
  )

  flags.DEFINE_integer('process_index', None, help='Index of this process.')
  flags.DEFINE_integer(
      'initialization_timeout',
      None,
      help=(
          'Timeout for jax.distributed.initialize. Default used is the '
          'default as specified in jax.distributed.initialize. '
      ),
  )


  def main(argv: Sequence[str]):
    """Wrapper for pdb post mortems."""
    _main(argv)

  def _main(argv: Sequence[str]):
    """True main function."""
    if len(argv) > 1:
      raise app.UsageError('Too many command-line arguments.')

    # OOM fix. Prevents TF from seeing GPUs to stop conflict with JAX.
    # This must go after InitGoogle(), which is called by
    # gin_utils.run(main).
    tf.config.experimental.set_visible_devices([], 'GPU')


    if FLAGS.multiprocess_gpu:
      logging.info(
          'Initializing distributed system for multi-host GPU:\n'
          '  coordinator_address: %s\n  process_count: %s\n  process_index: %s',
          FLAGS.coordinator_address,
          FLAGS.process_count,
          FLAGS.process_index,
      )

      if FLAGS.initialization_timeout:
        if jax.__version__ < '0.4.15':
          raise ValueError(
              'Specified'
              f' --initialization_timeout={FLAGS.initialization_timeout}, but'
              ' jax=={jax.__version__} does not support this yet. Use'
              ' jax>=0.4.15'
          )
        jax.distributed.initialize(
            FLAGS.coordinator_address,
            FLAGS.process_count,
            FLAGS.process_index,
            initialization_timeout=FLAGS.initialization_timeout,
        )
      else:
        jax.distributed.initialize(
            FLAGS.coordinator_address, FLAGS.process_count, FLAGS.process_index
        )

    if FLAGS.tfds_data_dir:
      seqio.set_tfds_data_dir_override(FLAGS.tfds_data_dir)

    seqio.add_global_cache_dirs(FLAGS.seqio_additional_cache_dirs)


    if config_utils.using_fdl():
      config = config_utils.config_with_fiddle(train)
      train_using_fiddle = fdl.build(config)
      train_using_fiddle()
    else:
      # Create gin-configurable version of `train`.
      train_using_gin = gin.configurable(train)

      gin_utils.parse_gin_flags(
          # User-provided gin paths take precedence if relative paths conflict.
          FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
          FLAGS.gin_file,
          FLAGS.gin_bindings,
      )
      train_using_gin()

    jax.effects_barrier()


  config_utils.run(main)
