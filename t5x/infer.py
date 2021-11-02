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
r"""This script runs inference on a T5X-compatible model.

"""
# pyformat: enable
# pylint:enable=line-too-long

import concurrent.futures
import functools
import hashlib
import json
import os
import re
import shutil
import time
from typing import Any, Callable, Iterator, List, Mapping, Optional, Sequence, Tuple

from absl import logging
import jax
import jax.numpy as jnp
import seqio
from t5x import models
from t5x import multihost_utils
from t5x import partitioning
from t5x import utils
import tensorflow as tf
from tensorflow.io import gfile

_DEFAULT_GIN_SEARCH_PATHS = []

AUTOTUNE = tf.data.experimental.AUTOTUNE


class FailFastThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
  """Wrapper for ThreadPoolExecutor that crashes main thread on exceptions.

  NOTE: this class should be used only from the main thread.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._incomplete_futures: List[concurrent.futures.Future] = []

  def check_for_exceptions(self, wait: bool = False):
    """Raises any exceptions from complete futures on the main thread."""
    still_incomplete_futures = []
    for future in self._incomplete_futures:
      try:
        exception = future.exception(timeout=0 if wait else None)
      except concurrent.futures.TimeoutError:
        still_incomplete_futures.append(future)
      if exception is not None:
        raise exception

    self._incomplete_futures = still_incomplete_futures

  def submit(self, *args, **kwargs) -> concurrent.futures.Future:
    """Submit function to threadpool, capturing the returned future."""
    future = super().submit(*args, **kwargs)
    self._incomplete_futures.append(future)
    self.check_for_exceptions(wait=False)
    return future

  def shutdown(self, *args, wait: bool = False, **kwargs):
    self.check_for_exceptions(wait=wait)
    super().shutdown(*args, **kwargs)


def create_task_from_tfexample_file(
    paths: Sequence[str], file_type: str, inputs_key: str,
    targets_key: Optional[str], features: Mapping[str, seqio.Feature]) -> str:
  """Registers ad-hoc Task for file-based dataset of TFExamples.

  Args:
    paths: Input file paths; all files should have type `file_type` and contain
      binary-serialized TFExample protos.
    file_type: Input file type; e.g., 'tfrecord', 'recordio', 'sstable'. For
      keyed formats like 'sstable', we ignore the keys and use only the values.
    inputs_key: Name of TFExample feature containing the input text for T5X. The
      value of this feature should be a UTF8-encoded string.
    targets_key: Optional name of a TFExample feature containing the target text
      (relevant only in scoring mode). The value of this feature should be a
      UTF8-encoded string.
    features: Should have entries for keys 'inputs' and (if targets_key is not
      None) 'targets', mapping to `seqio.Feature` objects that specify
      attributes like vocabulary, add_eos, etc. These attributes are used for
      preprocessing and featurizing the input text.

  Returns:
    Name of the newly-registered Task. This Task has a split named 'infer' that
    contains the preprocessed and featurized input dataset.
  """
  # tf.io.gfile.glob supports lists, in contrast to gfile.glob.
  files = tf.io.gfile.glob(paths)
  if files:
    logging.info('Using tfexample files %s', files)
  else:
    # Fail early if there's something wrong with the input file pattern.
    raise ValueError('Missing or invalid paths: %s' % paths)
  reader = {
      'tfrecord':
          tf.data.TFRecordDataset,
  }[file_type]

  # TODO(adarob): Remove after b/180658446 is resolved.
  def reserialize_tfexample(x):

    def _reserialize(s):
      ex = tf.train.Example()
      ex.ParseFromString(s)
      return ex.SerializeToString()

    return tf.compat.v1.py_func(
        _reserialize, inp=[x], Tout=tf.string, stateful=False)

  def reserialize_reader(filenames):
    return reader(filenames).map(
        reserialize_tfexample, num_parallel_calls=AUTOTUNE)

  feature_description = {inputs_key: tf.io.FixedLenFeature([], tf.string)}
  if targets_key:
    feature_description[targets_key] = tf.io.FixedLenFeature([], tf.string)

  # Create a unique, deterministic task name.
  task_id = hashlib.md5(
      ':'.join(list(paths) +
               [inputs_key, targets_key or '']).encode()).hexdigest()[:10]

  task = seqio.TaskRegistry.add(
      name=f'infer_{task_id}',
      source=seqio.TFExampleDataSource({'infer': paths},
                                       feature_description=feature_description,
                                       reader_cls=reserialize_reader),
      preprocessors=[
          functools.partial(
              seqio.preprocessors.rekey,
              key_map={
                  'inputs': inputs_key,
                  'targets': targets_key
              }), seqio.preprocessors.tokenize_and_append_eos
      ],
      output_features=features)

  return task.name


def write_inferences_to_file(
    path: str,
    inferences: Sequence[Any],
    task_ds: tf.data.Dataset,
    mode: str,
    vocabulary: Optional[seqio.Vocabulary] = None) -> None:
  """Write model predictions, along with pretokenized inputs, to JSONL file.

  Args:
    path: File path to write to.
    inferences: Model inferences, output of either score_batch or predict_batch.
    task_ds: Original task dataset. Features from task with suffix
      `_pretokenized` are added to the outputs.
    mode: Prediction mode, either 'predict', 'score' or 'predict_with_aux'.
    vocabulary: Task output vocabulary. Only used in `predict` mode in order to
      decode predicted outputs into string.
  """
  if mode == 'predict' and not vocabulary:
    raise ValueError('`vocabulary` parameter required in `predict` mode')

  def _json_compat(value):
    if isinstance(value, bytes):
      return value.decode('utf-8')
    elif isinstance(value, (jnp.bfloat16, jnp.floating)):
      return float(value)
    elif isinstance(value, jnp.integer):
      return float(value)
    elif isinstance(value, jnp.ndarray):
      return value.tolist()
    else:
      return value

  with gfile.GFile(path, 'w') as f:
    for inp, output in zip(task_ds, inferences):
      json_dict = {}
      pretokenized = {
          k: v for k, v in inp.items() if k.endswith('_pretokenized')
      }
      if pretokenized:
        json_dict['input'] = {
            k: _json_compat(v.numpy()) for k, v in pretokenized.items()
        }
      if mode == 'predict':
        json_dict['prediction'] = _json_compat(
            vocabulary.decode_tf(tf.constant(output)).numpy())  # pytype: disable=attribute-error
      elif mode == 'score':
        json_dict['score'] = _json_compat(output)
      elif mode == 'predict_with_aux':
        pred_text, pred_aux = output
        json_dict['prediction'] = _json_compat(
            vocabulary.decode_tf(tf.constant(pred_text)).numpy())  # pytype: disable=attribute-error
        json_dict['aux'] = jax.tree_map(_json_compat, pred_aux)
      else:
        raise ValueError(f'Invalid mode: {mode}')
      json_str = json.dumps(json_dict, cls=seqio.TensorAndNumpyEncoder)
      f.write(json_str + '\n')


WriteFn = Callable[
    [str, Sequence[Any], tf.data.Dataset, str, Optional[seqio.Vocabulary]],
    None]


def infer(*,
          mode: str,
          model: models.BaseTransformerModel,
          dataset_cfg: utils.DatasetConfig,
          restore_checkpoint_cfg: utils.RestoreCheckpointConfig,
          partitioner: partitioning.BasePartitioner,
          output_dir: str,
          checkpoint_period: int,
          shard_id: int = 0,
          num_shards: int = 1,
          run_xprof: bool = True,
          merge_epoch_results: bool = True,
          write_fn: WriteFn = write_inferences_to_file):
  """Infer function.

  Args:
    mode: Either 'predict' to decode targets, 'score' to compute the log
      likelihood of given targets, or 'predict_with_aux' for both.
    model: The model object to use for inference.
    dataset_cfg: Specification for the dataset to infer based on.
    restore_checkpoint_cfg: Specification for the model parameter checkpoint to
      load.
    partitioner: Partitioner for model parameters and data across devices.
    output_dir: Path to directory to write temporary files and final results.
    checkpoint_period: The intermediate results and dataset iterator will be
      checkpointed on each multiple of this number of batches to enable
      continuation after a failure.
    shard_id: Index of dataset shard for this instance to use if splitting the
      work across multiple jobs.
    num_shards: Total number of dataset shards to split dataset across.
    run_xprof: Whether to take an xprof snapshot during run.
    merge_epoch_results: Whether to merge results of all epochs into a single
      json file.
    write_fn: Callable function used to serialized and write inferences out to
      files.
  """
  if mode not in ('predict', 'score', 'predict_with_aux'):
    raise ValueError(
        "`mode` must be one of 'predict', 'score' or 'predict_with_aux'. "
        f"Got '{mode}'")

  # Remove double-slashes in directory path to avoid inconsistencies.
  output_dir = re.sub(r'(?<!gs:)([\/]{2,})', '/', output_dir)
  ds_vocabs = utils.get_vocabulary(dataset_cfg)
  if (ds_vocabs[0] != model.input_vocabulary or
      ds_vocabs[1] != model.output_vocabulary):
    raise ValueError(
        'Model and Task vocabularies do not match.\n'
        f'Task Input: {ds_vocabs[0]}, Model Input: {model.input_vocabulary}\n'
        f'Task Output: {ds_vocabs[1]}, Model Output: {model.output_vocabulary}')

  batch_size = dataset_cfg.batch_size

  # Set up dataset.
  if dataset_cfg.module:
    utils.import_module(dataset_cfg.module)
  host_shard_info = seqio.ShardInfo(index=shard_id, num_shards=num_shards)
  task_or_mixture = seqio.get_mixture_or_task(dataset_cfg.mixture_or_task_name)

  feature_converter = model.FEATURE_CONVERTER_CLS(pack=False)  # pytype:disable=not-instantiable

  def _get_dataset(dataset_provider):
    # TODO(adarob): assert pack is false, shuffle is false, seed?
    return dataset_provider.get_dataset(
        sequence_length=dataset_cfg.task_feature_lengths,
        split=dataset_cfg.split,
        shuffle=False,
        num_epochs=1,
        shard_info=host_shard_info,
        use_cached=dataset_cfg.use_cached,
        seed=dataset_cfg.seed)

  # Each "epoch" should be how often we checkpoint the input dataset and flush
  # the inferences to disk.
  logging.info('Inferring with checkpoints every %d batches of %d examples.',
               checkpoint_period, batch_size)

  logging.info('Initializing model, optimizer, and step functions.')
  input_shapes = {
      k: (batch_size,) + spec.shape for k, spec in feature_converter(
          _get_dataset(task_or_mixture),
          dataset_cfg.task_feature_lengths).element_spec.items()
  }
  # Initialize optimizer from the existing checkpoint.
  # TODO(adarob): Support inference over multiple checkpoints.
  train_step_initializer = utils.TrainStateInitializer(
      optimizer_def=model.optimizer_def,
      init_fn=model.get_initial_variables,
      input_shapes=input_shapes,
      partitioner=partitioner)
  train_state = train_step_initializer.from_checkpoint([restore_checkpoint_cfg])
  if mode == 'predict':
    infer_step = model.predict_batch
  elif mode == 'predict_with_aux':
    infer_step = model.predict_batch_with_aux
  else:  # mode == 'score'
    infer_step = model.score_batch

  infer_fn = functools.partial(
      utils.get_infer_fn(
          infer_step=infer_step,
          batch_size=batch_size,
          train_state_axes=train_step_initializer.train_state_axes,
          partitioner=partitioner),
      train_state=train_state)

  def infer_task(task: seqio.Task):
    tmp_dir = os.path.join(output_dir,
                           f'tmp-{task.name}-{shard_id:05}-of-{num_shards:05}')
    if jax.process_index() == 0:
      gfile.makedirs(tmp_dir)

    # Use `max_workers=1` to ensure writes occur sequentially.
    write_thread_pool = FailFastThreadPoolExecutor(max_workers=1)

    logging.info("Loading dataset for task '%s'.", task.name)
    ds = _get_dataset(task)

    model_ds = feature_converter(
        ds, task_feature_lengths=dataset_cfg.task_feature_lengths)

    # Zip task and model features.
    # (task, model)
    infer_ds = tf.data.Dataset.zip((ds, model_ds))

    # Create batches the size of each epoch and index them.
    # (i, [(task, model)] * epoch_size)
    infer_ds = infer_ds.padded_batch(
        checkpoint_period * batch_size, drop_remainder=False).enumerate()
    infer_ds_iter: Iterator[Tuple[int, Any]] = iter(infer_ds.prefetch(AUTOTUNE))
    # Create checkpoint manager and restore state, if applicable.
    ckpt_path = os.path.join(tmp_dir, 'input.ckpt')

    input_ckpt = tf.train.Checkpoint(ds=infer_ds_iter)
    if gfile.glob(ckpt_path + '*'):
      logging.info('Restoring input iterator from %s', ckpt_path)
      input_ckpt.read(ckpt_path).assert_consumed()

    output_fname = f'{task.name}-{mode}.jsonl-{shard_id:05}-of-{num_shards:05}'
    logging.info("Starting inference loop for shard %d of %d of task '%s'.",
                 shard_id, num_shards, task.name)

    def _write_epoch_and_canonicalize_ckpt(epoch: int, epoch_path: str,
                                           inferences: Sequence[Any],
                                           task_ds: tf.data.Dataset,
                                           epoch_ckpt_path: str):
      write_tick = time.time()
      logging.info('Writing epoch %d results to %s', epoch, epoch_path)
      write_fn(epoch_path, inferences, task_ds, mode,
               task.output_features['targets'].vocabulary)
      write_time = time.time() - write_tick
      logging.info('Writing completed in %02f seconds (%02f examples/sec).',
                   write_time,
                   len(inferences) / write_time)
      update_measurement_series('writing_total_sec', epoch, write_time)
      update_measurement_series('writing_examples_per_sec', epoch,
                                len(inferences) / write_time)

      # Canonicalize checkpoint.
      for fname in gfile.glob(epoch_ckpt_path + '*'):
        gfile.rename(
            fname, fname.replace(epoch_ckpt_path, ckpt_path), overwrite=True)

    # Main Loop over "epochs".
    for epoch, epoch_batch in infer_ds_iter:
      logging.info('Starting epoch %d', epoch)

      epoch_tick = time.time()

      # Take an Xprof trace after the first loop has compiled everything.
      if epoch == 1:
        multihost_utils.sync_devices(f'{task.name}:start_xprof')
        utils.start_xprof(seconds=5, maybe_run=run_xprof, description='infer')

      # Load the dataset for the next epoch. We can't use `infer_ds_iter`
      # directly since `infer_fn` needs to know the exact size of each epoch,
      # which may be smaller for the final one.
      epoch_ds = tf.data.Dataset.from_tensor_slices(epoch_batch)
      epoch_ds.cache().prefetch(AUTOTUNE)

      # Unzip epoch dataset in to pretokenized and model datasets.
      task_ds = epoch_ds.map(lambda p, m: p, num_parallel_calls=AUTOTUNE)
      model_ds = epoch_ds.map(lambda p, m: m, num_parallel_calls=AUTOTUNE)

      logging.info('Running inference on %d batches.', checkpoint_period)
      # Sort by and strip index.
      inferences = [
          x[1]
          for x in sorted(infer_fn(model_ds.enumerate()), key=lambda x: x[0])
      ]

      if jax.process_index() == 0:
        epoch_time = time.time() - epoch_tick
        logging.info('Epoch completed in %02f seconds (%02f examples/sec).',
                     epoch_time,
                     len(inferences) / epoch_time)
        update_measurement_series('inference_total_sec', epoch, epoch_time)
        update_measurement_series('inference_examples_per_sec', epoch,
                                  len(inferences) / epoch_time)

        epoch_path = os.path.join(tmp_dir, f'{output_fname}-epoch{epoch:05}')

        # Store iterator checkpoint in temporary location before writing the
        # model output asynchronously. After outputs are written, the checkpoint
        # will be moved to the canonical location to be used if restart occurs.
        ckpt_tick = time.time()
        epoch_ckpt_path = input_ckpt.write(
            os.path.join(tmp_dir, f'{epoch}.ckpt'))
        logging.info(
            'Checkpoint written to temporary location in %02f seconds.',
            time.time() - ckpt_tick)
        # These will execute sequentially since the ThreadPool size is 1.
        write_thread_pool.submit(
            _write_epoch_and_canonicalize_ckpt,
            epoch=epoch,
            epoch_path=epoch_path,
            inferences=inferences,
            task_ds=task_ds,
            epoch_ckpt_path=epoch_ckpt_path)

      # Wait for checkpoint to be written before continuing.
      multihost_utils.sync_devices(f'{task.name}:checkpoint_epoch{epoch:05}')

    logging.info("Finished inference for task '%s'.", task.name)

    logging.info('Waiting for epoch writes to complete.')
    write_thread_pool.shutdown(wait=True)

    if jax.process_index() == 0 and merge_epoch_results:
      logging.info('Merging epoch results.')
      # Merge epochs into single file.
      epoch_paths = sorted(
          gfile.glob(os.path.join(tmp_dir, f'{output_fname}-epoch?????')))
      assert int(epoch_paths[-1][-5:]) + 1 == len(epoch_paths), (
          f'Expecting {int(epoch_paths[-1][-5:])} epoch paths, found '
          f'{len(epoch_paths)}')
      output_path = os.path.join(output_dir, output_fname)
      with gfile.GFile(output_path, 'wb') as merged:
        for epoch_path in epoch_paths:
          with gfile.GFile(epoch_path, 'rb') as ef:
            shutil.copyfileobj(ef, merged)
      logging.info('Results written to %s.', output_path)
      logging.info('Deleting temporary files.')
      gfile.rmtree(tmp_dir)

    # Wait for host 0 to finish writing before exiting.
    multihost_utils.sync_devices(f'{task.name}:complete')

  for task in seqio.get_subtasks(task_or_mixture):
    logging.info("Starting inference for task '%s'", task.name)
    infer_task(task)
  logging.info('DONE')


def update_measurement_series(series_name: str, step: int, value: float):
  """Not implemented externally."""
  del series_name, step, value


if __name__ == '__main__':
  # pylint:disable=g-import-not-at-top
  from absl import app
  from absl import flags
  import gin
  from t5x import gin_utils
  # pylint:enable=g-import-not-at-top

  FLAGS = flags.FLAGS

  jax.config.parse_flags_with_absl()

  flags.DEFINE_integer(
      'shard_id',
      default=None,
      help='Index to use for splitting the Task across multiple inference '
      'runs. NB: If set, this overrides --gin.infer.shard_id')

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
      default=['third_party/py/t5x/configs'],
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


    # Create gin-configurable version of `infer`.
    infer_using_gin = gin.configurable(infer)

    gin_utils.parse_gin_flags(
        # User-provided gin paths take precedence if relative paths conflict.
        FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
        FLAGS.gin_file,
        FLAGS.gin_bindings)

    # See http://yaqs/7882016229479677952 for further gin-config discussion.
    def _get_gin_parameter(key: str) -> Any:
      value = gin.query_parameter(key)
      if isinstance(value, gin.config.ConfigurableReference):
        if value.evaluate:
          return value.scoped_configurable_fn()
        return value.scoped_configurable_fn
      return value

    shard_id = (
        FLAGS.shard_id
        if FLAGS.shard_id is not None else _get_gin_parameter('infer.shard_id'))
    if shard_id == 0:
      gin_utils.summarize_gin_config(
          model_dir=_get_gin_parameter('infer.output_dir'),
          summary_writer=None,
          step=0)
    if FLAGS.shard_id is not None:
      # We fall back to this flag since XM does not support sweeps over flags
      # with '.' in them (it treats them like nested dictionaries).
      # TODO(adarob): Figure out a workaround so we can deprecate this flag.
      infer_using_gin(shard_id=FLAGS.shard_id)
    else:
      infer_using_gin()



  gin_utils.run(main)
