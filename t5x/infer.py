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
from typing import Any, Callable, Iterator, List, Mapping, Optional, Sequence, Tuple, Type

# TODO(adarob): Re-enable once users are notified and tests are updated.
# Must be set before flax imports.
# pylint:disable=g-import-not-at-top
os.environ['FLAX_LAZY_RNG'] = 'no'
from absl import logging
from clu import metric_writers
import jax
import jax.numpy as jnp
import numpy as np
import seqio
from t5x import gin_utils
from t5x import models
from t5x import partitioning
from t5x import utils
import tensorflow as tf
from tensorflow.io import gfile
from typing_extensions import Protocol

# Automatically search for gin files relative to the T5X package.
_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]

AUTOTUNE = tf.data.experimental.AUTOTUNE


class SummarizeConfigFn(Protocol):

  def __call__(self, model_dir: str,
               summary_writer: Optional[metric_writers.SummaryWriter],
               step: int) -> None:
    ...


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


def create_task_from_tfexample_file(paths: Sequence[str],
                                    file_type: str,
                                    inputs_key: str,
                                    targets_key: Optional[str],
                                    features: Mapping[str, seqio.Feature],
                                    task_id: Optional[str] = None) -> str:
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
    task_id: Task name identifier. By default, it is set to a unique and
      deterministic hash id. Overrideable via this argument.

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

  feature_description = {inputs_key: tf.io.FixedLenFeature([], tf.string)}
  if targets_key:
    feature_description[targets_key] = tf.io.FixedLenFeature([], tf.string)

  # Create a unique, deterministic task name.
  if task_id is None:
    task_id = hashlib.md5(
        ':'.join(list(paths) +
                 [inputs_key, targets_key or '']).encode()).hexdigest()[:10]

  task = seqio.TaskRegistry.add(
      name=f'infer_{task_id}',
      source=seqio.TFExampleDataSource({'infer': paths},
                                       feature_description=feature_description,
                                       reader_cls=reader),
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


def merge_chunks_to_file(
    output_dir: str,
    output_fname: str,
    tmp_dir: str,
    step: Optional[int],
) -> None:
  """Merge the predictions from different chunks into a unified file."""
  logging.info('Merging chunk results.')
  # Merge chunks into single file.
  chunk_paths = sorted(
      gfile.glob(os.path.join(tmp_dir, f'{output_fname}-chunk?????')))

  if not chunk_paths:
    raise FileNotFoundError(
        'No chunk results found! One possible explanation is that your '
        'input did not contain any examples')

  assert int(chunk_paths[-1][-5:]) + 1 == len(chunk_paths), (
      f'Expecting {int(chunk_paths[-1][-5:])} chunk paths, found '
      f'{len(chunk_paths)}')
  output_path = os.path.join(output_dir, output_fname)
  del step
  with gfile.GFile(output_path, 'wb') as merged:
    for chunk_path in chunk_paths:
      with gfile.GFile(chunk_path, 'rb') as ef:
        shutil.copyfileobj(ef, merged)
  logging.info('Results written to %s.', output_path)


Inferences = Tuple[Sequence[Any], Mapping[str, Any]]
_Inferences = Inferences  # Backwards-compatible alias; used by Colabs


def write_inferences_to_file(
    path: str,
    inferences: Inferences,
    task_ds: tf.data.Dataset,
    mode: str,
    vocabulary: Optional[seqio.Vocabulary] = None,
    json_encoder_cls: Type[json.JSONEncoder] = seqio.TensorAndNumpyEncoder,
    include_all_inputs: bool = False,
    input_fields_to_include: Optional[Sequence[str]] = None,
    output_ids: bool = False,
) -> None:
  """Write model predictions, along with pretokenized inputs, to JSONL file.

  Args:
    path: File path to write to.
    inferences: A tuple containing (predictions, aux_values). If mode is
      'predict' then the `predictions` will be token IDs. If it's
      'score' then it'll be a collection of scores. `aux_values` will be an
      empty dictionary unless mode is 'predict_with_aux', in which case it'll
      contain the model's auxiliary outputs.
    task_ds: Original task dataset. Features from task with suffix
      `_pretokenized` are added to the outputs.
    mode: Prediction mode, either 'predict', 'score' or 'predict_with_aux'.
    vocabulary: Task output vocabulary. Only used in `predict` mode in order to
      decode predicted outputs into string.
    json_encoder_cls: a JSON encoder class used to customize JSON serialization
      via json.dumps.
    include_all_inputs: if True, will include all model inputs in the output
      JSONL file (including raw tokens) in addition to the pretokenized inputs.
    input_fields_to_include: List of input fields to include in the output JSONL
      file. This list should be None if `include_all_inputs` is set to True.
    output_ids: if True, will output the token ID sequence for the output, in
      addition to the decoded text.
  """
  all_predictions, all_aux_values = inferences

  if mode in ('predict', 'predict_with_aux') and vocabulary is None:
    raise ValueError('The `vocabulary` parameter is required in `predict` and '
                     '`predict_with_aux` modes')

  def _json_compat(value):
    if isinstance(value, bytes):
      return value.decode('utf-8')
    elif isinstance(value, (jnp.bfloat16, jnp.floating)):
      return float(value)
    elif isinstance(value, jnp.integer):
      return float(value)
    elif isinstance(value, (jnp.ndarray, np.ndarray)):
      # Flatten array features.
      return value.tolist()
    else:
      return value

  if include_all_inputs and input_fields_to_include is not None:
    raise ValueError(
        'include_all_inputs and input_fields_to_include should not be set'
        ' simultaneously.')
  with gfile.GFile(path, 'w') as f:
    for i, inp in task_ds.enumerate().as_numpy_iterator():
      predictions = all_predictions[i]
      aux_values = jax.tree_map(
          f=lambda v, i=i: v[i],
          tree=all_aux_values,
          is_leaf=lambda v: isinstance(v, (np.ndarray, list)),
      )

      if include_all_inputs:
        inputs = inp
      elif input_fields_to_include is not None:
        inputs = {
            k: v for k, v in inp.items() if k in input_fields_to_include or
            (k.endswith('_pretokenized') and
             k[:-len('_pretokenized')] in input_fields_to_include)
        }
      else:
        inputs = {k: v for k, v in inp.items() if k.endswith('_pretokenized')}

      json_dict = {}
      json_dict['inputs'] = {k: _json_compat(v) for k, v in inputs.items()}

      if mode == 'predict':
        assert vocabulary is not None
        json_dict['prediction'] = _json_compat(
            vocabulary.decode_tf(tf.constant(predictions)).numpy())
        if output_ids:
          pred = _json_compat(tf.constant(predictions).numpy())
          # Truncate padding tokens.
          assert isinstance(pred, list)
          pred = pred[:pred.index(0)] if 0 in pred else pred
          json_dict['prediction_tokens'] = pred
      elif mode == 'score':
        json_dict['score'] = _json_compat(predictions)
        if aux_values:
          json_dict['aux'] = jax.tree_map(_json_compat, aux_values)
      elif mode == 'predict_with_aux':
        assert vocabulary is not None
        json_dict['prediction'] = _json_compat(
            vocabulary.decode_tf(tf.constant(predictions)).numpy())
        if output_ids:
          pred = _json_compat(tf.constant(predictions).numpy())
          # Truncate padding tokens.
          pred = pred[:pred.index(0)] if 0 in pred else pred
          json_dict['prediction_tokens'] = pred
        json_dict['aux'] = jax.tree_map(_json_compat, aux_values)
      else:
        raise ValueError(f'Invalid mode: {mode}')
      json_str = json.dumps(json_dict, cls=json_encoder_cls)
      f.write(json_str + '\n')


WriteFn = Callable[
    [
        str,
        Inferences,
        tf.data.Dataset,
        str,
        Optional[seqio.Vocabulary],
    ],
    None,
]

MergeFn = Callable[[str, str, str, Optional[int]], None]


def _extract_tokens_and_aux_values(inference_fn_outputs) -> Inferences:
  """Extracts tokens and aux scores from a cached dataset."""
  all_aux_values = {}
  if isinstance(inference_fn_outputs, tuple):
    indices_and_tokens, all_aux_values = inference_fn_outputs
    indices, tokens = zip(*indices_and_tokens)

    permutation = np.argsort(indices)
    permute = lambda v: [v[permutation[i]] for i in range(len(permutation))]
    tokens = permute(tokens)
    all_aux_values = jax.tree_map(
        f=permute,
        tree=all_aux_values,
        is_leaf=lambda v: isinstance(v, (np.ndarray, list)),
    )

  else:
    indices_and_tokens = inference_fn_outputs
    _, tokens = zip(*sorted(indices_and_tokens, key=lambda x: x[0]))

  return tokens, all_aux_values


def infer(
    *,
    mode: str,
    model: models.BaseTransformerModel,
    dataset_cfg: utils.DatasetConfig,
    restore_checkpoint_cfg: utils.RestoreCheckpointConfig,
    partitioner: partitioning.BasePartitioner,
    output_dir: str,
    checkpoint_period: int,
    shard_id: int = 0,
    num_shards: int = 1,
    merge_chunked_results: bool = True,
    write_fn: WriteFn = write_inferences_to_file,
    checkpoint_ds_iter: bool = True,
    train_state_initializer_cls: Type[
        utils.TrainStateInitializer
    ] = utils.TrainStateInitializer,
    fallback_init_rng: Optional[int] = None,
    merge_fn: MergeFn = merge_chunks_to_file,
    summarize_config_fn: SummarizeConfigFn = gin_utils.summarize_gin_config,
    verify_matching_vocabs_fn: Optional[
        Callable[[utils.DatasetConfig, models.BaseTransformerModel], None]
    ] = utils.verify_matching_vocabs,
    output_vocab_feature_name: str = 'targets',
    file_extension: str = 'jsonl',
    keep_aux_as_numpy: bool = False,
    use_orbax: bool = False,
):
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
    merge_chunked_results: Whether to merge results of all chunks into a single
      json file.
    write_fn: Callable function used to serialized and write inferences out to
      files.
    checkpoint_ds_iter: if True, will checkpoint the dataset iterator every
      `checkpoint_period` to enable faster restore. This must be disabled for
      certain datasets, for example since stateful iterators (e.g. from
      seqio.FunctionTask) cannot be checkpointed.
    train_state_initializer_cls: t5x.utils.TrainStateInitializer class for
      initializing partitioned TrainState from checkpoints or scratch.
    fallback_init_rng: A random seed used for parameter initialization during
      model re-loading when utils.RestoreCheckpointConfig.fallback_to_scratch is
      set to True. If None, parameter initialization is not allowed during model
      loading and having fallback_to_scratch enabled will result in an error.
    merge_fn: Callable function used to merge inferences from multiple files.
    summarize_config_fn: A function that takes in the model directory, an
      optional SummaryWriter, and the step number, and writes a summary of the
      configuration. SummaryWriter will be None in most cases.
    verify_matching_vocabs_fn: Function to validate whether the task vocabulary
      matches the model vocabulary. Should raise an exception on error.
    output_vocab_feature_name: The name of the feature corresponding to the
      output vocabulary.
    file_extension: str. file extension used for file names
    keep_aux_as_numpy: bool. whether to leave aux values as numpy arrays; can be
      used to save space when saving bfloat16s
    use_orbax: if True, uses Orbax for checkpointing. Experimental feature.
  """
  jax.monitoring.record_event('/jax/t5x/infer/beacon')
  logging.info('Process ID: %d', jax.process_index())

  # Only allow `shard_id` 0 to write config summary, since the config summary
  # does NOT depend on `shard_id`.
  if shard_id == 0:
    summarize_config_fn(model_dir=output_dir, summary_writer=None, step=0)

  if mode not in ('predict', 'score', 'predict_with_aux'):
    raise ValueError(
        "`mode` must be one of 'predict', 'score' or 'predict_with_aux'. "
        f"Got '{mode}'")

  # Remove double-slashes in directory path to avoid inconsistencies.
  output_dir = re.sub(r'(?<!gs:)([\/]{2,})', '/', output_dir)
  if verify_matching_vocabs_fn is not None:
    verify_matching_vocabs_fn(dataset_cfg, model)

  batch_size = dataset_cfg.batch_size

  # Set up dataset.
  if dataset_cfg.module:
    utils.import_module(dataset_cfg.module)
  host_shard_info = seqio.ShardInfo(index=shard_id, num_shards=num_shards)
  task_or_mixture = seqio.maybe_get_mixture_or_task(
      dataset_cfg.mixture_or_task_name
  )

  feature_converter = model.FEATURE_CONVERTER_CLS(pack=False)

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

  # Each "chunk" should be how often we checkpoint the input dataset and flush
  # the inferences to disk.
  logging.info('Inferring with checkpoints every %d batches of %d examples.',
               checkpoint_period, batch_size)

  logging.info('Initializing model, optimizer, and step functions.')
  element_spec = feature_converter(
      _get_dataset(task_or_mixture),
      dataset_cfg.task_feature_lengths).element_spec
  input_shapes = {
      k: (batch_size,) + spec.shape for k, spec in element_spec.items()
  }
  input_types = {
      k: jnp.dtype(spec.dtype.as_numpy_dtype)
      for k, spec in element_spec.items()
  }
  # Initialize optimizer from the existing checkpoint.
  # TODO(adarob): Support inference over multiple checkpoints.
  train_state_initializer = train_state_initializer_cls(
      optimizer_def=None,  # Do not load optimizer state.
      init_fn=model.get_initial_variables,
      input_shapes=input_shapes,
      input_types=input_types,
      partitioner=partitioner)
  # Log the variable shapes information and write to a file.
  model_info_log_file = os.path.join(output_dir, 'model-info.txt')
  if shard_id == 0:
    utils.log_model_info(model_info_log_file,
                         train_state_initializer.global_train_state_shape,
                         partitioner)

  # Disable strictness since we are dropping the optimizer state.
  restore_checkpoint_cfg.strict = False
  if fallback_init_rng is not None:
    fallback_init_rng = jax.random.PRNGKey(fallback_init_rng)

  train_state, _ = utils.create_checkpoint_manager_and_restore(
      train_state_initializer,
      partitioner,
      restore_checkpoint_cfg,
      restore_checkpoint_cfg.path,
      fallback_init_rng,
      use_orbax=use_orbax,
  )
  if train_state is None:
    raise ValueError('TrainState was not found or could not be restored.')

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
          train_state_axes=train_state_initializer.train_state_axes,
          partitioner=partitioner,
          keep_aux_as_numpy=keep_aux_as_numpy,
      ),
      train_state=train_state,
  )

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

    # Create batches the size of each chunk and index them.
    # (i, [(task, model)] * chunk_size)
    infer_ds = infer_ds.padded_batch(
        checkpoint_period * batch_size, drop_remainder=False).enumerate()

    infer_ds_iter: Iterator[Tuple[int, Any]] = iter(infer_ds.prefetch(AUTOTUNE))

    if checkpoint_ds_iter:
      # Create checkpoint manager and restore state, if applicable.
      ckpt_path = os.path.join(tmp_dir, 'input.ckpt')

      input_ckpt = tf.train.Checkpoint(ds=infer_ds_iter)
      if gfile.glob(ckpt_path + '*'):
        logging.info('Restoring input iterator from %s', ckpt_path)
        input_ckpt.read(ckpt_path).assert_consumed()

    output_fname = (
        f'{task.name}-{mode}.{file_extension}-{shard_id:05}-of-{num_shards:05}'
    )
    if gfile.exists(os.path.join(output_dir, f'{output_fname}.COMPLETED')):
      logging.info(
          "File %s exists. Skipping inference for shard %d/%d of task '%s'",
          output_fname, shard_id, num_shards, task.name)
      return
    logging.info("Starting inference loop for shard %d of %d of task '%s'.",
                 shard_id, num_shards, task.name)

    def _write_chunk_and_canonicalize_ckpt(
        chunk: int,
        chunk_path: str,
        inferences: Inferences,
        task_ds: tf.data.Dataset,
        chunk_ckpt_path: Optional[str],
    ):
      write_tick = time.time()
      logging.info('Writing chunk %d results to %s', chunk, chunk_path)
      vocabulary = task.output_features[output_vocab_feature_name].vocabulary
      write_fn(chunk_path, inferences, task_ds, mode, vocabulary)
      with gfile.GFile(chunk_path + '.COMPLETED', 'w') as f:
        f.write('')
      write_time = time.time() - write_tick
      num_examples = len(inferences[0])
      logging.info('Writing completed in %02f seconds (%02f examples/sec).',
                   write_time, num_examples / write_time)
      update_measurement_series('writing_total_sec', chunk, write_time)
      update_measurement_series('writing_examples_per_sec', chunk,
                                num_examples / write_time)

      if chunk_ckpt_path:
        # Canonicalize checkpoint.
        for fname in gfile.glob(chunk_ckpt_path + '*'):
          gfile.rename(
              fname, fname.replace(chunk_ckpt_path, ckpt_path), overwrite=True)

    # Main Loop over "chunks".
    for chunk, chunk_batch in infer_ds_iter:
      logging.info('Starting chunk %d', chunk)

      chunk_tick = time.time()


      # Load the dataset for the next chunk. We can't use `infer_ds_iter`
      # directly since `infer_fn` needs to know the exact size of each chunk,
      # which may be smaller for the final one.
      chunk_ds = tf.data.Dataset.from_tensor_slices(chunk_batch)
      chunk_ds.cache().prefetch(AUTOTUNE)

      # Unzip chunk dataset in to pretokenized and model datasets.
      task_ds = chunk_ds.map(lambda p, m: p, num_parallel_calls=AUTOTUNE)
      model_ds = chunk_ds.map(lambda p, m: m, num_parallel_calls=AUTOTUNE)

      # Get a chunk-specific RNG key.
      chunk_rng = jax.random.fold_in(jax.random.PRNGKey(0), chunk)
      chunk_path = os.path.join(tmp_dir, f'{output_fname}-chunk{chunk:05}')
      if gfile.exists(chunk_path + '.COMPLETED') and not checkpoint_ds_iter:
        logging.info('Skipping chunk %s. Chunk file already exists.', chunk)
        continue

      logging.info('Running inference on %d batches.', checkpoint_period)
      infer_result = infer_fn(model_ds.enumerate(), rng=chunk_rng)
      inferences: Tuple[Sequence[Any], Mapping[str, Any]] = (
          _extract_tokens_and_aux_values(infer_result))
      num_examples = len(inferences[0])

      if jax.process_index() == 0:
        chunk_time = time.time() - chunk_tick
        logging.info('chunk completed in %02f seconds (%02f examples/sec).',
                     chunk_time, num_examples / chunk_time)
        update_measurement_series('inference_total_sec', chunk, chunk_time)
        update_measurement_series('inference_examples_per_sec', chunk,
                                  num_examples / chunk_time)

        chunk_ckpt_path = None
        if checkpoint_ds_iter:
          # Store iterator checkpoint in temporary location before writing the
          # model output asynchronously. After outputs are written, the
          # checkpoint will be moved to the canonical location to be used if
          # restart occurs.
          ckpt_tick = time.time()
          chunk_ckpt_path = input_ckpt.write(
              os.path.join(tmp_dir, f'{chunk}.ckpt'))
          logging.info(
              'Checkpoint written to temporary location in %02f seconds.',
              time.time() - ckpt_tick)

        # These will execute sequentially since the ThreadPool size is 1.
        write_thread_pool.submit(
            _write_chunk_and_canonicalize_ckpt,
            chunk=chunk,
            chunk_path=chunk_path,
            inferences=inferences,
            task_ds=task_ds,
            chunk_ckpt_path=chunk_ckpt_path)

      # Wait for checkpoint to be written before continuing.
      utils.sync_global_devices(f'{task.name}:checkpoint_chunk{chunk:05}')

    logging.info("Finished inference for task '%s'.", task.name)

    logging.info('Waiting for chunk writes to complete.')
    write_thread_pool.shutdown(wait=True)

    if jax.process_index() == 0 and merge_chunked_results:
      step = None if train_state is None else int(train_state.step)
      merge_fn(output_dir, output_fname, tmp_dir, step)
      logging.info('Deleting temporary files.')
      gfile.rmtree(tmp_dir)

    if jax.process_index() == 0:
      with gfile.GFile(
          os.path.join(output_dir, f'{output_fname}.COMPLETED'), 'w') as f:
        f.write('')

    # Wait for host 0 to finish writing before exiting.
    utils.sync_global_devices(f'{task.name}:complete')

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
  import fiddle as fdl
  import gin
  from t5x import config_utils
  # pylint:enable=g-import-not-at-top

  FLAGS = flags.FLAGS

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
      config = config_utils.config_with_fiddle(infer)
      shard_id = FLAGS.shard_id
      if shard_id is not None:
        config.shard_id = shard_id
      infer_with_fiddle = fdl.build(config)
      if shard_id == 0:
        config_utils.direct_summarize_fiddle_config(
            model_dir=infer_with_fiddle.output_dir,
            summary_writer=None,
            step=0,
            get_current_fiddle_config=lambda: infer_with_fiddle,
        )
      infer_with_fiddle()
    else:
      # Create gin-configurable version of `infer`.
      infer_using_gin = gin.configurable(infer)

      gin_utils.parse_gin_flags(
          # User-provided gin paths take precedence if relative paths conflict.
          FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
          FLAGS.gin_file,
          FLAGS.gin_bindings,
      )

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
          if FLAGS.shard_id is not None
          else _get_gin_parameter('infer.shard_id')
      )
      if shard_id == 0:
        gin_utils.summarize_gin_config(
            model_dir=_get_gin_parameter('infer.output_dir'),
            summary_writer=None,
            step=0,
        )
      if FLAGS.shard_id is not None:
        # We fall back to this flag since XM does not support sweeps over flags
        # with '.' in them (it treats them like nested dictionaries).
        # TODO(adarob): Figure out a workaround so we can deprecate this flag.
        infer_using_gin(shard_id=FLAGS.shard_id)
      else:
        infer_using_gin()


  config_utils.run(main)
