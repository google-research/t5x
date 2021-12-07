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
from functools import partial
import concurrent.futures
import functools
import hashlib
import json
import os
import re
import shutil
import time
from typing import Any, Callable, Iterator, List, Mapping, Optional, Sequence, Tuple
import task

from absl import logging
import jax
import jax.numpy as jnp
import seqio
from t5x import models
from t5x import multihost_utils
from t5x import partitioning
from t5x import utils
from t5x.infer import create_task_from_tfexample_file
import tensorflow as tf
from tensorflow.io import gfile

from lm_eval.base import LM
import numpy as np
from lm_eval import evaluator, tasks
from models import cross_entropy_with_logits
from flax.training import common_utils


# Automatically search for gin files relative to the T5X package.
_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]

AUTOTUNE = tf.data.experimental.AUTOTUNE

def create_task_from_tuples(data, vocab):
    tfrecord_writer = tf.io.TFRecordWriter("data.tfrecord")
    def _bytes_feature(value):
      """Returns a bytes_list from a string / byte."""
      if isinstance(value, type(tf.constant(0))):
          value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    for p in data:
      input, target = p
      input, target = input.encode('utf-8'), target.encode('utf-8')
      example = tf.train.Example(features=tf.train.Features(feature={
        'input': _bytes_feature(input),
        'target': _bytes_feature(target),
      }))
      tfrecord_writer.write(example.SerializeToString())

    tfrecord_writer.close()
    
    features = {'inputs': seqio.Feature(vocabulary=vocab, add_eos = False), 'targets': seqio.Feature(vocabulary=vocab, add_eos = False)}
    task_name = create_task_from_tfexample_file(['data.tfrecord'], 'tfrecord', 'input', 'target', features)
    return task_name

def infer(*,
          mode: str,
          model: models.BaseTransformerModel,
          dataset_cfg: utils.DatasetConfig,
          restore_checkpoint_cfg: utils.RestoreCheckpointConfig,
          partitioner: partitioning.BasePartitioner,
          output_dir: str,
          checkpoint_period: int,
          task_name : str,
          shard_id: int = 0,
          num_shards: int = 1,
          run_xprof: bool = True,
          merge_epoch_results: bool = True):
  """Funciton to run the inference and return the results as is. Slightly simpler version than the one in infer.py

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
  if mode not in ('predict', 'score', 'predict_with_aux', 'score_with_correct'):
    raise ValueError(
        "`mode` must be one of 'predict', 'score' or 'predict_with_aux'. "
        f"Got '{mode}'")

  # Remove double-slashes in directory path to avoid inconsistencies.
  output_dir = re.sub(r'(?<!gs:)([\/]{2,})', '/', output_dir)
  dataset_cfg.mixture_or_task_name = task_name
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
  task_or_mixture = seqio.get_mixture_or_task(task_name)

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
  
  def compute_logprobs_and_correct(params, batch):
    logits = model._compute_logits(params, batch)  # type: jnp.ndarray
    target_tokens = batch['decoder_target_tokens']
    weights = batch['decoder_loss_weights']
    token_scores = -cross_entropy_with_logits(
      logits,
      common_utils.onehot(
          target_tokens, logits.shape[-1], on_value=1, off_value=0),
      z_loss=0.0)[0] * weights
    correct = ((logits.argmax(-1) == target_tokens)*weights).sum(-1) == weights.sum(-1)
    sequence_logprob = token_scores.sum(-1)
    return sequence_logprob, correct

  infer_fn = functools.partial(
    utils.get_infer_fn(
      infer_step=compute_logprobs_and_correct,
      batch_size=batch_size,
      train_state_axes=train_step_initializer.train_state_axes,
      partitioner=partitioner),
    train_state=train_state)

  def infer_task(task: seqio.Task):
    tmp_dir = os.path.join(output_dir,
                           f'tmp-{task.name}-{shard_id:05}-of-{num_shards:05}')
    if jax.process_index() == 0:
      gfile.makedirs(tmp_dir)

    
    logging.info("Loading dataset for task '%s'.", task.name)
    ds = _get_dataset(task)

    model_ds = feature_converter(
      ds, task_feature_lengths=dataset_cfg.task_feature_lengths)

    infer_ds = tf.data.Dataset.zip((ds, model_ds))

    infer_ds = infer_ds.padded_batch(
      checkpoint_period * batch_size, drop_remainder=False).enumerate()
    infer_ds_iter: Iterator[Tuple[int, Any]] = iter(infer_ds.prefetch(AUTOTUNE))
    
    logging.info("Starting inference loop for shard %d of %d of task '%s'.",
                 shard_id, num_shards, task.name)

    all_infernces = []
    # Main Loop over "epochs".
    for epoch, epoch_batch in infer_ds_iter:
      logging.info('Starting epoch %d', epoch)

      # Load the dataset for the next epoch. We can't use `infer_ds_iter`
      # directly since `infer_fn` needs to know the exact size of each epoch,
      # which may be smaller for the final one.
      epoch_ds = tf.data.Dataset.from_tensor_slices(epoch_batch)
      epoch_ds.cache().prefetch(AUTOTUNE)

      # Unzip epoch dataset in to pretokenized and model datasets.
      model_ds = epoch_ds.map(lambda p, m: m, num_parallel_calls=AUTOTUNE)
      
      logging.info('Running inference on %d batches.', checkpoint_period)
      # Sort by and strip index.
      inferences = [
        x[1] for x in sorted(infer_fn(model_ds.enumerate()), key=lambda x: x[0])
      ]
      all_infernces += inferences
    return all_infernces 
  
  for task in seqio.get_subtasks(task_or_mixture):
    logging.info("Starting inference for task '%s'", task.name)
    return infer_task(task)
  logging.info('DONE')

def eval_task(output_path, create_task_func, infer_func):
  class EvalHarnessAdaptor(LM):
    def greedy_until(self, requests):
        raise Exception("unimplemented")

    def loglikelihood_rolling(self, requests):
        raise Exception("unimplemented")

    def __init__(self):
        super().__init__()

    def loglikelihood(self, requests):
        task_name = create_task_func(requests)
        return np.array(infer_func(task_name = task_name, mode = 'score_with_correct'))
  
  adaptor = EvalHarnessAdaptor()
  active_tasks = [
    "lambada",
    "hellaswag",
    "winogrande",
    "piqa",
    "mathqa",
    "pubmedqa"
    ]
  
  results = evaluator.evaluate(adaptor, tasks.get_task_dict(active_tasks), False, 0, None)
  dumped = json.dumps(results, indent=2)
  print(dumped)
  with open(output_path, 'w') as outfile:
    json.dump(results, outfile)

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
      'gin_file_',
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

  flags.DEFINE_string(
      'results_path', None, "Path to save the json with the results.")
  
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
    
    create_task_from_tuples_gin = gin.configurable(create_task_from_tuples)
    gin_utils.parse_gin_flags(
        # User-provided gin paths take precedence if relative paths conflict.
        FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
        FLAGS.gin_file_,
        FLAGS.gin_bindings)

    print(FLAGS.results_path)
    
    eval_task(FLAGS.results_path, create_task_from_tuples_gin, infer_using_gin)
    
  gin_utils.run(main)
