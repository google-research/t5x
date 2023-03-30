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

r"""Precompile and generates HLO from TPU metadata backend.

TPU Metadata backend is a TPU backend without real TPU devices while supporting
any TPU topologies, to allow work that doesn't require real TPUs to run as if
it is, e.g., compiling/lowering a HLO graph with the backend.

Ideally, the precompile defaults to cpu backend for default device array
placement since metadata backend does not have memory allocation.

The pjit function is pinned to use available TPU Metadata backend, for getting
a proper lowering under TPU mesh.

"""

import os
from typing import Callable, Optional

import clu.data

import jax
from jax import random
import numpy as np
import t5.data.mixtures  # pylint:disable=unused-import
from t5x import models
from t5x import partitioning
from t5x import trainer as trainer_lib
from t5x import utils
import tensorflow as tf



def precompile(
    *,
    model: models.BaseTransformerModel,
    train_dataset_cfg: utils.DatasetConfig,
    partitioner: partitioning.BasePartitioner,
    model_dir: str,
    random_seed: Optional[int],
    get_dataset_fn: utils.GetDatasetCallable = utils.get_dataset,
    verify_matching_vocabs_fn: Optional[
        Callable[[utils.DatasetConfig, models.BaseTransformerModel],
                 None]] = utils.verify_matching_vocabs,
):
  """Compiles and dump the HLO to model dir, with HLO text dumps."""
  rng = random.PRNGKey(random_seed or 42)
  _, trainer_rng = random.split(rng, 2)

  # TODO(hthu): Find a better way of getting dataset shapes instead of actually
  # reading database and iterate on it.
  data_layout = partitioner.get_data_layout(train_dataset_cfg.batch_size)
  ds_shard_id = data_layout.shard_id
  num_ds_shards = data_layout.num_shards

  if verify_matching_vocabs_fn is not None:
    verify_matching_vocabs_fn(train_dataset_cfg, model)

  train_iter = get_dataset_fn(train_dataset_cfg, ds_shard_id, num_ds_shards,
                              model.FEATURE_CONVERTER_CLS)
  if isinstance(train_iter, tf.data.Dataset):
    train_iter = clu.data.TfDatasetIterator(train_iter, checkpoint=True)
  elif not isinstance(train_iter, clu.data.dataset_iterator.DatasetIterator):
    raise ValueError(
        f'get_dataset_fn returned unsupported type {type(train_iter)}.')

  # Need to use full batch size.
  input_shapes = jax.tree_map(lambda x: (data_layout.batch_size, *x.shape[1:]),
                              train_iter.element_spec)
  input_types = jax.tree_map(lambda x: x.dtype, train_iter.element_spec)
  dummy_batch = jax.tree_map(lambda x: np.ones(x.shape, x.dtype),
                             train_iter.element_spec)

  # Compiling does not care about loading real weights.
  train_state_initializer = utils.TrainStateInitializer(
      optimizer_def=model.optimizer_def,
      init_fn=model.get_initial_variables,
      input_shapes=input_shapes,
      input_types=input_types,
      partitioner=partitioner)
  train_state_shape = train_state_initializer.global_train_state_shape
  train_state_axes = train_state_initializer.train_state_axes

  def train_step(train_state, batch):
    return trainer_lib.train_with_lr(  # pytype: disable=wrong-arg-types  # jax-ndarray
        train_state,
        batch,
        learning_rate=1e-3,
        dropout_rng=trainer_rng,
        model=model,
        num_microbatches=None,
        weight_metrics_computer=None)

  partitioned_step = partitioner.partition(
      train_step,
      in_axis_resources=(train_state_axes, partitioning.PartitionSpec('data',)),
      out_axis_resources=(train_state_axes, None),
      donate_argnums=(0,))

  # PartitionedTrainCallable has lower() defined but isn't exposed in pytype.
  # TODO(hthu): Explicitly expose the lower() interface.
  # pytype: disable=attribute-error
  lowered = partitioned_step.lower(train_state_shape, dummy_batch)
  # pytype: enable=attribute-error


  # TODO(hthu): Make this a proper library without writing files by default.
  tf.io.gfile.makedirs(model_dir)
  with tf.io.gfile.GFile(
      os.path.join(model_dir, 'lowered_hlo_pre_optimization'), 'w') as f:
    f.write(lowered.compiler_ir(dialect='hlo').as_serialized_hlo_module_proto())
  compiled = lowered.compile()
  output_path = os.path.join(model_dir, 'lowered_hlo_post_optimization')
  with tf.io.gfile.GFile(output_path, 'w') as f:
    f.write(compiled.compiler_ir()[0].as_serialized_hlo_module_proto())
  with tf.io.gfile.GFile(os.path.join(model_dir, 'assignment'), 'wb') as f:
    np.save(f, partitioner.mesh.device_ids)
