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

"""The model inference libraries for t5x evaluations."""

import abc
import collections
import os
import time
from typing import Any, List, Mapping, Optional, Type, Union

import jax
import jax.numpy as jnp
import seqio
from t5x import models
from t5x import partitioning
from t5x import utils
import tensorflow as tf



class BaseModelInference(abc.ABC):
  """The abstract base class for the model inference framework."""

  def __init__(self,
               dataset_cfg: utils.DatasetConfig,
               output_dir: Optional[str] = None):
    """The initialization that doesn't require shapes."""
    self._dataset_cfg = dataset_cfg
    self._output_dir = output_dir

  @abc.abstractmethod
  def initialize_with_element_spec(self,
                                   element_spec: Mapping[str, tf.TypeSpec],
                                   batch_size: int,
                                   log_info: Optional[bool] = True):
    """The initializations that requires shapes from the Evaluator."""
    pass

  @abc.abstractmethod
  def predict_fn(self, dataset: tf.data.Dataset, state: Any):
    """The functions for prediction.

    A user-defined function, which takes in a tf.data.Dataset and a state, and
    outputs the sequence of predicted tokens. Only called if predict metrics
    exist for the tasks.

    Args:
      dataset: A tf.data.Dataset.
      state: A user defined state that constains attributes step and stub.
    """
    pass

  @abc.abstractmethod
  def predict_fn_with_aux(self, dataset: tf.data.Dataset, state: Any):
    """The functions for prediction.

    A user-defined function, which takes in a tf.data.Dataset and a state, and
    outputs the sequence of predicted tokens. Only called if predict metrics
    exist for the tasks.

    Args:
      dataset: A tf.data.Dataset.
      state: A user defined state that constains attributes step and stub.
    """
    pass

  @abc.abstractmethod
  def score_fn(self, dataset: tf.data.Dataset, state: Any):
    """The functions for scoring.

    A user-defined function, which takes in a tf.data.Dataset and a state, and
    outputs the log likelihood score of the targets. Only called if score
    metrics exist for the task.

    Args:
      dataset: A tf.data.Dataset.
      state: A user defined state that constains attributes step and stub.
    """
    pass

  @abc.abstractmethod
  def state_iterations(self):
    """Iterating over different states, e.g. checkpoints."""
    pass

  @abc.abstractproperty
  def feature_converter(self):
    """Feature converter for the task."""
    pass


class LocalModelInference(BaseModelInference):
  """The inference with a local model."""

  def __init__(self,
               model: models.BaseTransformerModel,
               dataset_cfg: utils.DatasetConfig,
               restore_checkpoint_cfg: utils.RestoreCheckpointConfig,
               partitioner: partitioning.BasePartitioner,
               fallback_init_rng: Optional[int] = None,
               load_optimizer_def: bool = True,
               strict: bool = True,
               **kwargs):
    """The initialization that doesn't require shapes.

    Args:
      model: The model object to use for inference.
      dataset_cfg: Specification for the dataset to infer based on.
      restore_checkpoint_cfg: Specification for the model parameter checkpoint
        to load.
      partitioner: Partitioner for the model parameters and data across devices.
      fallback_init_rng: A random seed used for parameter initialization during
        model re-loading when utils.RestoreCheckpointConfig.fallback_to_scratch
        is set to True. If None, parameter initialization is not allowed during
        model loading and having fallback_to_scratch enabled will result in an
        error.
      load_optimizer_def: Whether to load optimizer def.
      strict: Whether to restore all optimizer parameters from the checkpoint.
      **kwargs: All other parameters to be passed to the base class.
    """
    super().__init__(dataset_cfg, **kwargs)
    ds_vocabs = utils.get_vocabulary(self._dataset_cfg)
    if (ds_vocabs[0] != model.input_vocabulary or
        ds_vocabs[1] != model.output_vocabulary):
      raise ValueError(f'Model and Task vocabularies do not match:\n'
                       f'  task={self._dataset_cfg.mixture_or_task_name}\n'
                       f'  ds_vocabs=({ds_vocabs[0]}, {ds_vocabs[1]})\n'
                       f'  model.input_vocabulary={model.input_vocabulary}\n'
                       f'  model.output_vocabulary={model.output_vocabulary}\n')
    self._model = model
    self._feature_converter = model.FEATURE_CONVERTER_CLS(pack=False)  # pytype:disable=not-instantiable
    self._restore_checkpoint_cfg = restore_checkpoint_cfg
    self._partitioner = partitioner
    self._fallback_init_rng = fallback_init_rng
    self._load_optimizer_def = load_optimizer_def
    self._strict = strict
    self._log_file = (
        os.path.join(self._output_dir, 'model-info.txt')
        if self._output_dir is not None else None)

  def initialize_with_element_spec(self,
                                   element_spec: Mapping[str, tf.TypeSpec],
                                   batch_size: int,
                                   log_info: Optional[bool] = True):
    """T5X model loading.

    Args:
      element_spec: mapping from model feature to its shape in the
        `cached_model_datasets`.
      batch_size: batch size of the datasets.
      log_info: Whether to log information to self._log_file.
    """
    # Initialize optimizer from the existing checkpoint.
    input_shapes = {
        k: (batch_size,) + spec.shape for k, spec in element_spec.items()
    }
    input_types = {
        k: jnp.dtype(spec.dtype.as_numpy_dtype)
        for k, spec in element_spec.items()
    }
    self._train_state_initializer = utils.TrainStateInitializer(
        optimizer_def=(self._model.optimizer_def
                       if self._load_optimizer_def else None),
        init_fn=self._model.get_initial_variables,
        input_shapes=input_shapes,
        input_types=input_types,
        partitioner=self._partitioner)
    self._train_state_axes = self._train_state_initializer.train_state_axes
    if log_info and self._log_file is not None:
      # Log the variable shapes information and write to a file.
      utils.log_model_info(
          self._log_file,
          self._train_state_initializer.global_train_state_shape,
          self._partitioner)
    if self._fallback_init_rng is not None:
      self._fallback_init_rng = jax.random.PRNGKey(self._fallback_init_rng)

    # Compile the model only once.
    self._predict_fn = utils.get_infer_fn(
        infer_step=self._model.predict_batch,
        batch_size=batch_size,
        train_state_axes=self._train_state_axes,
        partitioner=self._partitioner)

    self._predict_fn_with_aux = utils.get_infer_fn(
        infer_step=self._model.predict_batch_with_aux,
        batch_size=batch_size,
        train_state_axes=self._train_state_axes,
        partitioner=self._partitioner)

    self._score_fn = utils.get_infer_fn(
        infer_step=self._model.score_batch,
        batch_size=batch_size,
        train_state_axes=self._train_state_axes,
        partitioner=self._partitioner)

  def predict_fn(self,
                 dataset: tf.data.Dataset,
                 state: Any,
                 rng: Optional[jnp.ndarray] = None):
    return self._predict_fn(dataset, state, rng)

  def predict_fn_with_aux(self,
                          dataset: tf.data.Dataset,
                          state: Any,
                          rng: Optional[jnp.ndarray] = None):
    return self._predict_fn_with_aux(dataset, state, rng)

  def score_fn(self,
               dataset: tf.data.Dataset,
               state: Any,
               rng: Optional[jnp.ndarray] = None):
    return self._score_fn(dataset, state, rng)

  def state_iterations(self):
    """Iterates over checkpoints and returns TrainState as state."""
    self._restore_checkpoint_cfg.strict = self._strict
    return self._train_state_initializer.from_checkpoints(
        [self._restore_checkpoint_cfg], init_rng=self._fallback_init_rng)

  @property
  def feature_converter(self):
    """Feature converter for the task."""
    return self._feature_converter


