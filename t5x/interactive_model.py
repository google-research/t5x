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

"""InteractiveModel class for use in T5X Colabs.

The InteractiveModel can be used to run training, inference, and evaluation on
natural text inputs and targets.

"""

import abc
from collections.abc import Mapping, Sequence
import enum
import functools
import inspect
import itertools
import logging
import os
import re
from typing import Any, Callable, Iterator, Optional, Tuple, Union

import clu.data.dataset_iterator
import jax
from jax import random
from jax.experimental import multihost_utils
import numpy as np
import seqio
from t5x import checkpoints
from t5x import models
from t5x import partitioning
from t5x import trainer as trainer_lib
from t5x import utils
from t5x.infer import _extract_tokens_and_aux_values
from t5x.infer import _Inferences
import tensorflow as tf
import tensorflow_datasets as tfds

BatchesType = Union[Sequence[Mapping[str, str]],
                    Sequence[Sequence[Mapping[str, str]]]]


class InferenceType(enum.Enum):
  PREDICT_WITH_AUX = 1
  SCORE = 2


class T5XScriptType(enum.Enum):
  FINETUNING = 1
  INFERENCE = 2
  EVALUATION = 3
  PRETRAINING = 4


class InteractiveModel(abc.ABC):
  """Wrapper around T5X components to enable interactive train/infer/eval."""

  def __init__(
      self,
      batch_size: int,
      task_feature_lengths: Mapping[str, int],
      output_dir: str,
      partitioner: partitioning.BasePartitioner,
      model: models.BaseTransformerModel,
      dtype: Optional[str],
      restore_mode: str,
      checkpoint_path: str,
      input_shapes: Mapping[str, utils.Array],
      input_types: Optional[Mapping[str, utils.DType]] = None,
      init_random_seed: int = 42,
      add_eos: bool = True,
      eval_names: Optional[Sequence[str]] = None,
  ):
    """Init function.

    Configures the output directory, RNGs, and partitioner given the provided
    arguments.

    Args:
      batch_size: number of examples per batch for training, inference, and
        evaluation.
      task_feature_lengths: dictionary mapping feature key to maximum length
        (int) for that feature. If feature is longer than this length after
        preprocessing, the feature will be truncated. May be set to None to
        avoid truncation.
      output_dir: Path to directory where we will write temporary files and
        final results.
      partitioner: the partitioner that defines how we divide and replicate
        machine learning model parameters, activations, and data across the
        accelerator devices (TPU/GPU). See https://github.com/google-research/t5x/blob/main/docs/usage.md/partitioning for
        details.
      model: the model object to use for training, inference, and evaluation.
      dtype: The dtype to restore ('float32' or 'bfloat16'), or None to load as
        saved.
      restore_mode: One of 'specific', 'latest', or 'all'. `specific` loads the
        checkpoint specified by `path`. `latest` loads the most recent
        checkpoint in the directory specified by `path`. `all` sequentially
        loads all of checkpoints in the directory `path`.
      checkpoint_path: Path(s) to checkpoint to restore from or directory
        (depending on `restore_mode`).
      input_shapes: a mapping from key to array shape for each feature in the
        global (unsharded) input batch.
      input_types: a mapping from key to array type for each feature in the
        global (unshared) input batch. If not provided, the type is assumed to
        be `jnp.float32`.
      init_random_seed: the random seed used to initialize all RNGs.
      add_eos: whether or not to add the EOS token to inputs/targets.
      eval_names: names of evaluation datasets, which must match the keys of the
        mapping passed to trainer's `eval` method.

    Raises:
      ValueError: the partitioner has an incorrect submesh, or the checkpoint
        restore function returned a sequence of TrainStates, when it should have
        returned a single TrainState.
    """
    self._batch_size = batch_size
    self._task_feature_lengths = task_feature_lengths
    self._cached_infer_fns = {}
    # --------------------------------------------------------------------------
    # Configure the output directory
    # --------------------------------------------------------------------------
    self._output_dir = output_dir
    # Remove double-slashes in directory path to avoid inconsistencies.
    self._output_dir = re.sub(r"(?<!gs:)([\/]{2,})", "/", self._output_dir)
    if not os.path.exists(self._output_dir):
      os.mkdir(self._output_dir)

    # --------------------------------------------------------------------------
    # Initialize RNGs
    # --------------------------------------------------------------------------
    self._init_random_seed = init_random_seed
    random_seed = multihost_utils.broadcast_one_to_all(
        np.int32(self._init_random_seed))
    utils.set_hardware_rng_ops()

    rng = random.PRNGKey(random_seed)
    self._init_rng, self._trainer_rng = random.split(rng, 2)

    # --------------------------------------------------------------------------
    # Initialize the partitioner.
    # --------------------------------------------------------------------------
    if partitioner._model_parallel_submesh:
      num_partitions = np.prod(partitioner._model_parallel_submesh)
    else:
      num_partitions = partitioner._num_partitions
    if jax.device_count() % num_partitions != 0:
      raise ValueError(
          "The number of devices available must be a multiple of the number of",
          f" partitions. There are {jax.device_count()} devices available, but",
          f" the number of partitions is set to {num_partitions}. Please",
          " provide a different number of partitions.")
    self._partitioner = partitioner

    # --------------------------------------------------------------------------
    # Create and save a checkpoint manager.
    # --------------------------------------------------------------------------
    logging.info("Initializing model, optimizer, and step functions.")
    self._model = model
    self._feature_converter = self._model.FEATURE_CONVERTER_CLS(pack=False)
    self._input_shapes = input_shapes
    self._input_types = input_types
    # Save the model vocabulary as features.
    output_features = {
        "inputs":
            seqio.Feature(
                vocabulary=self._model.input_vocabulary, add_eos=add_eos),
        "targets":
            seqio.Feature(
                vocabulary=self._model.output_vocabulary, add_eos=add_eos)
    }
    self._features = dict(sorted(output_features.items()))

    # Define restore and save checkpoints.
    if checkpoint_path:
      self._restore_checkpoint_cfg = utils.RestoreCheckpointConfig(
          dtype=dtype, mode=restore_mode, path=checkpoint_path
      )
    else:
      self._restore_checkpoint_cfg = None
    self._save_checkpoint_cfg = utils.SaveCheckpointConfig(
        dtype=dtype, keep=5, save_dataset=False, period=1000
    )
    self._train_state_initializer = utils.TrainStateInitializer(
        optimizer_def=self._model.optimizer_def,
        init_fn=self._model.get_initial_variables,
        input_shapes=self._input_shapes,
        input_types=self._input_types,
        partitioner=self._partitioner)

    # Initialize checkpoint manager.
    self._checkpoint_manager = utils.LegacyCheckpointManager(
        save_cfg=self._save_checkpoint_cfg,
        restore_cfg=self._restore_checkpoint_cfg,
        train_state_shape=(
            self._train_state_initializer.global_train_state_shape
        ),
        partitioner=self._partitioner,
        ds_iter=None,
        model_dir=self._output_dir,
    )

    # --------------------------------------------------------------------------
    # Restore a model from a checkpoint or from scratch.
    # --------------------------------------------------------------------------
    def get_state(rng):
      return self._train_state_initializer.from_scratch(rng).state_dict()

    restore_cfgs = []
    # 1. From a checkpoint specified by `self._restore_checkpoint_cfg.path`, if
    # set.
    if self._restore_checkpoint_cfg:
      restore_cfgs.append(self._restore_checkpoint_cfg)
    # 2. If no checkpoint provided, look for one in the model directory.
    if self._restore_checkpoint_cfg is not None:
      state_transforms_for_restore = [
          functools.partial(fn, is_resuming=True)
          for fn in self._restore_checkpoint_cfg.state_transformation_fns
      ]
    else:
      state_transforms_for_restore = []
    restore_cfgs.append(
        utils.RestoreCheckpointConfig(
            path=self._output_dir,
            mode="latest",
            dtype=self._save_checkpoint_cfg.dtype
            if self._save_checkpoint_cfg else "float32",
            checkpointer_cls=self._save_checkpoint_cfg.checkpointer_cls
            if self._save_checkpoint_cfg else checkpoints.Checkpointer,
            # Restore dataset state if it is being saved.
            restore_dataset=(self._save_checkpoint_cfg and
                             self._save_checkpoint_cfg.save_dataset),
            state_transformation_fns=state_transforms_for_restore))

    # Restore the model using a checkpoint.
    valid_restore_cfg, restore_paths = (
        utils.get_first_valid_restore_config_and_paths(restore_cfgs)
    )
    self._train_state = self._checkpoint_manager.restore(
        restore_paths, valid_restore_cfg,
        utils.get_fallback_state(valid_restore_cfg, get_state, self._init_rng))

    # 3. If no checkpoint to restore, init from scratch.
    if self._train_state is None:
      self._train_state = self._train_state_initializer.from_scratch(
          self._init_rng)
    self._train_state_axes = self._train_state_initializer.train_state_axes

    # Log the variable shapes information and write to a file.
    log_file = os.path.join(self._output_dir, "model-info.txt")
    utils.log_model_info(log_file,
                         self._train_state_initializer.global_train_state_shape,
                         self._partitioner)

    # --------------------------------------------------------------------------
    # Trainer
    # --------------------------------------------------------------------------
    if isinstance(self._train_state, Sequence):
      raise ValueError(
          "Expected a single train state, but instead received a Sequence.")
    self._trainer = trainer_lib.Trainer(
        model=self._model,
        train_state=self._train_state,
        partitioner=self._partitioner,
        eval_names=eval_names if eval_names else [],
        summary_dir=self._output_dir,
        train_state_axes=self._train_state_axes,
        rng=self._trainer_rng,
        learning_rate_fn=utils.create_learning_rate_scheduler(),
        num_microbatches=None,
    )

  @property
  def trainer(self):
    return self._trainer

  @property
  def partitioner(self):
    return self._partitioner

  @property
  def model(self):
    return self._model

  @property
  def train_state(self):
    return self._train_state

  @property
  def train_state_axes(self):
    return self._train_state_axes

  @property
  def train_summary(self):
    return self._train_summary.result()

  @property
  def step(self):
    if isinstance(self._train_state, Sequence):
      raise ValueError(
          "Expected a single train state, but instead received a Sequence.")
    return int(self._train_state.step)

  def train_step(self, examples: Sequence[Union[str, dict[str, str]]]):
    """Train function.

    Args:
      examples: examples that should be transformed into a tf.data.Dataset. The
        examples can either take the form of a string (ex: a single input for
        inference), or a dictionary mapping "input"/"target" to a string
        containing that element. At least `self._batch_size` examples must be
        provided.

    Raises:
      ValueError: the user provided less than `batch_size` examples, or
        `self._train_state` was set to a sequence of TrainStates, when it should
        have been a single TrainState.
    """
    # By default, only tokenize and append EOS.
    preprocessors = [
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos,
    ]
    self.train_step_with_preprocessors(
        examples=examples, preprocessors=preprocessors)

  def train_step_with_preprocessors(
      self, examples: Sequence[Union[str, dict[str, str]]],
      preprocessors: Sequence[Callable[..., tf.data.Dataset]]):
    """Train function.

    Args:
      examples: examples that should be transformed into a tf.data.Dataset. The
        examples can either take the form of a string (ex: a single input for
        inference), or a dictionary mapping "input"/"target" to a string
        containing that element. At least `self._batch_size` examples must be
        provided.
      preprocessors: list(callable), an optional list of functions that receive
        a tf.data.Dataset and return a tf.data.Dataset. These will be executed
        sequentially and the final dataset must include features matching
        `self._features`.

    Raises:
      ValueError: the user provided less than `batch_size` examples, or
        `self._train_state` was set to a sequence of TrainStates, when it should
        have been a single TrainState.
    """
    # --------------------------------------------------------------------------
    # Initialize dataset and dataset iterator
    # --------------------------------------------------------------------------
    if len(examples) < self._batch_size:
      raise ValueError(
          "At least one batch of data must be provided. Please decrease the "
          "batch_size or provide more examples.")

    train_dataset = get_dataset_from_natural_text_examples(
        examples,
        preprocessors=preprocessors,
        task_feature_lengths=self._task_feature_lengths,
        features=self._features)
    train_dataset = self._feature_converter(
        train_dataset, task_feature_lengths=self._task_feature_lengths)
    train_dataset = train_dataset.padded_batch(
        self._batch_size, drop_remainder=True)
    train_iter = clu.data.dataset_iterator.TfDatasetIterator(
        train_dataset, checkpoint=True)

    # --------------------------------------------------------------------------
    # Take 1 train step.
    # --------------------------------------------------------------------------
    # `stop_training` is requested, break out the main loop immediately.
    if self._trainer.stop_training:
      logging.info(
          "Stopping training early since `stop_training` is requested.")
      return

    try:
      self.train_step_from_batch_iterator(train_iter)
    except trainer_lib.PreemptionError as e:
      logging.info("Saving emergency checkpoint.")
      self.save_checkpoint()
      logging.info("Saving emergency checkpoint done.")
      raise e

    # Save a checkpoint.
    logging.info("Saving checkpoint.")
    self.save_checkpoint()

  def train_step_from_batch_iterator(self, iterator):
    """Runs one training step from a batch iterator."""
    if isinstance(self._train_state, Sequence):
      raise ValueError(
          "Expected a single train state, but instead received a Sequence."
      )

    first_step = int(utils.get_local_data(self._train_state.step))
    self._train_summary = self._trainer.train(
        iterator, 1, start_step=first_step
    )

    # Wait until computations are done before exiting
    utils.sync_global_devices("complete")
    self._train_state = self._trainer.train_state

  def save_checkpoint(self):
    """Saves model checkpoint."""
    self._checkpoint_manager.save(
        self._trainer.train_state,
        self._save_checkpoint_cfg.state_transformation_fns,
    )

  def infer_with_preprocessors(
      self,
      mode: InferenceType,
      examples: Sequence[Union[str, dict[str, str]]],
      preprocessors: Sequence[Callable[..., tf.data.Dataset]],
      **inference_kwargs,
  ) -> _Inferences:
    """Infer function.

    Args:
      mode: Either 'score' to compute the log likelihood of given targets, or
        'predict_with_aux' to score and decode targets.
      examples: examples that should be transformed into a tf.data.Dataset. The
        examples can either take the form of a string (ex: a single input for
        inference), or a dictionary mapping "input"/"target" to a string
        containing that element.
      preprocessors: list(callable), an optional list of functions that receive
        a tf.data.Dataset and return a tf.data.Dataset. These will be executed
        sequentially and the final dataset must include features matching
        `self._features`.
      **inference_kwargs: additional keyword arguments to pass to the inference
        function (e.g., `model.predict_batch_with_aux` or `score_batch`).

    Returns:
      Returns a tuple of predictions/scores and any auxiliary values.
    """
    # --------------------------------------------------------------------------
    # Parse Mode
    # --------------------------------------------------------------------------
    if mode == InferenceType.PREDICT_WITH_AUX:
      infer_step = self._model.predict_batch_with_aux
    elif mode == InferenceType.SCORE:
      infer_step = self._model.score_batch
    else:
      raise ValueError("Mode must be `predict_with_aux`, or `score`,"
                       f" but instead was {mode}.")
    key_array = seqio.utils.flatten_dict(inference_kwargs)
    key_array["mode"] = mode
    infer_fn_key = tuple(key_array.items())
    if infer_fn_key not in self._cached_infer_fns:
      self._cached_infer_fns[infer_fn_key] = utils.get_infer_fn(
          infer_step=functools.partial(infer_step, **inference_kwargs),
          batch_size=self._batch_size,
          train_state_axes=self._train_state_initializer.train_state_axes,
          partitioner=self._partitioner,
      )
    infer_fn = functools.partial(
        self._cached_infer_fns[infer_fn_key],
        train_state=self._train_state,
    )

    # --------------------------------------------------------------------------
    # Construct a dataset and dataset iterator.
    # --------------------------------------------------------------------------
    dataset = get_dataset_from_natural_text_examples(
        examples,
        preprocessors=preprocessors,
        task_feature_lengths=self._task_feature_lengths,
        features=self._features)
    model_dataset = self._feature_converter(
        dataset, task_feature_lengths=self._task_feature_lengths)
    # Zip task and model features.
    infer_dataset = tf.data.Dataset.zip((dataset, model_dataset))
    # Create batches and index them.
    infer_dataset = infer_dataset.padded_batch(
        self._batch_size, drop_remainder=False).enumerate()
    infer_dataset_iter: Iterator[Tuple[int, Any]] = iter(
        infer_dataset.prefetch(tf.data.experimental.AUTOTUNE))

    # --------------------------------------------------------------------------
    # Run inference
    # --------------------------------------------------------------------------
    # Main Loop over "batches".
    all_inferences = []
    all_aux_values = {}
    for chunk, chunk_batch in infer_dataset_iter:
      # Load the dataset for the next chunk. We can't use `infer_dataset_iter`
      # directly since `infer_fn` needs to know the exact size of each chunk,
      # which may be smaller for the final one.
      chunk_dataset = tf.data.Dataset.from_tensor_slices(chunk_batch)
      chunk_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)

      # Unzip chunk dataset in to pretokenized and model datasets.
      task_dataset = chunk_dataset.map(
          lambda p, m: p, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      model_dataset = chunk_dataset.map(
          lambda p, m: m, num_parallel_calls=tf.data.experimental.AUTOTUNE)

      # Get a chunk-specific RNG key.
      chunk_rng = jax.random.fold_in(jax.random.PRNGKey(0), chunk)

      inferences = _extract_tokens_and_aux_values(
          infer_fn(model_dataset.enumerate(), rng=chunk_rng))

      predictions, aux_values = inferences
      accumulated_inferences = []
      for idx, inputs in task_dataset.enumerate().as_numpy_iterator():
        prediction = predictions[idx]
        # Decode predictions if applicable.
        if mode == InferenceType.PREDICT_WITH_AUX:
          prediction = self._features["targets"].vocabulary.decode_tf(
              tf.constant(prediction)).numpy()
        accumulated_inferences.append((inputs, prediction))
      all_inferences += accumulated_inferences
      # Accumulate aux values over batches.
      if not all_aux_values:
        all_aux_values = aux_values
      else:
        for key, values in aux_values.items():
          all_aux_values[key] += values

    return all_inferences, all_aux_values

  def predict_with_aux(
      self, examples: Sequence[Union[str, dict[str, str]]]) -> _Inferences:
    """Predict with auxiliary values method."""
    # By default, only tokenize and append EOS.
    preprocessors = [
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos,
    ]
    return self.infer_with_preprocessors(
        mode=InferenceType.PREDICT_WITH_AUX,
        examples=examples,
        preprocessors=preprocessors)

  def score(self, examples: Sequence[Union[str, dict[str,
                                                     str]]]) -> Sequence[Any]:
    """Score method."""
    # By default, only tokenize and append EOS.
    preprocessors = [
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos,
    ]
    # Ignore auxiliary values.
    scores, _ = self.infer_with_preprocessors(
        mode=InferenceType.SCORE,
        examples=examples,
        preprocessors=preprocessors)
    return scores

  def _compute_metrics(
      self, targets: Sequence[Any], predictions: Sequence[Any],
      aux_values: Sequence[Any], scores: Sequence[Any],
      predict_metric_fns: Sequence[seqio.dataset_providers.MetricFnCallable],
      predict_with_aux_metric_fns: Sequence[
          seqio.dataset_providers.MetricFnCallable],
      score_metric_fns: Sequence[seqio.dataset_providers.MetricFnCallable]):
    """Computes the metrics specified in the metric_fns lists."""
    # Only compute metrics once
    if jax.process_index() != 0:
      return {}

    def compute_metrics_fn():
      task_metrics = []
      if predict_metric_fns:
        task_metrics.extend([
            metric_fn(targets, predictions) for metric_fn in predict_metric_fns
        ])
      if predict_with_aux_metric_fns:
        task_metrics.extend([
            metric_fn(targets, predictions, aux_values)
            for metric_fn in predict_with_aux_metric_fns
        ])
      if score_metric_fns:
        is_tuple = isinstance(scores, tuple)
        if ((not is_tuple and len(targets) != len(scores)) or
            (is_tuple and len(targets) != len(scores[0]))):
          raise ValueError(f"len(targets)({len(targets)}) != "
                           f"len(output_scores)({len(scores)})")
        task_metrics.extend(
            [metric_fn(targets, scores) for metric_fn in score_metric_fns])

      all_metrics = {}
      for k, v in itertools.chain(*[m.items() for m in task_metrics]):
        if k in all_metrics:
          raise ValueError(f"Duplicate metric key '{k}' in Task.")
        all_metrics[k] = v
      return all_metrics

    if not tf.executing_eagerly():

      def wrap_graph(fn):
        graph = tf.compat.v1.get_default_graph()

        def wrapped_fn():
          with graph.as_default():
            return fn()

        return wrapped_fn

      compute_metrics_fn = wrap_graph(compute_metrics_fn)

    all_metrics = compute_metrics_fn()
    # Wait until computations are done before continuing.
    utils.sync_global_devices("Completed.")
    return all_metrics

  def evaluate(
      self,
      examples: Sequence[Union[str, dict[str, str]]],
      metric_fns: Sequence[seqio.dataset_providers.MetricFnCallable],
  ) -> Mapping[Any, Any]:
    """Evaluation function.

    Args:
      examples: examples that should be transformed into a tf.data.Dataset. The
        examples can either take the form of a string (ex: a single input for
        inference), or a dictionary mapping "input"/"target" to a string
        containing that element.
      metric_fns: list(callable), an optional list of metric functions with a
        signature that matches one of three possible forms: - (targets, scores)
        - Note that `scores` refers to the score the model assigned the target
        sequence, given the input. - (targets, predictions) - (targets,
        predictions, aux_values) - Note that `aux_values` refers to a dictionary
        of auxiliary values that the model assigned to each sequence.

    Returns:
      Mapping of metrics names to metrics values.
    """
    # By default, only tokenize and append EOS.
    preprocessors = [
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos,
    ]
    return self.evaluate_with_preprocessors(
        examples=examples,
        preprocessors=preprocessors,
        metric_fns=metric_fns,
        postprocessor=None)

  def evaluate_with_preprocessors(
      self,
      examples: Sequence[dict[str, str]],
      preprocessors: Sequence[Callable[..., tf.data.Dataset]],
      metric_fns: Sequence[seqio.dataset_providers.MetricFnCallable],
      postprocessor: Optional[Callable[..., Any]] = None,
  ) -> Mapping[Any, Any]:
    """Evaluation function.

    Args:
      examples: examples that should be transformed into a tf.data.Dataset. The
        examples must take the form of a dictionary mapping "input"/"target" to
        a string containing that element.
      preprocessors: list(callable), an optional list of functions that receive
        a tf.data.Dataset and return a tf.data.Dataset. These will be executed
        sequentially and the final dataset must include features matching
        `self._features`.
      metric_fns: list(callable), an optional list of metric functions with a
        signature that matches one of three possible forms: - (targets, scores)
        - Note that `scores` refers to the score the model assigned the target
        sequence, given the input. - (targets, predictions) - (targets,
        predictions, aux_values) - Note that `aux_values` refers to a dictionary
        of auxiliary values that the model assigned to each sequence.
      postprocessor: callable, an optional function that receives decoded model
        outputs and converts them to a form that is ready for evaluation using
        the metric functions in `metric_fns`.

    Returns:
      Mapping of metrics names to metrics values.
    """
    # --------------------------------------------------------------------------
    # Parse Metrics functions
    # --------------------------------------------------------------------------
    predict_metric_fns = []
    predict_with_aux_metric_fns = []
    score_metric_fns = []
    for metric_fn in metric_fns:
      pos_args = tuple(
          key for key, param in inspect.signature(metric_fn).parameters.items()
          if param.default == inspect.Parameter.empty)
      if pos_args == ("targets", "scores"):
        score_metric_fns.append(metric_fn)
      elif pos_args == ("targets", "predictions"):
        predict_metric_fns.append(metric_fn)
      elif pos_args == ("targets", "predictions", "aux_values"):
        predict_with_aux_metric_fns.append(metric_fn)
      else:
        raise ValueError(
            "Metric functions must have positional arguments matching either "
            "('targets', 'scores'), ('targets', 'predictions') or "
            "('targets', 'predictions', 'aux_values'). "
            f"Got: {pos_args}")

    # ------------------------------------------------------------------------
    # Get targets, predictions, and scores
    # ------------------------------------------------------------------------
    dataset = get_dataset_from_natural_text_examples(
        examples,
        preprocessors=preprocessors,
        task_feature_lengths=self._task_feature_lengths,
        features=self._features)

    # Get targets.
    def postprocess_fn(decoded_model_output: Any, **postprocess_kwargs) -> Any:
      """Returns the model output after applying the postprocess function."""
      if postprocessor:
        return postprocessor(decoded_model_output, **postprocess_kwargs)
      return decoded_model_output

    targets = []
    for ex in tfds.as_numpy(dataset):
      targets.append(
          postprocess_fn(
              decoded_model_output=ex["targets_pretokenized"],
              example=ex,
              is_target=True))

    # Get predictions.
    predictions = []
    if predict_with_aux_metric_fns or predict_metric_fns:
      predictions, aux_values = self.infer_with_preprocessors(
          mode=InferenceType.PREDICT_WITH_AUX,
          examples=examples,
          preprocessors=preprocessors)
      predictions = [
          prediction.decode("utf-8") for example, prediction in predictions
      ]
    # Get scores.
    scores = []
    if score_metric_fns:
      scores, _ = self.infer_with_preprocessors(
          mode=InferenceType.SCORE,
          examples=examples,
          preprocessors=preprocessors)
      scores = [score for example, score in scores]

    return self._compute_metrics(
        targets,
        predictions,
        aux_values,
        scores,  # pytype: disable=wrong-arg-types  # mapping-is-not-sequence
        predict_metric_fns,
        predict_with_aux_metric_fns,
        score_metric_fns)

  def train_loop(
      self,
      num_steps: int,
      eval_period: Optional[int] = 1,
      train_batches: Optional[BatchesType] = None,
      predict_batches: Optional[BatchesType] = None,
      score_batches: Optional[BatchesType] = None,
      eval_batches: Optional[BatchesType] = None,
      metrics_fns: Optional[Sequence[
          seqio.dataset_providers.MetricFnCallable]] = None,
  ):
    """Runs training, inference, and evaluation for `num_steps`.

    It should be noted that there are many different possible variants of the
    `train_loop` function that a user might want to use. The primary goal of the
    `train_loop` function is not to cover all the potential training loop
    variants that a user may want; rather, the goal is to demonstrate how the
    user could stack the `InteractiveModel` train, predict, score, and evaluate
    methods.

    Args:
      num_steps: the number of steps to run for training, inference, and
        evaluation.
      eval_period: specifies how many steps to take between
        inference/evaluation.
      train_batches: an optional list of batches that we should run training on.
        If no batches are provided, then training will be skipped. If a single
        batch is provided, we will repeat training on this batch for
        `num_steps`.
      predict_batches: an optional list of batches that we should get
        predictions for. If no batches are provided, then predicting will be
        skipped. If a single batch is provided, we will repeatedly get
        predictions on this batch for `num_steps`.
      score_batches: an optional list of batches that we should score. If no
        batches are provided, then scoring will be skipped. If a single batch is
        provided, we will repeatedly score this batch for `num_steps`.
      eval_batches: an optional list of batches that we should run eval on. If
        no batches are provided, then evaluation will be skipped. If a single
        batch is provided, we will repeatedly evaluate this batch for
        `num_steps`.
      metrics_fns: list(callable), an optional list of metric functions with a
        signature that matches one of three possible forms: - (targets, scores)
        - Note that `scores` refers to the score the model assigned the target
        sequence, given the input. - (targets, predictions) - (targets,
        predictions, aux_values) - Note that `aux_values` refers to a dictionary
        of auxiliary values that the model assigned to each sequence.

    Returns:
      Predictions, scores, and metrics for the final step of the training loop.
    """
    # Ensure all batches are `num_steps` in length
    train_batches = _get_equal_length_batches(train_batches, num_steps)

    predictions = None
    scores = None
    metrics = None
    for step_num, train_batch in enumerate(train_batches):
      if train_batch:
        self.train_step(train_batch)
      # Run inference/evaluation every `eval_period` steps.
      if step_num % eval_period == 0:
        # Run on all batches for inference/evaluation.
        if predict_batches:
          for predict_batch in predict_batches:
            predictions, _ = self.predict_with_aux(predict_batch)  # pytype: disable=wrong-arg-types  # mapping-is-not-sequence
        if score_batches:
          for score_batch in score_batches:
            scores = self.score(score_batch)  # pytype: disable=wrong-arg-types  # mapping-is-not-sequence
        if eval_batches:
          for eval_batch in eval_batches:
            metrics = self.evaluate(eval_batch, metrics_fns)  # pytype: disable=wrong-arg-types  # mapping-is-not-sequence
    return predictions, scores, metrics


def get_dataset_from_natural_text_examples(
    examples: Sequence[Union[str, dict[str, str]]],
    preprocessors: Sequence[Callable[..., tf.data.Dataset]],
    task_feature_lengths: Mapping[str, int],
    features: Mapping[str, Any]) -> tf.data.Dataset:
  """Returns a tf.data.Dataset from a list of examples.

  Args:
    examples: a single batch of examples that should be transformed into a
      tf.data.Dataset. The examples can either take the form of a string (ex: a
      single input for inference), or a dictionary mapping "input"/"target" to a
      string containing that element.
    preprocessors: an optional list of functions that receive a tf.data.Dataset
      and return a tf.data.Dataset. These will be executed sequentially and the
      final dataset must include features matching `self._features`.
    task_feature_lengths: dictionary mapping feature key to maximum length (int)
      for that feature. If feature is longer than this length after
      preprocessing, the feature will be truncated. May be set to None to avoid
      truncation.
    features: dictionary defining what features should be present in all
      examples.

  Returns:
    A tf.data.Dataset.
  """
  # ------------------------------------------------------------------------
  # Construct a `tf.data.Dataset` from the provided examples
  # ------------------------------------------------------------------------
  merged_examples = {"inputs": [], "targets": []}
  for example in examples:
    # If the provided example is just a string, add an empty target string
    if isinstance(example, dict):
      example_dict = example
    else:
      example_dict = {"input": example, "target": ""}
    merged_examples["inputs"].append(example_dict["input"])
    merged_examples["targets"].append(example_dict["target"])
  dataset = tf.data.Dataset.from_tensor_slices(merged_examples)

  # Define `ShardInfo` that doesn't shard the data pipeline.
  shard_info = seqio.ShardInfo(0, 1)
  dataset = dataset.shard(shard_info.num_shards, shard_info.index)

  # ------------------------------------------------------------------------
  # Preprocess data
  # ------------------------------------------------------------------------
  for prep_fn in preprocessors:
    # prep_fn must not rely on variable length keyword args such as **kwargs.
    fn_args = set(inspect.signature(prep_fn).parameters.keys())
    kwargs = {}
    if "sequence_length" in fn_args:
      kwargs["sequence_length"] = task_feature_lengths
    if "output_features" in fn_args:
      kwargs["output_features"] = features
    dataset = prep_fn(dataset, **kwargs)

  def _validate_preprocessing(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Validates preprocessed dataset, raising Exceptions if needed.

    Args:
      dataset: a tf.data.Dataset to validate.

    Returns:
      a validated tf.data.Dataset.

    Raises:
      ValueError: dataset has missing feature or the incorrect type/rank for a
        feature.
    """
    actual_specs = dataset.element_spec
    for feat, feat_spec in features.items():
      if feat not in actual_specs:
        if feat_spec.required:
          raise ValueError(
              "Task dataset is missing expected output feature after "
              f"preprocessing: {feat}")
        else:
          # It's ok that this feature does not exist.
          continue
      actual_spec = actual_specs[feat]
      if feat_spec.dtype != actual_spec.dtype:
        raise ValueError(
            f"Task dataset has incorrect type for feature '{feat}' after "
            f"preprocessing: Got {actual_spec.dtype.name}, expected "
            f"{feat_spec.dtype.name}")
      if feat_spec.rank != actual_spec.shape.rank:
        raise ValueError(
            f"Task dataset has incorrect rank for feature '{feat}' after "
            f"preprocessing: Got {actual_spec.shape.rank}, expected "
            f"{feat_spec.rank}")

    return dataset

  dataset = _validate_preprocessing(dataset)
  dataset = seqio.utils.trim_dataset(dataset, task_feature_lengths, features)
  return dataset.prefetch(tf.data.experimental.AUTOTUNE)


def _get_equal_length_batches(batches: BatchesType,
                              length: int) -> Sequence[Any]:
  """Produces a list of batches that is `length` batches long.

  Given a single batch, repeat the batch `length` times.

  Given a list of batches, either repeat the batches to get `length` total
  batches or take the first 'length' batches.

  Args:
    batches: either a single batch of examples, or a list of batches.
    length: the total number of batches that should be present in the final
      list.

  Returns:
    A list of batches.
  """
  # Given a list of batches, return a list of batches that is `length` long,
  # either by repeating the batches or taking the first `length` batches
  if not batches:
    return [None] * length
  if isinstance(batches[0], Mapping):
    return [batches for i in range(length)]
  if len(batches) < length:
    batches = batches * (length // len(batches))
  # If multiple batches are provided, only use the first `length` batches.
  logging.warning(
      "We will only use the first %s batches provided for training.", length)
  return batches[:length]


def get_batches_from_seqio(
    task_or_mixture_name: str,
    split: str,
    batch_size: int,
    num_batches: int,
    get_pretokenized_examples: bool = True,
    sequence_length: Optional[Mapping[str, int]] = None,
    **get_dataset_kwargs) -> Sequence[Sequence[Mapping[str, str]]]:
  """Returns a batch of examples from a provided SeqIO task.

  Args:
    task_or_mixture_name: the SeqIO task/mixture to read data from.
    split: the split of the SeqIO task/mixture to read data from.
    batch_size: how many examples should be in each batch.
    num_batches: the total number of batches to return.
    get_pretokenized_examples: a bool, where True indicates that we should
      return the natural text (pre-tokenization) inputs and targets. Default to
      True in order to make the examples easy to debug/inspect.
    sequence_length: dictionary mapping feature key to maximum length (int) for
      that feature. Used by SeqIO to get the dataset.
    **get_dataset_kwargs: any additional arguments that should be passed to the
      SeqIO `get_dataset()` call.

  Returns:
    A sequence of batches, where each batch is a sequence of examples. Each
      example is a dictionary mapping 'input' and 'target' to the corresponding
      values for a single example.
  """
  task_or_mixture = seqio.get_mixture_or_task(task_or_mixture_name)
  total_examples_requested = batch_size * num_batches
  dataset = task_or_mixture.get_dataset(
      sequence_length=sequence_length, split=split, **get_dataset_kwargs
  )

  all_batches = []
  current_batch = []
  input_key = "inputs_pretokenized" if get_pretokenized_examples else "inputs"
  target_key = (
      "targets_pretokenized" if get_pretokenized_examples else "targets"
  )
  total_examples_seen = 0
  # It should be noted that we could replace the following loop with tf.Dataset
  # operations (like
  # `list(dataset.batch(batch_size).take(num_batches).as_numpy_iterator())`),
  # but this would require us to pad batches first or represent the token IDs as
  # ragged tensors. These approaches are currently overkill for the
  # InteractiveModel, but may be investigated in the future.
  dataset = dataset.take(total_examples_requested)
  for idx, element in enumerate(dataset.as_numpy_iterator()):
    total_examples_seen += 1
    if idx >= total_examples_requested:
      # Because we force `num_examples_requested` to be a multiple of
      # `batch_size`, this should enforce that the last batch always has the
      # same number of examples as all other batches.
      break

    example_input = element[input_key]
    example_target = element[target_key]
    if not get_pretokenized_examples:
      example_input = example_input.tolist()
      example_target = example_target.tolist()
    current_example = {"input": example_input, "target": example_target}
    current_batch.append(current_example)

    # If we've collected `batch_size` examples, save the current batch and start
    # a new batch.
    if len(current_batch) == batch_size:
      all_batches.append(current_batch)
      current_batch = []

  if total_examples_seen < total_examples_requested:
    raise ValueError("Not enough examples in Task/Mixture. User requested "
                     f"{num_batches} batches of size {batch_size} for a total "
                     f"of {total_examples_requested} examples. Only "
                     f"{total_examples_seen} available in "
                     "Task/Mixture.")

  return all_batches


def get_seqio_task_from_examples(
    task_name: str,
    interactive_model: InteractiveModel,
    examples: Sequence[Union[str, dict[str, str]]],
    preprocessors: Sequence[Callable[..., tf.data.Dataset]],
    metric_fns: Optional[
        Sequence[seqio.dataset_providers.MetricFnCallable]
    ] = None,
    add_to_registry: bool = True,
) -> Union[seqio.Task, seqio.Mixture]:
  """Registers and returns a SeqIO task from the provided inputs.

  This function will be used to graduate people to the T5X/SeqIO-based
  train/infer/eval scripts.

  Args:
    task_name: the name of the SeqIO task to be created and registered.
    interactive_model: an instance of the InteractiveModel.
    examples: a single batch of examples that should be transformed into a
      tf.data.Dataset. The examples can either take the form of a string (ex: a
      single input for inference), or a dictionary mapping "input"/"target" to a
      string containing that element.
    preprocessors: an optional list of functions that receive a tf.data.Dataset
      and return a tf.data.Dataset. These will be executed sequentially and the
      final dataset must include features matching `self._features`.
    metric_fns: list(callable), an optional list of metric functions with a
      signature that matches one of three possible forms: - (targets, scores) -
      Note that `scores` refers to the score the model assigned the target
      sequence, given the input. - (targets, predictions) - (targets,
      predictions, aux_values) - Note that `aux_values` refers to a dictionary
      of auxiliary values that the model assigned to each sequence.
    add_to_registry: if True, will register the new task.

  Returns:
    A SeqIO task.
  """

  def dataset_fn(split, shuffle_files):
    del split, shuffle_files
    return get_dataset_from_natural_text_examples(
        examples,
        preprocessors=[],
        task_feature_lengths=interactive_model._task_feature_lengths,  # pylint: disable=protected-access
        features={})

  data_source = seqio.FunctionDataSource(
      dataset_fn=dataset_fn, splits=["train", "validation"])

  if add_to_registry:
    seqio.TaskRegistry.add(
        task_name,
        data_source,
        preprocessors=preprocessors,
        output_features=interactive_model._features,  # pylint: disable=protected-access
        metric_fns=metric_fns)

  return seqio.get_mixture_or_task(task_name)


# pylint: disable=protected-access
def get_gin_config_from_interactive_model(interactive_model: InteractiveModel,
                                          script_type: T5XScriptType,
                                          task_name: str,
                                          partitioner_config_str: str,
                                          model_config_str: str,
                                          train_steps: int = 1,
                                          imports_str: str = ""):
  """Converts an InteractiveModel instance into a Gin config string.

  This function will be used to graduate people to the T5X/SeqIO-based
  train/infer/eval scripts.

  Args:
    interactive_model: an instance of the InteractiveModel.
    script_type: which T5X script the Gin config should function with.
    task_name: the name of the SeqIO task to be used.
    partitioner_config_str: a string that defines the Partitioner object in the
      Gin config.
    model_config_str: a string that defines the Model object in the Gin config.
    train_steps: the number of steps to train for, only used if FINETUNING or
      PRETRAINING is selected as the script type.
    imports_str: if the `model_config_str` or `partitioner_config_str` relies on
      some other files to be imported, these import statements can be included
      in the final Gin file by adding them to this string.

  Returns:
    A string that contains the full Gin file to be used for train/infer/eval.py.
  """
  restore_config_str = ""
  if interactive_model._restore_checkpoint_cfg:
    restore_config_str = f"""CHECKPOINT_PATH = '{interactive_model._restore_checkpoint_cfg.path}'
utils.RestoreCheckpointConfig:
  path = %CHECKPOINT_PATH
  mode = '{interactive_model._restore_checkpoint_cfg.mode}'
  dtype = '{interactive_model._restore_checkpoint_cfg.dtype}'"""

  base_config_str = f"""
{imports_str}

MODEL_DIR = "{interactive_model._output_dir}"
MIXTURE_OR_TASK_NAME = "{task_name}"
TASK_FEATURE_LENGTHS = {interactive_model._task_feature_lengths}
USE_CACHED_TASKS = False
SHUFFLE_TRAIN_EXAMPLES = False
BATCH_SIZE = {interactive_model._batch_size}

{model_config_str}
{partitioner_config_str}
{restore_config_str}"""

  if script_type == T5XScriptType.INFERENCE:
    if not interactive_model._restore_checkpoint_cfg:
      raise ValueError("A checkpoint must be provided to run inference.")
    gin_config = f"""
include 't5x/configs/runs/infer.gin'
{base_config_str}

INFER_OUTPUT_DIR = %MODEL_DIR

utils.DatasetConfig:
  use_cached = %USE_CACHED_TASKS
  batch_size = %BATCH_SIZE
  shuffle = False
  seed = 0
  pack = False
"""
  elif (
      script_type == T5XScriptType.FINETUNING
      or script_type == T5XScriptType.PRETRAINING
      or script_type == T5XScriptType.EVALUATION
  ):
    gin_config = f"""
from __gin__ import dynamic_registration

import __main__ as train_script
from t5x import utils

include 't5x/configs/runs/pretrain.gin'
{base_config_str}
utils.SaveCheckpointConfig:
  period = {interactive_model._save_checkpoint_cfg.period}
  dtype = '{interactive_model._save_checkpoint_cfg.dtype}'
  keep = {interactive_model._save_checkpoint_cfg.keep}
  save_dataset = {interactive_model._save_checkpoint_cfg.save_dataset}

TRAIN_STEPS = {train_steps}
SHUFFLE_TRAIN_EXAMPLES = False
DROPOUT_RATE = 0.0

train/utils.DatasetConfig:
  pack = False

train_eval/utils.DatasetConfig:
  pack = False
"""
    if script_type == T5XScriptType.EVALUATION:
      gin_config += """
train_script.train:
  run_eval_before_training = True
  eval_period = 0
  total_steps = 0
"""
  return gin_config


# pylint: enable=protected-access
