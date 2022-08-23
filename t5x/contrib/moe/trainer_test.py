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

"""Tests for trainer."""

import contextlib

from absl.testing import absltest
import jax
from jax._src import dispatch as jax_dispatch
import numpy as np
from t5x import metrics as metrics_lib
from t5x import models as models_lib
from t5x import optimizers
from t5x import train_state as train_state_lib
from t5x.contrib.moe import partitioning
from t5x.contrib.moe import trainer as trainer_lib
import tensorflow as tf

mock = absltest.mock
jax.config.parse_flags_with_absl()


# Make `log_elapsed_time` a no-op to simplify mocking of `time.time()`.
@contextlib.contextmanager
def fake_log_elapsed_time(_):
  yield


jax_dispatch.log_elapsed_time = fake_log_elapsed_time


def fake_accum_grads(model, optimizer, batch, rng, num_microbatches,
                     data_partition_spec):
  del model, num_microbatches, rng, data_partition_spec
  # Add `i` to each optimzer value.
  i = batch['i'].sum()
  grad_accum = jax.tree_map(lambda x: i, optimizer)
  # Add j to each metric.
  j = batch['j'].sum()
  metrics = {
      'loss': metrics_lib.Sum.from_model_output(j),
      'accuracy': metrics_lib.Sum.from_model_output(j)
  }
  return grad_accum, metrics, None


def fake_apply_grads(optimizer,
                     grad_accum,
                     metrics,
                     learning_rate,
                     weight_metrics_computer,
                     other_state_variables=None):
  del weight_metrics_computer
  del other_state_variables
  metrics['learning_rate'] = metrics_lib.Sum.from_model_output(learning_rate)
  optimizer = jax.tree_map(lambda x, g: x + g, optimizer, grad_accum)
  return optimizer, metrics


class MoeTrainerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.init_optimizer = optimizers.Optimizer(
        optimizers.sgd(0.1),
        state=optimizers.OptimizerState(
            step=0, param_states={
                'expert_bias': 0,
                'kernel': 0
            }),
        target={
            'expert_bias': np.zeros(4),
            'kernel': np.zeros((2, 4))
        })
    self.init_train_state = train_state_lib.FlaxOptimTrainState(
        self.init_optimizer)
    train_state_axes = jax.tree_map(lambda x: None, self.init_train_state)
    model_dir = self.create_tempdir().full_path

    mapfn = lambda i: {'i': [tf.cast(i, tf.int32)], 'j': [tf.cast(1, tf.int32)]}
    self.dataset = tf.data.Dataset.range(6).map(mapfn).batch(
        2, drop_remainder=True)

    num_experts = 10
    self.test_trainer = trainer_lib.MoeTrainer(
        model=mock.create_autospec(models_lib.BaseModel, instance=True),
        train_state=self.init_train_state,
        partitioner=partitioning.MoePjitPartitioner(
            num_experts=num_experts, num_partitions=1),
        eval_names=['task1', 'task2'],
        summary_dir=model_dir,
        train_state_axes=train_state_axes,
        rng=np.ones(2, np.uint32),
        learning_rate_fn=lambda step: 2 * step,
        num_microbatches=None,
        num_experts=num_experts)

  @mock.patch('time.time')
  @mock.patch('t5x.trainer.accumulate_grads_microbatched', fake_accum_grads)
  @mock.patch('t5x.trainer.apply_grads', fake_apply_grads)
  @mock.patch('absl.logging.log', lambda *_: None)  # avoids time.time() calls
  def _test_train(self, precompile, mock_time=None):
    trainer = self.test_trainer
    initial_rng = trainer._base_rng

    if precompile:
      mock_time.side_effect = [0, 1]
      trainer.compile_train(next(self.dataset.as_numpy_iterator()))
      trainer._compiled_train_step = mock.Mock(
          side_effect=trainer._compiled_train_step)

    trainer._partitioned_train_step = mock.Mock(
        side_effect=trainer._partitioned_train_step)

    # train start, logging, train end, logging
    mock_time.side_effect = [1, 5]
    num_steps = 2
    trainer.train(self.dataset.as_numpy_iterator(), num_steps)

    # Base rng must remain the same.
    np.testing.assert_array_equal(trainer._base_rng, initial_rng)

    expected_optimizer = optimizers.Optimizer(
        self.init_optimizer.optimizer_def,
        state=optimizers.OptimizerState(
            step=[6],
            param_states={
                'expert_bias': 60,  # 10 * (0+1+2+3) = 60
                'kernel': 6  # 0+1+2+3 = 6
            }),
        target={
            'expert_bias': 60 * np.ones(4),
            'kernel': 6 * np.ones((2, 4))
        })
    expected_train_state = train_state_lib.FlaxOptimTrainState(
        expected_optimizer)
    jax.tree_map(np.testing.assert_allclose, trainer.train_state,
                 expected_train_state)

    if precompile:
      self.assertEqual(trainer._compiled_train_step.call_count, num_steps)
      trainer._partitioned_train_step.assert_not_called()
    else:
      self.assertIsNone(trainer._compiled_train_step)
      self.assertEqual(trainer._partitioned_train_step.call_count, num_steps)

  def test_train_noprecompile(self):
    self._test_train(False)

  def test_train_precompile(self):
    self._test_train(True)


if __name__ == '__main__':
  absltest.main()
