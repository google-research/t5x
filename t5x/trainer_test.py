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

"""Tests for t5x.trainer_lib."""
import collections
import contextlib
import os

from absl.testing import absltest
from absl.testing import parameterized
import chex
from clu import metric_writers
import clu.metrics
import clu.values
import flax
import jax
from jax._src import dispatch as jax_dispatch
import jax.numpy as jnp
import numpy as np
from t5x import metrics as metrics_lib
from t5x import models as models_lib
from t5x import optimizers
from t5x import partitioning
from t5x import test_utils
from t5x import train_state as train_state_lib
from t5x import trainer as trainer_lib
import tensorflow as tf
from tensorflow.io import gfile

mock = absltest.mock
jax.config.parse_flags_with_absl()

FlaxMutables = flax.core.FrozenDict


# Make `log_elapsed_time` a no-op to simplify mocking of `time.time()`.
@contextlib.contextmanager
def fake_log_elapsed_time(_, event=None):  # pylint: disable=unused-argument
  yield


jax_dispatch.log_elapsed_time = fake_log_elapsed_time


def _validate_events(test_case, summary_dir, expected_metrics, steps):
  summaries = gfile.listdir(summary_dir)
  test_case.assertLen(summaries, 1)
  summary_path = os.path.join(summary_dir, summaries[0])
  event_file = os.path.join(summary_path)
  events = list(tf.compat.v1.train.summary_iterator(event_file))
  actual_events = {}
  # First event is boilerplate
  test_case.assertLen(events, len(steps) + 1)
  for step, event in zip(steps, events[1:]):
    test_case.assertEqual(event.step, step)
    test_case.assertLen(event.summary.value, 1)
    tensor = event.summary.value[0].tensor
    if tensor.string_val:
      actual_events[event.summary.value[0].tag] = tensor.string_val[0].decode()
    else:
      actual_events[event.summary.value[0].tag] = float(tf.make_ndarray(tensor))

  jax.tree_map(test_case.assertAlmostEqual, actual_events, expected_metrics)


class MetricsManagerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model_dir = self.create_tempdir().full_path

  def test_summary_dir(self):
    # All hosts have the summary dir.
    with mock.patch('jax.process_index', return_value=0):
      mm = trainer_lib.MetricsManager('eval', self.model_dir)
    self.assertEqual(mm.summary_dir, os.path.join(self.model_dir, 'eval'))
    mm.close()

    with mock.patch('jax.process_index', return_value=1):
      mm = trainer_lib.MetricsManager('eval', self.model_dir)
    self.assertEqual(mm.summary_dir, os.path.join(self.model_dir, 'eval'))
    mm.close()

  def test_summary_writer(self):
    # Only host 0 creates a non-empty summary writer.
    with mock.patch('jax.process_index', return_value=1):
      mm = trainer_lib.MetricsManager('eval', self.model_dir)
    self.assertFalse(gfile.exists(mm.summary_dir))
    mm.close()

    with mock.patch('jax.process_index', return_value=0):
      mm = trainer_lib.MetricsManager('eval', self.model_dir)
    self.assertIsInstance(mm.summary_writer, metric_writers.MetricWriter)
    self.assertTrue(gfile.exists(mm.summary_dir))
    mm.close()

  def test_write_scalar(self):
    gfile.makedirs(os.path.join(self.model_dir, 'eval'))

    # tag, value, step
    scalars = [('loss', 1.0, 1), ('accuracy', 100.0, 2)]

    # Only host 0 has actually writes summaries.
    with mock.patch('jax.process_index', return_value=1):
      mm = trainer_lib.MetricsManager('eval', self.model_dir)
      for s in scalars:
        mm.write_scalar(*s)
    self.assertEmpty(gfile.listdir(mm.summary_dir))
    mm.close()

    with mock.patch('jax.process_index', return_value=0):
      mm = trainer_lib.MetricsManager('eval', self.model_dir)
      for s in scalars:
        mm.write_scalar(*s)
    mm.flush()

    summaries = gfile.listdir(mm.summary_dir)
    self.assertLen(summaries, 1)

    event_file = os.path.join(mm.summary_dir, summaries[0])
    events = list(tf.compat.v1.train.summary_iterator(event_file))
    # First event is boilerplate
    self.assertLen(events, 3)
    for event, (tag, value, step) in zip(events[1:], scalars):
      self.assertEqual(event.step, step)
      self.assertLen(event.summary.value, 1)
      self.assertEqual(event.summary.value[0].tag, tag)
      self.assertEqual(tf.make_ndarray(event.summary.value[0].tensor), value)
    mm.close()

  def test_write_metrics_summary(self):
    gfile.makedirs(os.path.join(self.model_dir, 'eval'))

    @flax.struct.dataclass
    class MockTextMetric(clu.metrics.Metric):

      def compute_value(self):
        return clu.values.Text('test metric')

    accumulated_metrics = {
        'loss': metrics_lib.Sum(40.0),
        'accuracy': metrics_lib.AveragePerStep.from_model_output(20.0),
        'steps_per_second': metrics_lib.StepsPerTime(),
        'text': MockTextMetric()
    }
    expected_values = {
        'loss': clu.values.Scalar(40.0),
        'accuracy': clu.values.Scalar(10.0),
        'steps_per_second': clu.values.Scalar(0.05),
        'text': clu.values.Text('test metric')
    }
    with mock.patch(
        'jax.process_index', return_value=0), mock.patch(
            'time.time',
            side_effect=[0, 40]  # start_time, end_time
        ), mock.patch('absl.logging.log'):  # avoids hidden calls to time.time()
      mm = trainer_lib.MetricsManager('eval', summary_dir=self.model_dir)
      mm.start_duration_timer()
      summary = mm.write_metrics_summary(
          accumulated_metrics, step=4, num_steps=2)
      mm.flush()

    self.assertDictEqual(summary.result(), expected_values)
    _validate_events(
        self,
        mm.summary_dir, {k: v.value for k, v in expected_values.items()},
        steps=[4, 4, 4, 4])

    mm.close()

  def test_timer_blocking_on_donated_buffer(self):
    mm = trainer_lib.MetricsManager('train', summary_dir=None)
    x = jnp.zeros(1)

    # Not deleted.
    mm.start_duration_timer(block_on=x)
    mm._duration_timer._start_future.result()

    # Deleted/donated.
    x.device_buffer.delete()
    mm.start_duration_timer(block_on=x)
    mm._duration_timer._start_future.result()

  def test_timer_concurrency(self):
    mm = trainer_lib.MetricsManager('train')

    n = 10
    with mock.patch(
        'time.time',
        side_effect=range(2 * n)  # start_time, end_time
    ), mock.patch('absl.logging.log'):  # avoids hidden calls to time.time()
      for _ in range(n):
        mm.start_duration_timer()
        summary = mm.write_metrics_summary({'time': metrics_lib.Time()}, 0, 1)
        self.assertEqual(1, summary.result()['time'].value)
      mm.flush()


def fake_accum_grads(model, optimizer, batch, rng, num_microbatches,
                     data_partition_spec):
  del model, num_microbatches, rng, data_partition_spec
  # Add `i` to each optimzer value.
  i = batch['i'].sum()
  grad_accum = jax.tree_map(lambda x: i, optimizer)
  # Add j to each metric.
  j = batch['j'].sum()
  metrics = {'loss': metrics_lib.Sum(j), 'accuracy': metrics_lib.Sum(j)}
  return grad_accum, metrics, None


def fake_apply_grads(optimizer,
                     grad_accum,
                     metrics,
                     learning_rate,
                     weight_metrics_computer,
                     other_state_variables=None):
  del weight_metrics_computer
  del other_state_variables
  metrics['learning_rate'] = clu.metrics.Average(learning_rate, count=1)
  optimizer = jax.tree_map(lambda x, g: x + g, optimizer, grad_accum)
  return optimizer, metrics


def fake_eval_step(model, optimizer, batch):
  del model, optimizer
  # Add `i` to each metric.
  i = batch['i'].sum()

  return {'loss': metrics_lib.Sum(i), 'accuracy': metrics_lib.Sum(i)}


def fake_eval_fn_without_weight_sum(params, batch):
  del params
  # Add `i` to each metric.
  i = batch['i'].sum()

  loss = metrics_lib.Sum(i)
  return loss, {'loss': loss, 'accuracy': metrics_lib.Sum(i)}


def build_fake_grad_fn_without_weight_sum(has_aux, require_flax_mutables):
  def fake_grad_fn_without_weight_sum(train_state_params,
                                      batch,
                                      dropout_rng,
                                      flax_mutables=None):
    del dropout_rng, train_state_params
    # Add `i` to each optimzer value.
    i = batch['i'].sum()
    optimizer = optimizers.Optimizer(
        optimizers.sgd(0.1),
        state=optimizers.OptimizerState(
            step=0, param_states={
                'bias': 0,
                'kernel': 0
            }),
        target={
            'bias': np.zeros(4),
            'kernel': np.zeros((2, 4))
        })
    train_state = train_state_lib.FlaxOptimTrainState(optimizer)
    grad_accum = jax.tree_map(lambda x: i, train_state)
    # Add j to each metric.
    j = batch['j'].sum()
    metrics = {'loss': metrics_lib.Sum(j), 'accuracy': metrics_lib.Sum(j)}

    if require_flax_mutables or flax_mutables is not None:
      aux = metrics, flax_mutables
    else:
      aux = metrics

    if has_aux:
      return (None, aux), grad_accum.params
    else:
      return None, grad_accum.params

  return fake_grad_fn_without_weight_sum


def fake_value_and_grad_fn_without_weight_sum(callable_fn, has_aux=False):
  del callable_fn
  return build_fake_grad_fn_without_weight_sum(has_aux, False)


def fake_value_and_grad_fn_wo_weight_sum_w_mutables(callable_fn, has_aux=False):
  del callable_fn
  return build_fake_grad_fn_without_weight_sum(has_aux, True)


class TrainerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.init_optimizer = optimizers.Optimizer(
        optimizers.sgd(0.1),
        state=optimizers.OptimizerState(
            step=0, param_states={
                'bias': 0,
                'kernel': 0
            }),
        target={
            'bias': np.zeros(4),
            'kernel': np.zeros((2, 4))
        })
    self.init_train_state = train_state_lib.FlaxOptimTrainState(
        self.init_optimizer)
    train_state_axes = jax.tree_map(lambda x: None, self.init_train_state)
    model_dir = self.create_tempdir().full_path

    mapfn = lambda i: {'i': [tf.cast(i, tf.int32)], 'j': [tf.cast(1, tf.int32)]}
    self.dataset = tf.data.Dataset.range(6).map(mapfn).batch(
        2, drop_remainder=True)

    with mock.patch(
        'time.time',
        side_effect=[0]  # trainer init
    ), mock.patch('absl.logging.log'):  # avoids hidden calls to time.time()
      self.test_trainer = trainer_lib.Trainer(
          mock.create_autospec(models_lib.BaseModel, instance=True),
          self.init_train_state,
          partitioning.PjitPartitioner(num_partitions=1),
          eval_names=['task1', 'task2'],
          summary_dir=model_dir,
          train_state_axes=train_state_axes,
          rng=np.ones(2, np.uint32),
          learning_rate_fn=lambda step: 2 * step,
          num_microbatches=None)

  def tearDown(self) -> None:
    self.test_trainer.close()
    return super().tearDown()

  @mock.patch('t5x.trainer.accumulate_grads_microbatched', fake_accum_grads)
  @mock.patch('t5x.trainer.apply_grads', fake_apply_grads)
  def _test_train(self, precompile):
    trainer = self.test_trainer
    initial_rng = trainer._base_rng

    if precompile:
      with mock.patch(
          'time.time',
          side_effect=[0, 1]  # compile start, end
      ), mock.patch('absl.logging.log'):  # avoids hidden calls to time.time()
        trainer.compile_train(next(self.dataset.as_numpy_iterator()))
      trainer._compiled_train_step = mock.Mock(
          side_effect=trainer._compiled_train_step)

    trainer._partitioned_train_step = mock.Mock(
        side_effect=trainer._partitioned_train_step)

    num_steps = 2
    with mock.patch(
        'time.time',
        side_effect=[1, 5, 6]  # start_time, uptime logged, end_time
    ), mock.patch('absl.logging.log'):  # avoids hidden calls to time.time()
      trainer.train(self.dataset.as_numpy_iterator(), num_steps).result()

    initial_metrics = {
        'loss': 0.,
        'accuracy': 0.,
    }
    expected_metrics = {
        k: (v + 2 * num_steps) for k, v in initial_metrics.items()
    }
    # (0 + 2) / 2 = 1
    expected_metrics['learning_rate'] = 1
    # 5.0 - 0.0
    expected_metrics['timing/uptime'] = 5.0
    # 0+1+2+3 = 6
    expected_train_state = jax.tree_map(lambda x: np.array(x + 6),
                                        self.init_train_state)

    # Base rng must remain the same
    np.testing.assert_array_equal(trainer._base_rng, initial_rng)
    jax.tree_map(np.testing.assert_equal, trainer.train_state,
                 expected_train_state)
    # Expected step is 6 since we increment it along with the other optimizer
    # values.
    steps = [2, 2, 2, 2]
    if precompile:
      steps = [0] + steps
      expected_metrics['timing/compilation_seconds'] = 1
      self.assertEqual(trainer._compiled_train_step.call_count, num_steps)
      trainer._partitioned_train_step.assert_not_called()
    else:
      self.assertIsNone(trainer._compiled_train_step)
      self.assertEqual(trainer._partitioned_train_step.call_count, num_steps)
    trainer.train_metrics_manager.flush()
    _validate_events(
        self,
        trainer.train_metrics_manager.summary_dir,
        expected_metrics,
        steps=steps)

  def test_train_noprecompile(self):
    self._test_train(False)

  def test_train_precompile(self):
    self._test_train(True)

  @mock.patch('t5x.trainer.eval_step', fake_eval_step)
  def _test_eval(self, precompile):
    trainer = self.test_trainer
    initial_rng = trainer._base_rng

    task_datasets = {
        'task1': self.dataset.take(2),
        'task2': self.dataset.repeat().take(5)
    }

    if precompile:
      # [task1 start, task1 end, task2 start, task2 end]
      with mock.patch(
          'time.time',
          side_effect=[0, 1, 2, 3]  # [t1 start, t1 end, t2 start, t2 end]
      ), mock.patch('absl.logging.log'):  # avoids hidden calls to time.time()
        trainer.compile_eval({
            task: next(ds.as_numpy_iterator())
            for task, ds in task_datasets.items()
        })
      trainer._compiled_eval_steps = {
          task: mock.Mock(side_effect=trainer._compiled_eval_steps[task])
          for task in task_datasets
      }

    trainer._partitioned_eval_step = mock.Mock(
        side_effect=trainer._partitioned_eval_step)

    with mock.patch(
        'time.time',
        side_effect=[1, 5, 5, 8]  # t1 start, t1 end, t2 start, t2 end]
    ), mock.patch('absl.logging.log'):  # avoids hidden calls to time.time()
      trainer.eval(
          {task: ds.as_numpy_iterator() for task, ds in task_datasets.items()})

    all_expected_metrics = {
        # 0+1+2+3 = 6
        'task1': {
            'loss': 6,
            'accuracy': 6,
        },
        # 0+1+2+3+4+5+0+1+2+3 = 21
        'task2': {
            'loss': 21,
            'accuracy': 21,
        },
    }

    np.testing.assert_array_equal(trainer._base_rng, initial_rng)
    for task_name, expected_metrics in all_expected_metrics.items():
      steps = [0, 0]
      if precompile:
        steps = [0] + steps
        expected_metrics['timing/compilation_seconds'] = 1
        self.assertEqual(  # pylint:disable=g-generic-assert
            trainer._compiled_eval_steps[task_name].call_count,
            len(task_datasets[task_name]))
        trainer._partitioned_eval_step.assert_not_called()
      else:
        self.assertEmpty(trainer._compiled_eval_steps)
        self.assertEqual(trainer._partitioned_eval_step.call_count,
                         sum(len(ds) for ds in task_datasets.values()))
      mm = trainer.eval_metrics_managers[task_name]
      mm.flush()
      _validate_events(self, mm.summary_dir, expected_metrics, steps=steps)

  def test_eval_noprecompile(self):
    self._test_eval(False)

  def test_eval_precompile(self):
    self._test_eval(True)

  @parameterized.named_parameters([
      {
          'testcase_name': 'max_no_increase',
          'mode': 'max',
          'metrics': [1, 1, 1],
          'atol': 0.0,
          'rtol': 0.0,
          'stop_training': True,
      },
      {
          'testcase_name': 'max_no_atol',
          'mode': 'max',
          'metrics': [1, 0.9, 0.8],
          'atol': 0.0,
          'rtol': 0.0,
          'stop_training': True,
      },
      {
          'testcase_name': 'max_not_enough_atol',
          'mode': 'max',
          'metrics': [1, 1.09, 1.18],
          'atol': 0.1,
          'rtol': 0.0,
          'stop_training': True,
      },
      {
          'testcase_name': 'max_enough_atol',
          'mode': 'max',
          'metrics': [1, 1.2, 1.4],
          'atol': 0.1,
          'rtol': 0.0,
          'stop_training': False,
      },
      {
          'testcase_name': 'max_enough_atol_rtol',
          'mode': 'max',
          # first delta = 0.1 + 1* 0.08 = 0.18
          # second delta = 0.1 + 1.2 * 0.08 = 0.196
          'metrics': [1, 1.2, 1.4],
          'atol': 0.1,
          'rtol': 0.08,
          'stop_training': False,
      },
      {
          'testcase_name': 'max_not_enough_rtol',
          'mode': 'max',
          'metrics': [1, 1.2, 1.4],
          'atol': 0.0,
          'rtol': 0.2,
          'stop_training': True,
      },
      {
          'testcase_name': 'min_no_decrease',
          'mode': 'min',
          'metrics': [1, 1, 1],
          'atol': 0.0,
          'rtol': 0.0,
          'stop_training': True,
      },
      {
          'testcase_name': 'min_no_atol',
          'mode': 'min',
          'metrics': [1, 1, 1],
          'atol': 0.0,
          'rtol': 0.0,
          'stop_training': True,
      },
      {
          'testcase_name': 'min_not_enough_atol',
          'mode': 'min',
          'metrics': [1, 0.9, 0.71],
          'atol': 0.2,
          'rtol': 0.0,
          'stop_training': True,
      },
      {
          'testcase_name': 'min_enough_atol',
          'mode': 'min',
          'metrics': [1, 0.8, 0.6],
          'atol': 0.15,
          'rtol': 0.0,
          'stop_training': False,
      },
      {
          'testcase_name': 'min_enough_atol_rtol',
          'mode': 'min',
          # first delta = 0.1 + 1* 0.09 = 0.19
          # second delta = 0.1 + 0.8 * 0.09 = 0.172
          'metrics': [1, 0.8, 0.6],
          'atol': 0.1,
          'rtol': 0.09,
          'stop_training': False,
      },
      {
          'testcase_name': 'min_not_enough_rtol',
          'mode': 'min',
          'metrics': [1, 0.8, 0.6],
          'atol': 0.0,
          'rtol': 0.3,
          'stop_training': True,
      },
      {
          'testcase_name': 'longer_history',
          'mode': 'min',
          'metrics': [1, 0.8, 0.7, 0.6],
          'atol': 0.15,
          'rtol': 0.0,
          'stop_training': True,
      },
  ])
  def test_early_stopping_action(
      self, mode, metrics, atol, rtol, stop_training
  ):
    trainer = self.test_trainer
    metrics = [clu.values.Scalar(metric) for metric in metrics]
    hook = trainer_lib.EarlyStoppingAction(
        ('test_task', 'metric'), mode=mode, patience=3, atol=atol, rtol=rtol
    )

    for metric in metrics:
      trainer_stop_training = hook.run(
          trainer.train_state, {'test_task': {'metric': metric}}
      )

    self.assertEqual(trainer_stop_training, stop_training)

  @parameterized.named_parameters([
      {
          'testcase_name': 'allow_clu_scalar_early_stopping',
          'metrics': [
              clu.values.Scalar(1),
              clu.values.Scalar(0.9),
              clu.values.Scalar(0.71),
          ],
          'atol': 0.2,
          'stop_training': True,
      },
      {
          'testcase_name': 'allow_float_early_stopping',
          'metrics': [1.0, 0.9, 0.71],
          'atol': 0.2,
          'stop_training': True,
      },
      {
          'testcase_name': 'error_for_other_type',
          'metrics': [3, 2, 1],
          'atol': 1.1,
          'stop_training': False,
      },
  ])
  def test_early_stopping_action_value(self, metrics, atol, stop_training):
    trainer = self.test_trainer
    hook = trainer_lib.EarlyStoppingAction(
        ('test_task', 'metric'), mode='min', patience=3, atol=atol
    )

    for metric in metrics:
      trainer_stop_training = hook.run(
          trainer.train_state, {'test_task': {'metric': metric}}
      )

    self.assertEqual(trainer_stop_training, stop_training)

  @parameterized.named_parameters([
      {
          'testcase_name': 'invalid_task',
          'task': 'wrong_task',
          'metric': 'metric',
          'value': clu.values.Scalar(np.nan),
      },
      {
          'testcase_name': 'invalid_metric_name',
          'task': 'task',
          'metric': 'wrong_metric_name',
          'value': clu.values.Scalar(np.nan),
      },
  ])
  def test_early_stopping_action_error(self, task, metric, value):
    trainer = self.test_trainer
    hook = trainer_lib.EarlyStoppingAction((task, metric),
                                           mode='min',
                                           patience=5,
                                           atol=1,
                                           rtol=1)

    trainer_stop_training = hook.run(trainer.train_state,
                                     {task: {
                                         metric: value
                                     }})

    self.assertFalse(trainer_stop_training)

  @parameterized.named_parameters([{
      'testcase_name': 'valid_loss',
      'metric': 'loss',
      'value': 1.0,
      'stop_training': False,
  }, {
      'testcase_name': 'nan',
      'metric': 'loss',
      'value': np.nan,
      'stop_training': True,
  }, {
      'testcase_name': 'inf',
      'metric': 'loss',
      'value': np.inf,
      'stop_training': True,
  }, {
      'testcase_name': 'other_metric',
      'metric': 'some_metric',
      'value': np.inf,
      'stop_training': True,
  }])
  def test_terminate_on_nan_action(self, metric, value, stop_training):
    trainer = self.test_trainer
    value = clu.values.Scalar(value)
    hook = trainer_lib.TerminateOnNanAction(task='test_task', metric=metric)

    trainer_stop_training = hook.run(trainer.train_state,
                                     {'test_task': {
                                         metric: value
                                     }})

    self.assertEqual(trainer_stop_training, stop_training)

  @parameterized.named_parameters([
      {
          'testcase_name': 'invalid_task',
          'task': 'wrong_task',
          'metric': 'metric',
          'value': clu.values.Scalar(np.nan),
      },
      {
          'testcase_name': 'invalid_metric_name',
          'task': 'task',
          'metric': 'wrong_metric_name',
          'value': clu.values.Scalar(np.nan),
      },
      {
          'testcase_name': 'invalid_value',
          'task': 'task',
          'metric': 'metric',
          'value': 1.0,
      },
  ])
  def test_terminate_on_nan_action_error(self, task, metric, value):
    trainer = self.test_trainer
    hook = trainer_lib.TerminateOnNanAction(task=task, metric=metric)

    trainer_stop_training = hook.run(trainer.train_state,
                                     {'task': {
                                         'metric': value
                                     }})

    self.assertFalse(trainer_stop_training)

  def test_compile_train(self):
    trainer = self.test_trainer
    trainer._partitioned_train_step = mock.Mock()
    trainer.train_metrics_manager = mock.Mock()

    batch = {
        'i': np.arange(10, dtype=np.int32).reshape((2, 5)),
        'j': np.ones((), dtype=np.float32)
    }
    # compile start, compile end
    with mock.patch('time.time', side_effect=[1, 5]):
      trainer.compile_train(batch)

    trainer.train_metrics_manager.write_scalar.assert_called_with(
        'timing/compilation_seconds', 4, trainer.train_state.step)
    trainer._partitioned_train_step.lower.assert_called_once()
    train_step_args = trainer._partitioned_train_step.lower.call_args[0]
    self.assertLen(train_step_args, 2)
    self.assertEqual(train_step_args[0], trainer.train_state)
    test_utils.assert_same(train_step_args[1], batch)

  def test_compile_eval(self):
    trainer = self.test_trainer
    trainer._partitioned_eval_step = mock.Mock()
    trainer.eval_metrics_managers = {
        'eval1': mock.Mock(),
        'eval2': mock.Mock(),
        'eval3': mock.Mock(),
        'eval4': mock.Mock()
    }
    trainer._partitioned_eval_step.lower().compile.side_effect = [
        'compiled1', 'compiled2', 'compiled3'
    ]

    batches = {
        'eval1': {
            'i': np.zeros((2, 5), dtype=np.int32)
        },
        'eval2': {
            'j': np.zeros((), dtype=np.float32)
        },
        'eval3': {
            'j': np.zeros((), dtype=np.float32)
        },
        'eval4': {
            'k': np.zeros((4), dtype=np.float32)
        },
    }

    # eval1 start/end, eval2 start/end, eval3 start/end, eval 4 start/end
    with mock.patch('time.time', side_effect=[1, 5, 6, 9, 10, 11, 12, 13]):
      trainer.compile_eval(collections.OrderedDict(sorted(batches.items())))

    trainer.eval_metrics_managers['eval1'].write_scalar.assert_called_with(
        'timing/compilation_seconds', 4, trainer.train_state.step)
    trainer.eval_metrics_managers['eval2'].write_scalar.assert_called_with(
        'timing/compilation_seconds', 3, trainer.train_state.step)
    trainer.eval_metrics_managers['eval3'].write_scalar.assert_called_with(
        'timing/compilation_seconds', 1, trainer.train_state.step)
    trainer.eval_metrics_managers['eval4'].write_scalar.assert_called_with(
        'timing/compilation_seconds', 1, trainer.train_state.step)
    eval_step_args = trainer._partitioned_eval_step.lower.call_args_list[1:]
    self.assertLen(eval_step_args, 3)

    eval1_call_args = eval_step_args[0][0]
    self.assertLen(eval1_call_args, 2)
    self.assertEqual(eval1_call_args[0], trainer.train_state)
    test_utils.assert_same(eval1_call_args[1], {
        'i': np.zeros((2, 5), dtype=np.int32),
    })

    eval2_call_args = eval_step_args[1][0]
    self.assertLen(eval2_call_args, 2)
    self.assertEqual(eval2_call_args[0], trainer.train_state)
    test_utils.assert_same(eval2_call_args[1], {
        'j': np.zeros((), dtype=np.float32),
    })

    eval3_call_args = eval_step_args[2][0]
    self.assertLen(eval3_call_args, 2)
    self.assertEqual(eval3_call_args[0], trainer.train_state)
    test_utils.assert_same(eval3_call_args[1], {
        'k': np.zeros((4), dtype=np.float32),
    })

    self.assertDictEqual(
        trainer._compiled_eval_steps, {
            'eval1': 'compiled1',
            'eval2': 'compiled2',
            'eval3': 'compiled2',
            'eval4': 'compiled3'
        })

  @mock.patch('jax.value_and_grad', fake_value_and_grad_fn_without_weight_sum)
  def test_accumulate_grads_microbatched_without_weight_sum_single_batch(self):
    batch_iter = self.dataset.as_numpy_iterator()
    batch = next(batch_iter)
    num_microbatches = 1
    grad_accum, metrics, flax_mutables = trainer_lib.accumulate_grads_microbatched(
        self.test_trainer._model, self.init_train_state, batch,
        self.test_trainer._base_rng, num_microbatches)

    i = batch['i'].sum()
    expected_grad_accum = jax.tree_map(lambda x: i,
                                       self.init_train_state).params
    self.assertEqual(expected_grad_accum, grad_accum)
    self.assertEqual(metrics['loss'].compute(), 2)
    self.assertEqual(metrics['accuracy'].compute(), 2)
    self.assertIsNone(flax_mutables)

  @mock.patch('jax.value_and_grad', fake_value_and_grad_fn_without_weight_sum)
  def test_accumulate_grads_microbatched_without_weight_sum_multiple_batches(
      self):
    batch_iter = self.dataset.as_numpy_iterator()
    batch = next(batch_iter)
    num_micro_batches = 2
    grad_accum, metrics, flax_mutables = trainer_lib.accumulate_grads_microbatched(
        self.test_trainer._model, self.init_train_state, batch,
        self.test_trainer._base_rng, num_micro_batches)

    expected_grad_accum = {'bias': jnp.ones(4), 'kernel': jnp.ones((2, 4))}
    chex.assert_trees_all_equal(expected_grad_accum, grad_accum)
    self.assertEqual(metrics['loss'].compute(), 2)
    self.assertEqual(metrics['accuracy'].compute(), 2)
    self.assertIsNone(flax_mutables)

  def test_eval_step_without_weight_sum(self):
    batch_iter = self.dataset.as_numpy_iterator()
    batch = next(batch_iter)
    self.test_trainer._model.eval_fn = fake_eval_fn_without_weight_sum
    metrics = trainer_lib.eval_step(self.test_trainer._model,
                                    self.init_train_state, batch)

    self.assertEqual(metrics['loss'].compute(), 1)
    self.assertEqual(metrics['accuracy'].compute(), 1)


class TrainerRngDeterminismTest(parameterized.TestCase):

  def create_trainer(self, step, random_seed):
    init_optimizer = optimizers.Optimizer(
        optimizers.sgd(0.1),
        state=optimizers.OptimizerState(
            step=step, param_states={
                'bias': 0,
                'kernel': 0
            }),
        target={
            'bias': np.zeros(4),
            'kernel': np.zeros((2, 4))
        })
    init_train_state = train_state_lib.FlaxOptimTrainState(init_optimizer)
    train_state_axes = jax.tree_map(lambda x: None, init_train_state)

    test_trainer = trainer_lib.Trainer(
        mock.create_autospec(models_lib.BaseModel, instance=True),
        init_train_state,
        partitioning.PjitPartitioner(num_partitions=1),
        eval_names=['task1', 'task2'],
        summary_dir=None,
        train_state_axes=train_state_axes,
        rng=jax.random.PRNGKey(random_seed),
        learning_rate_fn=lambda step: 2 * step,
        num_microbatches=None)
    return test_trainer

  @mock.patch('t5x.trainer.accumulate_grads_microbatched')
  @mock.patch('t5x.trainer.apply_grads', fake_apply_grads)
  def test_rng_determinism(self, mock_accum_grads):

    def fake_accum_grads_rng(model, optimizer, batch, rng, num_microbatches,
                             data_partition_spec):
      del model, batch, num_microbatches, data_partition_spec
      # Add 1, which will increment the step as a side effect.
      grad_accum = jax.tree_map(lambda x: 1, optimizer)
      m = {'rng': metrics_lib.Sum(jnp.sum(rng))}
      return grad_accum, m, None

    mock_accum_grads.side_effect = fake_accum_grads_rng
    # Create a trainer at a given step (53) with a given random seed (23),
    # train up to a given train step (100), check the sum of the rngs from the
    # metrics.
    start_step = 47
    end_step = 100
    random_seed = 23
    trainer = self.create_trainer(step=start_step, random_seed=random_seed)
    # 500 batches of size 2
    ds = [np.zeros(2)] * 500

    metrics = trainer.train(iter(ds), num_steps=end_step - start_step)
    base_rng = jax.random.PRNGKey(random_seed)
    expected_rng_sum = np.sum(
        [jax.random.fold_in(base_rng, i) for i in range(start_step, end_step)],
        dtype=np.uint32)
    np.testing.assert_array_equal(metrics.result()['rng'].value,
                                  expected_rng_sum)


def fake_mut_accum_grads(model, optimizer, batch, rng, num_microbatches,
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
  return grad_accum, metrics, {'mutables': 0}


def fake_mut_apply_grads(optimizer, grad_accum, metrics, learning_rate,
                         weight_metrics_computer, other_state_variables):
  del weight_metrics_computer, other_state_variables
  metrics['learning_rate'] = clu.metrics.Average.from_model_output(
      learning_rate)
  optimizer = jax.tree_map(lambda x, g: x + g, optimizer, grad_accum)
  return optimizer, metrics


class MutableTrainerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.init_optimizer = optimizers.Optimizer(
        optimizers.sgd(0.1),
        state=optimizers.OptimizerState(
            step=0, param_states={
                'bias': 0,
                'kernel': 0
            }),
        target={
            'bias': np.zeros(4),
            'kernel': np.zeros((2, 4))
        })
    self.init_train_state = train_state_lib.FlaxOptimTrainState(
        _optimizer=self.init_optimizer,
        flax_mutables=FlaxMutables(variables={
            'keys': np.zeros((10, 2)),
            'values': np.zeros((10, 5)),
        }))
    train_state_axes = jax.tree_map(lambda x: None, self.init_train_state)
    model_dir = self.create_tempdir().full_path

    mapfn = lambda i: {'i': [tf.cast(i, tf.int32)], 'j': [tf.cast(1, tf.int32)]}
    self.dataset = tf.data.Dataset.range(6).map(mapfn).batch(
        2, drop_remainder=True)

    self.test_trainer = trainer_lib.Trainer(
        mock.create_autospec(models_lib.BaseModel, instance=True),
        self.init_train_state,
        partitioning.PjitPartitioner(num_partitions=1),
        eval_names=['task1', 'task2'],
        summary_dir=model_dir,
        train_state_axes=train_state_axes,
        rng=np.ones(2, np.uint32),
        learning_rate_fn=lambda step: 2 * (step + 1),
        num_microbatches=None)

  @mock.patch('time.time')
  @mock.patch('t5x.trainer.accumulate_grads_microbatched', fake_mut_accum_grads)
  @mock.patch('t5x.trainer.apply_grads', fake_mut_apply_grads)
  # avoids calls time.time() during logging
  @mock.patch('absl.logging.info', lambda *_: None)
  @mock.patch('absl.logging.log_every_n_seconds', lambda *_: None)
  def test_train(self, mock_time=None):
    trainer = self.test_trainer
    initial_rng = trainer._base_rng

    trainer._partitioned_train_step = mock.Mock(
        side_effect=trainer._partitioned_train_step)

    # train start, logging, train end, logging
    mock_time.side_effect = [1, 5, 5, 5]
    num_steps = 1
    ds_iter = self.dataset.as_numpy_iterator()
    batch = next(ds_iter)
    train_state, _ = trainer._partitioned_train_step(trainer.train_state, batch)

    expected_train_state = jax.tree_map(lambda x: np.array(x + 1),
                                        self.init_train_state)
    # Base rng must remain the same
    np.testing.assert_array_equal(trainer._base_rng, initial_rng)
    jax.tree_map(np.testing.assert_equal, train_state, expected_train_state)

    self.assertIsNone(trainer._compiled_train_step)
    self.assertEqual(trainer._partitioned_train_step.call_count, num_steps)

  @mock.patch('jax.value_and_grad', fake_value_and_grad_fn_without_weight_sum)
  def test_accumulate_grads_microbatched_without_weight_sum_single_batch(self):
    batch_iter = self.dataset.as_numpy_iterator()
    batch = next(batch_iter)
    num_microbatches = 1
    grad_accum, metrics, flax_mutables = trainer_lib.accumulate_grads_microbatched(
        self.test_trainer._model, self.init_train_state, batch,
        self.test_trainer._base_rng, num_microbatches)

    i = batch['i'].sum()
    expected_grad_accum = jax.tree_map(lambda x: i,
                                       self.init_train_state).params
    self.assertEqual(expected_grad_accum, grad_accum)
    self.assertEqual(metrics['loss'].compute(), 2)
    self.assertEqual(metrics['accuracy'].compute(), 2)
    self.assertIsNotNone(flax_mutables)

  @mock.patch(
      'jax.value_and_grad', fake_value_and_grad_fn_wo_weight_sum_w_mutables
  )
  def test_accumulate_grads_microbatched_without_weight_sum_multiple_batches(
      self):
    batch_iter = self.dataset.as_numpy_iterator()
    batch = next(batch_iter)
    num_micro_batches = 2
    grad_accum, metrics, flax_mutables = trainer_lib.accumulate_grads_microbatched(
        self.test_trainer._model, self.init_train_state, batch,
        self.test_trainer._base_rng, num_micro_batches)

    expected_grad_accum = {'bias': jnp.ones(4), 'kernel': jnp.ones((2, 4))}
    chex.assert_trees_all_equal(expected_grad_accum, grad_accum)
    self.assertEqual(metrics['loss'].compute(), 2)
    self.assertEqual(metrics['accuracy'].compute(), 2)
    self.assertIsNotNone(flax_mutables)

  def tearDown(self) -> None:
    # Manually close managers to avoid phantom threads crossing test cases.
    self.test_trainer.train_metrics_manager.close()
    for mm in self.test_trainer.eval_metrics_managers.values():
      mm.close()
    return super().tearDown()


if __name__ == '__main__':
  absltest.main()
