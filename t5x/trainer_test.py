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

"""Tests for t5x.trainer_lib."""
import collections
import os

from absl.testing import absltest
from absl.testing import parameterized
from clu import metric_writers
from flax import optim
import jax
import numpy as np
from t5x import partitioning
from t5x import test_utils
from t5x import train_state as train_state_lib
from t5x import trainer as trainer_lib
import tensorflow as tf
from tensorflow.io import gfile

mock = absltest.mock
jax.config.parse_flags_with_absl()


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
    actual_events[event.summary.value[0].tag] = float(
        tf.make_ndarray(event.summary.value[0].tensor))
  jax.tree_multimap(test_case.assertAlmostEqual, actual_events,
                    expected_metrics)


class MetricsManagerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.accumulator = {'loss': 0, 'accuracy': 1}
    self.model_dir = self.create_tempdir().full_path

  def test_initial_accumulator(self):
    mm = trainer_lib.MetricsManager('eval', self.accumulator, lambda x, y: x,
                                    self.model_dir)
    out_accumulator = mm.initial_accumulator
    for v in out_accumulator.values():
      self.assertIsInstance(v, np.ndarray)
    self.assertDictEqual(out_accumulator, self.accumulator)
    out_accumulator['loss'] = 1
    self.assertDictEqual(mm.initial_accumulator, self.accumulator)

  @mock.patch('jax.process_index')
  def test_summary_dir(self, mock_process_index):
    # All hosts have the summary dir.
    mock_process_index.return_value = 0
    mm = trainer_lib.MetricsManager('eval', {}, lambda x, y: x, self.model_dir)
    self.assertEqual(mm.summary_dir, os.path.join(self.model_dir, 'eval'))

    mock_process_index.return_value = 1
    mm = trainer_lib.MetricsManager('eval', {}, lambda x, y: x, self.model_dir)
    self.assertEqual(mm.summary_dir, os.path.join(self.model_dir, 'eval'))

  @mock.patch('jax.process_index')
  def test_summary_writer(self, mock_process_index):
    # Only host 0 has a summary writer.
    mock_process_index.return_value = 1
    mm = trainer_lib.MetricsManager('eval', {}, lambda x, y: x, self.model_dir)
    self.assertIsNone(mm.summary_writer)
    self.assertFalse(gfile.exists(mm.summary_dir))

    mock_process_index.return_value = 0
    mm = trainer_lib.MetricsManager('eval', {}, lambda x, y: x, self.model_dir)
    self.assertIsInstance(mm.summary_writer, metric_writers.SummaryWriter)
    self.assertTrue(gfile.exists(mm.summary_dir))

  @mock.patch('jax.process_index')
  def test_write_scalar(self, mock_process_index):
    gfile.makedirs(os.path.join(self.model_dir, 'eval'))

    # tag, value, step
    scalars = [('loss', 1.0, 1), ('accuracy', 100.0, 2)]

    # Only host 0 has a summary writer.
    mock_process_index.return_value = 1
    mm = trainer_lib.MetricsManager('eval', {}, lambda x, y: x, self.model_dir)
    for s in scalars:
      mm.write_scalar(*s)
    self.assertEmpty(gfile.listdir(mm.summary_dir))

    mock_process_index.return_value = 0
    mm = trainer_lib.MetricsManager('eval', {}, lambda x, y: x, self.model_dir)
    for s in scalars:
      mm.write_scalar(*s)
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

  @mock.patch('jax.process_index')
  def test_write_metrics_summary(self, mock_process_index):
    gfile.makedirs(os.path.join(self.model_dir, 'eval'))

    def summarize_fn(metrics, duration, num_steps):
      return {
          'loss': metrics['loss'] / num_steps,
          'accuracy': metrics['accuracy'] / num_steps,
          'batches_per_second': num_steps / duration
      }

    accumulated_metrics = {'loss': 40.0, 'accuracy': 198.0}
    expected_events = {
        'loss': 20.0,
        'accuracy': 99.0,
        'batches_per_second': 0.05
    }

    # Only host 0 has a summary writer.
    mock_process_index.return_value = 1
    mm = trainer_lib.MetricsManager('eval', self.accumulator, summarize_fn,
                                    self.model_dir)
    mm.write_metrics_summary(
        accumulated_metrics, step=4, duration=40.0, num_steps=2)
    self.assertEmpty(gfile.listdir(mm.summary_dir))

    mock_process_index.return_value = 0
    mm = trainer_lib.MetricsManager('eval', self.accumulator, summarize_fn,
                                    self.model_dir)
    mm.write_metrics_summary(
        accumulated_metrics, step=4, duration=40.0, num_steps=2)

    _validate_events(self, mm.summary_dir, expected_events, steps=[4, 4, 4])


def fake_accum_grads(model, optimizer, batch, rng, num_microbatches):
  del model, num_microbatches, rng
  # Add `i` to each optimzer value.
  i = batch['i'].sum()
  grad_accum = jax.tree_map(lambda x: i, optimizer)
  # Add j to each metric.
  j = batch['j'].sum()
  metrics = {'loss': j, 'accuracy': j}
  return grad_accum, metrics


def fake_apply_grads(optimizer, grad_accum, metrics, learning_rate,
                     log_weight_metrics):
  del log_weight_metrics
  metrics['learning_rate'] = learning_rate
  optimizer = jax.tree_multimap(lambda x, g: x + g, optimizer, grad_accum)
  return optimizer, metrics


def fake_eval_step(model, optimizer, batch, metrics):
  del model, optimizer
  # Add `i` to each metric.
  i = batch['i'].sum()
  metrics = jax.tree_map(lambda x: x + i, metrics)
  return metrics


class TrainerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.init_optimizer = optim.Optimizer(
        optim.GradientDescent(),
        state=optim.OptimizerState(
            step=0, param_states={
                'bias': 0,
                'kernel': 0
            }),
        target={
            'bias': np.zeros(4),
            'kernel': np.zeros((2, 4))
        })
    self.init_train_state = train_state_lib.TrainState.from_flax_optimizer(
        self.init_optimizer)
    train_state_axes = jax.tree_map(lambda x: None, self.init_train_state)
    model_dir = self.create_tempdir().full_path

    mapfn = lambda i: {'i': [tf.cast(i, tf.int32)], 'j': [tf.cast(1, tf.int32)]}
    self.dataset = tf.data.Dataset.range(6).map(mapfn).batch(
        2, drop_remainder=True)

    self.test_trainer = trainer_lib.Trainer(
        mock.Mock(
            get_initial_metrics=lambda:  # pylint:disable=g-long-lambda
            {
                'loss': 1.0,
                'accuracy': 2.0,
            },
            summarize_metrics_fn=lambda metrics, duration, num_steps: jax.  # pylint:disable=g-long-lambda
            tree_map(lambda x: x / duration, metrics)),
        self.init_train_state,
        partitioning.ModelBasedPjitPartitioner(num_partitions=1),
        eval_names=['task1', 'task2'],
        summary_dir=model_dir,
        train_state_axes=train_state_axes,
        rng=np.ones(2, np.uint32),
        learning_rate_fn=lambda step: 2 * step,
        num_microbatches=None)

  @mock.patch('time.time')
  @mock.patch('t5x.trainer.accumulate_grads_microbatched', fake_accum_grads)
  @mock.patch('t5x.trainer.apply_grads', fake_apply_grads)
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
    mock_time.side_effect = [1, 5, 5, 5]
    num_steps = 2
    trainer.train(self.dataset.as_numpy_iterator(), num_steps)

    expected_metrics = jax.tree_map(
        lambda x: (x + 2 * num_steps) / 4,  # divide by duration
        trainer.train_metrics_manager.initial_accumulator)
    # (0 + 2) / 2 = 1
    expected_metrics['learning_rate'] = 1
    # 0+1+2+3 = 6
    expected_train_state = jax.tree_map(lambda x: np.array(x + 6),
                                        self.init_train_state)

    # Base rng must remain the same
    np.testing.assert_array_equal(trainer._base_rng, initial_rng)
    jax.tree_multimap(np.testing.assert_equal, trainer.train_state,
                      expected_train_state)
    # Expected step is 6 since we increment it along with the other optimizer
    # values.
    steps = [6, 6, 6]
    if precompile:
      steps = [0] + steps
      expected_metrics['timing/compilation_seconds'] = 1
      self.assertEqual(trainer._compiled_train_step.call_count, num_steps)
      trainer._partitioned_train_step.assert_not_called()
    else:
      self.assertIsNone(trainer._compiled_train_step)
      self.assertEqual(trainer._partitioned_train_step.call_count, num_steps)
    _validate_events(
        self,
        trainer.train_metrics_manager.summary_dir,
        expected_metrics,
        steps=steps)

  def test_train_noprecompile(self):
    self._test_train(False)

  def test_train_precompile(self):
    self._test_train(True)

  @mock.patch('time.time')
  @mock.patch('t5x.trainer.eval_step', fake_eval_step)
  def _test_eval(self, precompile, mock_time=None):
    trainer = self.test_trainer
    initial_rng = trainer._base_rng

    task_datasets = {
        'task1': self.dataset.take(2),
        'task2': self.dataset.take(1)
    }

    if precompile:
      mock_time.side_effect = [0, 1, 2, 3]
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

    # [task1 start, logging, task1 end, logging,
    #  task2 start, logging, task2 end, logging]
    mock_time.side_effect = [1, 1, 5, 5, 1, 1, 5, 5]
    trainer.eval(
        {task: ds.as_numpy_iterator() for task, ds in task_datasets.items()})

    all_expected_metrics = {
        # 0+1+2+3 = 6
        'task1':
            jax.tree_map(
                lambda x: (x + 6) / 4,  # divide by duration
                trainer.eval_metrics_managers['task1'].initial_accumulator),
        # 0+1 = 1
        'task2':
            jax.tree_map(
                lambda x: (x + 1) / 4,  # divide by duration
                trainer.eval_metrics_managers['task2'].initial_accumulator),
    }

    np.testing.assert_array_equal(trainer._base_rng, initial_rng)
    for task_name, expected_metrics in all_expected_metrics.items():
      # Expected step is 0 for each metric ssince it comes from the optimizer.
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

      _validate_events(
          self,
          trainer.eval_metrics_managers[task_name].summary_dir,
          expected_metrics,
          steps=steps)

  def test_eval_noprecompile(self):
    self._test_eval(False)

  def test_eval_precompile(self):
    self._test_eval(True)

  @parameterized.named_parameters([
      {
          'testcase_name': 'max_no_increase',
          'mode': 'max',
          'metrics': [1, 1, 1],
          'atol': 0.,
          'rtol': 0.,
          'stop_training': True,
      },
      {
          'testcase_name': 'max_no_atol',
          'mode': 'max',
          'metrics': [1, 0.9, 0.8],
          'atol': 0.,
          'rtol': 0.,
          'stop_training': True,
      },
      {
          'testcase_name': 'max_not_enough_atol',
          'mode': 'max',
          'metrics': [1, 1.09, 1.18],
          'atol': 0.1,
          'rtol': 0.,
          'stop_training': True,
      },
      {
          'testcase_name': 'max_enough_atol',
          'mode': 'max',
          'metrics': [1, 1.2, 1.4],
          'atol': 0.1,
          'rtol': 0.,
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
          'atol': 0.,
          'rtol': 0.2,
          'stop_training': True,
      },
      {
          'testcase_name': 'min_no_decrease',
          'mode': 'min',
          'metrics': [1, 1, 1],
          'atol': 0.,
          'rtol': 0.,
          'stop_training': True,
      },
      {
          'testcase_name': 'min_no_atol',
          'mode': 'min',
          'metrics': [1, 1, 1],
          'atol': 0.,
          'rtol': 0.,
          'stop_training': True,
      },
      {
          'testcase_name': 'min_not_enough_atol',
          'mode': 'min',
          'metrics': [1, 0.9, 0.71],
          'atol': 0.2,
          'rtol': 0.,
          'stop_training': True,
      },
      {
          'testcase_name': 'min_enough_atol',
          'mode': 'min',
          'metrics': [1, 0.8, 0.6],
          'atol': 0.15,
          'rtol': 0.,
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
          'rtol': 0.,
          'stop_training': True,
      }
  ])
  def test_early_stopping_action(self, mode, metrics, atol, rtol,
                                 stop_training):
    trainer = self.test_trainer
    hook = trainer_lib.EarlyStoppingAction(('test_task', 'metric'),
                                           mode=mode,
                                           patience=3,
                                           atol=atol,
                                           rtol=rtol)

    for metric in metrics:
      trainer_stop_training = hook.run(trainer.train_state,
                                       {'test_task': {
                                           'metric': metric
                                       }})

    self.assertEqual(trainer_stop_training, stop_training)

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
    hook = trainer_lib.TerminateOnNanAction(task='test_task', metric=metric)

    trainer_stop_training = hook.run(trainer.train_state,
                                     {'test_task': {
                                         metric: value
                                     }})

    self.assertEqual(trainer_stop_training, stop_training)

  @mock.patch('time.time')
  def test_compile_train(self, mock_time=None):
    trainer = self.test_trainer
    trainer._partitioned_train_step = mock.Mock()
    trainer.train_metrics_manager = mock.Mock(
        initial_accumulator={'fake_metric': 0})

    # compile start, compile end
    mock_time.side_effect = [1, 5]

    batch = {
        'i': np.arange(10, dtype=np.int32).reshape((2, 5)),
        'j': np.ones((), dtype=np.float32)
    }
    trainer.compile_train(batch)

    trainer.train_metrics_manager.write_scalar.assert_called_with(
        'timing/compilation_seconds', 4, trainer.train_state.step)
    trainer._partitioned_train_step.lower.assert_called_once()
    train_step_args = trainer._partitioned_train_step.lower.call_args[0]
    self.assertLen(train_step_args, 3)
    self.assertEqual(train_step_args[0], trainer.train_state)
    test_utils.assert_same(train_step_args[1], batch)
    self.assertDictEqual(train_step_args[2], {'fake_metric': 0})

  @mock.patch('time.time')
  def test_compile_eval(self, mock_time=None):
    trainer = self.test_trainer
    trainer._partitioned_eval_step = mock.Mock()
    trainer.eval_metrics_managers = {
        'eval1': mock.Mock(initial_accumulator={'fake_metric1': 0}),
        'eval2': mock.Mock(initial_accumulator={'fake_metric2': 1}),
        'eval3': mock.Mock(initial_accumulator={'fake_metric2': 1}),
        'eval4': mock.Mock(initial_accumulator={'fake_metric3': 1})
    }
    trainer._partitioned_eval_step.lower().compile.side_effect = [
        'compiled1', 'compiled2', 'compiled3'
    ]
    # eval1 start/end, eval2 start/end, eval3 start/end, eval 4 start/end
    mock_time.side_effect = [1, 5, 6, 9, 10, 11, 12, 13]

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
            'j': np.zeros((), dtype=np.float32)
        },
    }

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
    self.assertLen(eval1_call_args, 3)
    self.assertEqual(eval1_call_args[0], trainer.train_state)
    test_utils.assert_same(eval1_call_args[1], {
        'i': np.zeros((2, 5), dtype=np.int32),
    })
    self.assertDictEqual(eval1_call_args[2], {'fake_metric1': 0})

    eval2_call_args = eval_step_args[1][0]
    self.assertLen(eval2_call_args, 3)
    self.assertEqual(eval2_call_args[0], trainer.train_state)
    test_utils.assert_same(eval2_call_args[1], {
        'j': np.zeros((), dtype=np.float32),
    })
    self.assertDictEqual(eval2_call_args[2], {'fake_metric2': 1})

    eval3_call_args = eval_step_args[2][0]
    self.assertLen(eval3_call_args, 3)
    self.assertEqual(eval3_call_args[0], trainer.train_state)
    test_utils.assert_same(eval3_call_args[1], {
        'j': np.zeros((), dtype=np.float32),
    })
    self.assertDictEqual(eval3_call_args[2], {'fake_metric3': 1})

    self.assertDictEqual(
        trainer._compiled_eval_steps, {
            'eval1': 'compiled1',
            'eval2': 'compiled2',
            'eval3': 'compiled2',
            'eval4': 'compiled3'
        })


class TrainerRngDeterminismTest(parameterized.TestCase):

  def create_trainer(self, step, random_seed):
    init_optimizer = optim.Optimizer(
        optim.GradientDescent(),
        state=optim.OptimizerState(
            step=step, param_states={
                'bias': 0,
                'kernel': 0
            }),
        target={
            'bias': np.zeros(4),
            'kernel': np.zeros((2, 4))
        })
    init_train_state = train_state_lib.TrainState.from_flax_optimizer(
        init_optimizer)
    train_state_axes = jax.tree_map(lambda x: None, init_train_state)

    test_trainer = trainer_lib.Trainer(
        mock.Mock(
            get_initial_metrics=lambda:  # pylint:disable=g-long-lambda
            {
                'rng': np.zeros(2, np.uint32),
            },
            summarize_metrics_fn=lambda metrics, duration, num_steps: metrics),
        init_train_state,
        partitioning.ModelBasedPjitPartitioner(num_partitions=1),
        eval_names=['task1', 'task2'],
        summary_dir=None,
        train_state_axes=train_state_axes,
        rng=jax.random.PRNGKey(random_seed),
        learning_rate_fn=lambda step: 2 * step,
        num_microbatches=None)
    return test_trainer

  def create_ds(self, batch_size: int, num_batches: int):
    # The fake trainer increments train_state.step by sum(batch['i']). Hence
    # create datasets where sum(batch['i']) == 1 to increment train_state.step
    # by 1 for the test.
    batch = [1] + [0] * (batch_size - 1)
    batches = batch * num_batches
    mapfn = lambda i: {'i': [tf.cast(i, tf.int32)], 'j': [tf.cast(1, tf.int32)]}
    return tf.data.Dataset.from_tensor_slices(batches).map(mapfn).batch(
        batch_size, drop_remainder=True)

  @mock.patch('t5x.trainer.accumulate_grads_microbatched')
  @mock.patch('t5x.trainer.apply_grads', fake_apply_grads)
  def test_rng_determinism(self, mock_accum_grads):

    def fake_accum_grads_rng(model, optimizer, batch, rng, num_microbatches):
      del model, batch, num_microbatches
      # Add 1, which will increment the step as a side effect.
      grad_accum = jax.tree_map(lambda x: 1, optimizer)
      tf.compat.v1.logging.info(rng)
      return grad_accum, {'rng': rng}

    mock_accum_grads.side_effect = fake_accum_grads_rng
    # Create a trainer at a given step (53) with a given random seed (23),
    # train up to a given train step (100), check the sum of the rngs from the
    # metrics.
    start_step = 47
    end_step = 100
    random_seed = 23
    trainer = self.create_trainer(step=start_step, random_seed=random_seed)
    ds = self.create_ds(batch_size=2, num_batches=500)

    metrics = trainer.train(
        ds.as_numpy_iterator(), num_steps=end_step - start_step)
    base_rng = jax.random.PRNGKey(random_seed)
    expected_rng_sum = np.sum(
        [jax.random.fold_in(base_rng, i) for i in range(start_step, end_step)],
        axis=0,
        dtype=np.uint32)
    tf.compat.v1.logging.info(metrics)
    np.testing.assert_array_equal(metrics['rng'], expected_rng_sum)


if __name__ == '__main__':
  absltest.main()
