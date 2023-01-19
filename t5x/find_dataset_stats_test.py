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

"""Tests for find_dataset_stats."""

from absl.testing import absltest
from absl.testing import parameterized
from t5x import find_dataset_stats


class FindDatasetStatsTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='x2_multiple8', x=2, multiple=8, expected=8),
      dict(testcase_name='x9_multiple8', x=9, multiple=8, expected=16),
      dict(testcase_name='x64_multiple8', x=64, multiple=8, expected=64),
      dict(testcase_name='x130_multiple8', x=130, multiple=8, expected=136),
      dict(testcase_name='x9_multiple32', x=9, multiple=32, expected=32),
      dict(testcase_name='x130_multipl32', x=130, multiple=32, expected=160),
  )
  def test_nearest_multiple_of_8(self, x, multiple, expected):
    res = find_dataset_stats._nearest_multiple(x, multiple)
    self.assertEqual(res, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='dual_encoder_use_negatives=False',
          dual_encoder_use_negatives=False,
          expected_set=frozenset(),
      ),
      dict(
          testcase_name='dual_encoder_use_negatives=True',
          dual_encoder_use_negatives=True,
          expected_set=frozenset(['negative_targets']),
      ),
  )
  def test_feature_keys_to_ignore(
      self, dual_encoder_use_negatives, expected_set
  ):
    self.assertEqual(
        find_dataset_stats._feature_keys_to_ignore(dual_encoder_use_negatives),
        expected_set,
    )

  def test_calculate_seq_len_and_report(self):
    reports = []

    def testable_reporter(text: str) -> None:
      reports.append(text)

    seq_len = find_dataset_stats._calculate_seq_len_and_report(
        is_multilabel=False,
        reporter=testable_reporter,
        dataset=[{'foo': [1, 1, 1, 0], 'bar': [0, 0, 0, 0]}],
        feature_keys=('foo', 'bar'),
        seq_len_percentile=50,
        hist_bins=3,
        max_hist_stars=5,
        dual_encoder_use_negatives=False,
        seq_roundup_multiple=8,  # only applies to `inputs`
    )

    self.assertEqual(
        seq_len,
        {
            'foo': 3,
            'bar': 0,
        },
    )
    self.assertEqual(
        reports,
        [
            'feature: foo',
            'count:  1',
            'min:    3.00',
            'max:    3.00',
            'range:  0.00',
            'mean:   3.00',
            'median: 3.00',
            '50th:   3.00',
            '85th:   3.00',
            '90th:   3.00',
            '95th:   3.00',
            '99th:   3.00',
            'sum:    3.00',
            'std:    0.00',
            'var:    0.00',
            'histogram:',
            '  2.5 -> 2.8:       0',
            '  2.8 -> 3.2: ***** 1',
            '  3.2 -> 3.5:       0',
            'feature: bar',
            'count:  1',
            'min:    0.00',
            'max:    0.00',
            'range:  0.00',
            'mean:   0.00',
            'median: 0.00',
            '50th:   0.00',
            '85th:   0.00',
            '90th:   0.00',
            '95th:   0.00',
            '99th:   0.00',
            'sum:    0.00',
            'std:    0.00',
            'var:    0.00',
            'histogram:',
            '  -0.5 -> -0.2:       0',
            '  -0.2 -> 0.2: ***** 1',
            '  0.2 -> 0.5:       0',
            "FEATURE_LENGTHS: {'foo': 3, 'bar': 0}",
        ],
    )

  def test_calculate_seq_len_and_report_empty_feature_keys(self):
    reports = []

    def testable_reporter(text: str) -> None:
      reports.append(text)

    seq_len = find_dataset_stats._calculate_seq_len_and_report(
        is_multilabel=False,
        reporter=testable_reporter,
        dataset=[
            {'foo': [1, 2]},
        ],
        feature_keys=(),
        seq_len_percentile=95,
        hist_bins=3,
        max_hist_stars=5,
        dual_encoder_use_negatives=False,
        seq_roundup_multiple=8,
    )

    self.assertEqual(seq_len, {})
    self.assertEqual(reports, ['FEATURE_LENGTHS: {}'])

  @parameterized.named_parameters(
      dict(testcase_name='8', multiple=8, expected_inputs_length=16),
      dict(testcase_name='32', multiple=32, expected_inputs_length=32),
  )
  def test_calculate_seq_len_and_report_feature_round_to_nearest_multiple(
      self, multiple, expected_inputs_length
  ):
    reports = []

    def testable_reporter(text: str) -> None:
      reports.append(text)

    seq_len = find_dataset_stats._calculate_seq_len_and_report(
        is_multilabel=False,
        reporter=testable_reporter,
        dataset=[
            {
                'inputs': [1],
            },
            {
                'inputs': [1] * 9,
            },
            {
                'inputs': [1] * 15,
            },
        ],
        feature_keys=('inputs',),
        seq_len_percentile=100,
        hist_bins=5,
        max_hist_stars=2,
        dual_encoder_use_negatives=False,
        seq_roundup_multiple=multiple,
    )

    self.assertEqual(seq_len, {'inputs': expected_inputs_length})
    self.assertEqual(
        reports,
        [
            'feature: inputs',
            'count:  3',
            'min:    1.00',
            'max:    15.00',
            'range:  14.00',
            'mean:   8.33',
            'median: 9.00',
            '50th:   9.00',
            '85th:   13.80',
            '90th:   13.80',
            '95th:   14.40',
            '99th:   14.88',
            'sum:    1.00',
            'std:    5.73',
            'var:    32.89',
            'histogram:',
            '   1.0 ->  3.8: ** 1',
            '   3.8 ->  6.6:    0',
            '   6.6 ->  9.4: ** 1',
            '   9.4 -> 12.2:    0',
            '  12.2 -> 15.0: ** 1',
            f"FEATURE_LENGTHS: {{'inputs': {expected_inputs_length}}}",
        ],
    )

  def test_calculate_seq_len_and_report_uniform_buckets(self):
    reports = []

    def testable_reporter(text: str) -> None:
      reports.append(text)

    seq_len = find_dataset_stats._calculate_seq_len_and_report(
        is_multilabel=False,
        reporter=testable_reporter,
        dataset=[
            {
                'inputs': [1],
            },
            {
                'inputs': [1] * 9,
            },
            {
                'inputs': [1] * 15,
            },
        ],
        feature_keys=('inputs',),
        seq_len_percentile=100,
        hist_bins=3,
        max_hist_stars=2,
        dual_encoder_use_negatives=False,
        seq_roundup_multiple=8,
    )

    self.assertEqual(seq_len, {'inputs': 16})
    self.assertEqual(
        reports,
        [
            'feature: inputs',
            'count:  3',
            'min:    1.00',
            'max:    15.00',
            'range:  14.00',
            'mean:   8.33',
            'median: 9.00',
            '50th:   9.00',
            '85th:   13.80',
            '90th:   13.80',
            '95th:   14.40',
            '99th:   14.88',
            'sum:    1.00',
            'std:    5.73',
            'var:    32.89',
            'histogram:',
            '   1.0 ->  5.7: ** 1',
            '   5.7 -> 10.3: ** 1',
            '  10.3 -> 15.0: ** 1',
            "FEATURE_LENGTHS: {'inputs': 16}",
        ],
    )

  def test_calculate_seq_len_and_report_complex_sequences(self):
    reports = []

    def testable_reporter(text: str) -> None:
      reports.append(text)

    seq_len = find_dataset_stats._calculate_seq_len_and_report(
        is_multilabel=False,
        reporter=testable_reporter,
        dataset=[
            {'foo': [1, 2]},
            {'foo': [2, 1, 0]},
            {'foo': [1, 1]},
            {'foo': [1] * 10},
            {'foo': [0] * 100},
            {'foo': [0.5] * 20},
        ],
        feature_keys=('foo',),
        seq_len_percentile=95,
        hist_bins=3,
        max_hist_stars=5,
        dual_encoder_use_negatives=False,
        seq_roundup_multiple=8,
    )

    self.assertEqual(seq_len, {'foo': 17})
    self.assertEqual(
        reports,
        [
            'feature: foo',
            'count:  6',
            'min:    0.00',
            'max:    20.00',
            'range:  20.00',
            'mean:   6.00',
            'median: 2.00',
            '50th:   2.00',
            '85th:   15.00',
            '90th:   15.00',
            '95th:   17.50',
            '99th:   19.50',
            'sum:    0.00',
            'std:    7.02',
            'var:    49.33',
            'histogram:',
            '   0.0 ->  6.7: ***** 4',
            '   6.7 -> 13.3:       1',
            '  13.3 -> 20.0:       1',
            "FEATURE_LENGTHS: {'foo': 17}",
        ],
    )

  def test_calculate_seq_len_and_report_non_existing_feature_key(self):
    with self.assertRaises(KeyError):
      find_dataset_stats._calculate_seq_len_and_report(
          is_multilabel=False,
          reporter=lambda text: None,
          dataset=[{'foo': [1, 2]}],
          feature_keys=('foo', 'bar'),
          seq_len_percentile=50,
          hist_bins=3,
          max_hist_stars=5,
          dual_encoder_use_negatives=False,
          seq_roundup_multiple=8,
      )

  def test_calculate_seq_len_and_report_multilabel_adds_eos(self):
    reports = []

    def testable_reporter(text: str) -> None:
      reports.append(text)

    seq_len = find_dataset_stats._calculate_seq_len_and_report(
        is_multilabel=True,
        reporter=testable_reporter,
        dataset=[{'logit_tokens': [1, 2]}],
        feature_keys=('logit_tokens',),
        seq_len_percentile=50,
        hist_bins=5,
        max_hist_stars=5,
        dual_encoder_use_negatives=False,
        seq_roundup_multiple=8,
    )

    self.assertEqual(seq_len, {'targets': 3, 'logit_tokens': 2})
    self.assertIn('  1.9 -> 2.1: ***** 1', reports)
    self.assertIn("FEATURE_LENGTHS: {'logit_tokens': 2, 'targets': 3}", reports)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_negatives',
          input_dataset=[
              {'inputs': [1, 2], 'targets': [1, 2, 3, 4, 5, 6]},
              {'inputs': [1, 2, 3, 4], 'targets': [1, 2, 3, 4]},
          ],
          feature_keys=('inputs', 'targets'),
          dual_encoder_use_negatives=False,
          expected_seq_lens={
              'inputs': 8,
              'targets': 6,
          },
      ),
      dict(
          testcase_name='single_negative_per_example',
          input_dataset=[
              {
                  'inputs': [1, 2],
                  'targets': [1, 2, 3, 4, 5, 6],
                  'negative_targets': [[1, 2, 3, 4]],
              },
              {
                  'inputs': [1, 2, 3, 4],
                  'targets': [1, 2, 3, 4],
                  'negative_targets': [[1, 2, 3, 4, 5]],
              },
          ],
          feature_keys=('inputs', 'targets', 'negative_targets'),
          dual_encoder_use_negatives=True,
          expected_seq_lens={'inputs': 8, 'targets': 6, 'negative_targets': 6},
      ),
      dict(
          testcase_name='multiple_negatives_per_example',
          input_dataset=[
              {
                  'inputs': [1, 2],
                  'targets': [1, 2, 3, 4, 5, 6, 7],
                  'negative_targets': [[1, 2, 3, 4], [2, 3, 4]],
              },
              {
                  'inputs': [1, 2, 3, 4],
                  'targets': [1, 2, 3, 4],
                  'negative_targets': [[1, 2, 3, 4, 5], [2, 3, 4, 5]],
              },
          ],
          feature_keys=('inputs', 'targets', 'negative_targets'),
          dual_encoder_use_negatives=True,
          expected_seq_lens={'inputs': 8, 'targets': 7, 'negative_targets': 7},
      ),
  )
  def test_calculate_seq_len_dual_encoder(
      self,
      input_dataset,
      feature_keys,
      dual_encoder_use_negatives,
      expected_seq_lens,
  ):
    reports = []

    def testable_reporter(text: str) -> None:
      reports.append(text)

    seq_lens = find_dataset_stats._calculate_seq_len_and_report(
        is_multilabel=False,
        reporter=testable_reporter,
        dataset=input_dataset,
        feature_keys=feature_keys,
        seq_len_percentile=100,
        hist_bins=3,
        max_hist_stars=5,
        dual_encoder_use_negatives=dual_encoder_use_negatives,
        seq_roundup_multiple=8,
    )
    self.assertEqual(seq_lens, expected_seq_lens)


if __name__ == '__main__':
  absltest.main()
