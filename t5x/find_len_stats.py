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

"""Find feature sequence length statistics for a task specified by gin file."""

import collections
from collections.abc import MutableMapping
import math
from typing import Any, Callable, Dict, FrozenSet, List, Sequence, Mapping, Tuple, Union

from absl import logging
import numpy as np
import seqio

# Features that require lengths to be rounded up to a particular multiple
# (defaults to 8).
_FEATURE_KEYS_WITH_LENGTH_ROUNDUP = ("inputs",)

_TARGETS_FEATURE_KEY = "targets"
_NEGATIVE_TARGETS_FEATURE_KEY = "negative_targets"


def _feature_keys_to_ignore(dual_encoder_use_negatives: bool) -> FrozenSet[str]:
  """Returns the feature keys to ignore."""
  if dual_encoder_use_negatives:
    return frozenset([_NEGATIVE_TARGETS_FEATURE_KEY])
  return frozenset()


def _log_histogram(
    seq_lens: List[int],
    reporter: Callable[[str], None],
    hist_bins=10,
    max_hist_stars=40,
) -> None:
  """Print a human-readable histogram of the sequence lengths.

  This is an adaption of the histograms produced by gqui.

  Args:
    seq_lens: List of the sequence lengths.
    reporter: A reporter for writing the historgam.
    hist_bins: The number of bins to use for the histogram.
    max_hist_stars: The max number of stars to use in the histogram
  """
  counts, bin_edges = np.histogram(seq_lens, bins=hist_bins)

  min_count = np.amin(counts)
  max_count = np.amax(counts)
  delta_count = max_count - min_count
  # If all buckets have the same number of sequences, show the same number of
  # stars for each bucket.
  if delta_count == 0:
    delta_count = 1
    min_count -= 1
  max_bin = bin_edges[-1]

  # Estimate the maximum number of spaces to allocate for the bin and count
  # boundaries.
  bin_char_width = int(math.log10(max_bin)) + 3
  count_char_width = int(math.log10(max_count)) + 1

  for i in range(hist_bins):
    bin_low, bin_high = (bin_edges[i], bin_edges[i + 1])
    count = counts[i]
    stars = "*" * int(max_hist_stars * (count - min_count) / delta_count)
    # %- left justify the strings while numbers are right justified
    # by default.
    text = "  %*.1f -> %*.1f: %-*s %*d" % (
        bin_char_width,
        bin_low,
        bin_char_width,
        bin_high,
        max_hist_stars,
        stars,
        count_char_width,
        count,
    )
    reporter(text)


def _log_stats(
    seq_lens: List[int],
    reporter: Callable[[str], None],
    title: str,
    hist_bins=10,
    max_hist_stars=40,
) -> None:
  """Print a human readable statistics and histograms of sequence lengths.

  Args:
    seq_lens: List of the sequence lengths.
    reporter: A reporter for writing the statistical summary.
    title: A title string to write on top.
    hist_bins: The number of bins to use for the histogram.
    max_hist_stars: The max number of stars to use in the histogram
  """

  reporter(title)
  reporter("count:  %d" % len(seq_lens))
  reporter("min:    %.2f" % np.amin(seq_lens))
  reporter("max:    %.2f" % np.amax(seq_lens))
  reporter("range:  %.2f" % np.ptp(seq_lens))
  reporter("mean:   %.2f" % np.mean(seq_lens))
  reporter("median: %.2f" % np.median(seq_lens))
  reporter("50th:   %.2f" % np.percentile(seq_lens, 50))
  reporter("85th:   %.2f" % np.percentile(seq_lens, 90))
  reporter("90th:   %.2f" % np.percentile(seq_lens, 90))
  reporter("95th:   %.2f" % np.percentile(seq_lens, 95))
  reporter("99th:   %.2f" % np.percentile(seq_lens, 99))
  reporter("sum:    %.2f" % np.amin(seq_lens))
  reporter("std:    %.2f" % np.std(seq_lens))
  reporter("var:    %.2f" % np.var(seq_lens))
  reporter("histogram:")
  _log_histogram(seq_lens, reporter, hist_bins, max_hist_stars)


def _get_percentile_len(seq_lens: List[int], percentile: int) -> int:
  if np.std(seq_lens) > 0:
    return int(np.percentile(seq_lens, percentile))
  else:  # Constant sequence length
    return int(seq_lens[0])


def _generate_seq_lens(ds: Sequence[Any], feature_key: str) -> List[int]:
  """Generates sequence lengths for a keyed field in the input dataset."""
  # TODO(portalfire): Support pair/list-wise examples.
  # For dense float features, set sequence length to 1.
  # Checks that feature is 1d or 2d with 1 as the first dimension (batch).
  x = ds[0][feature_key]
  if (
      isinstance(x, np.ndarray)
      and (
          (len(x.shape) == 1 and x.shape[0] != 0)
          or (len(x.shape) == 2 and x.shape[0] == 1)
      )
      and isinstance(x.flat[0], np.floating)
  ):
    return [1]

  seq_lens = []
  for x in ds:
    ids = x[feature_key]
    non_pad_index = 0
    for token_id in ids[::-1]:
      if token_id != 0:  # PAD
        break
      non_pad_index = non_pad_index + 1

    seq_len = len(ids) - non_pad_index
    seq_lens.append(seq_len)
  return seq_lens


def _nearest_multiple(x: int, multiple: int) -> int:
  if x % multiple == 0:
    return x
  return (x // multiple + 1) * multiple


def _roundup_seq_len(seq_len_dict: Dict[str, int], seq_multiple: int) -> None:
  """Round a subset of feature lengths to nearest multiple of `seq_multiple`."""

  for feature, length in seq_len_dict.items():
    if feature not in _FEATURE_KEYS_WITH_LENGTH_ROUNDUP:
      continue
    seq_len_dict[feature] = _nearest_multiple(length, seq_multiple)


def _log_selected_seq_lens(
    seq_len_dict: Dict[str, int],
    reporter: Callable[[str], None],
    report_name: str = "FEATURE_LENGTHS",
) -> None:
  reporter(f"{report_name}: {seq_len_dict}")


def _calculate_seq_len_and_report(
    is_multilabel: bool,
    reporter: Callable[[str], None],
    dataset: Sequence[Any],
    feature_keys: Sequence[str],
    seq_len_percentile: int,
    hist_bins: int,
    max_hist_stars: int,
    dual_encoder_use_negatives: bool,
    seq_roundup_multiple: int,
) -> Mapping[str, int]:
  """Calculates sequence length and prints statistics and histogram.

  Args:
    is_multilabel: Whether this task is multilabel classification or not.
    reporter: A reporter for writing the statistics and histogram.
    dataset: Data that statistic is generated from.
    feature_keys: Keys of a data structure with which statistics from the
      `dataset` is extracted from.
    seq_len_percentile: Percentile of sequence length to report. Has to be in
      between 0 and 100, inclusive.
    hist_bins: Number of histogram bins.
    max_hist_stars: Max number of "*" to print when printing histograms.
    dual_encoder_use_negatives: Boolean that's true only when task type is
      dual_encoder and hard negatives are used.
    seq_roundup_multiple: Features in _FEATURE_KEYS_WITH_LENGTH_ROUNDUP are
      rounded up a multiple of `seq_roundup_multiple`. This is helpful to
      generally avoid TPU hardware unfriendly numbers and in models that perform
      token level operations (e.g. SparseMUM).

  Returns:
    An object that has mapping of a feature key to its statistics.
  """
  selected_seq_lens = {}
  feature_keys_to_ignore = _feature_keys_to_ignore(dual_encoder_use_negatives)

  for feature_key in feature_keys:
    if feature_key in feature_keys_to_ignore:
      logging.info("Skipping sequence length for feature: %s", feature_key)
      continue
    logging.info("Generate sequence length for feature: %s", feature_key)
    seq_lens = _generate_seq_lens(dataset, feature_key)
    _log_stats(
        seq_lens,
        reporter,
        title=f"feature: {feature_key}",
        hist_bins=hist_bins,
        max_hist_stars=max_hist_stars,
    )
    selected_seq_lens[feature_key] = _get_percentile_len(
        seq_lens, seq_len_percentile
    )
    _roundup_seq_len(selected_seq_lens, seq_roundup_multiple)

  # Multi-label classification can have between 0 and logit_tokens positives
  # per example. Also +1 for EOS token.
  if is_multilabel:
    selected_seq_lens[_TARGETS_FEATURE_KEY] = (
        selected_seq_lens["logit_tokens"] + 1
    )

  # For dual encoders, feature sequence length of "negative_targets" feature
  # should be same as "targets" feature.
  # TODO(b/237061422): Consider negative_targets lengths when computing
  # sequence length of targets feature.
  if dual_encoder_use_negatives:
    logging.info(
        "Overriding sequence length for %s feature",
        _NEGATIVE_TARGETS_FEATURE_KEY,
    )
    selected_seq_lens[_NEGATIVE_TARGETS_FEATURE_KEY] = selected_seq_lens[
        _TARGETS_FEATURE_KEY
    ]

  _log_selected_seq_lens(selected_seq_lens, reporter)
  return selected_seq_lens


def find_len_stats_from_task(
    is_multilabel: bool,
    task: Union[seqio.Mixture, seqio.Task],
    split_name: str = "train",
    n_examples: int = 10000,
    seq_len_percentile: int = 99,
    hist_bins: int = 10,
    max_hist_stars: int = 40,
    dual_encoder_use_negatives: bool = False,
    seq_roundup_multiple: int = 8,
) -> Tuple[Mapping[str, int], str]:
  """Computes sequence length for a given task."""
  # Generate statistics for each output feature.
  output_strs = []

  def _log_text(text: str) -> None:
    output_strs.append(text)

  max_seq_lens: MutableMapping[str, int] = collections.defaultdict(lambda: 0)
  subtasks = seqio.get_subtasks(task)
  for task in subtasks:
    output_features = task.output_features.keys()
    logging.info("[%s] Output features: %s", task.name, output_features)

    if split_name not in task.splits:
      logging.warning(
          "Task %s does not have split, %s. Skipping for length calculation.",
          task.name,
          split_name,
      )
      continue

    ds = task.get_dataset(split=split_name, sequence_length=None)
    ds = list(ds.take(n_examples).as_numpy_iterator())

    _log_text(f"task: {task.name}")
    _log_text("=" * max_hist_stars)
    selected_seq_lens = _calculate_seq_len_and_report(
        is_multilabel=is_multilabel,
        reporter=_log_text,
        dataset=ds,
        feature_keys=output_features,
        seq_len_percentile=seq_len_percentile,
        hist_bins=hist_bins,
        max_hist_stars=max_hist_stars,
        dual_encoder_use_negatives=dual_encoder_use_negatives,
        seq_roundup_multiple=seq_roundup_multiple,
    )
    _log_text("=" * max_hist_stars)

    for feature, length in selected_seq_lens.items():
      max_seq_lens[feature] = max(length, max_seq_lens[feature])

  _log_selected_seq_lens(dict(max_seq_lens), _log_text, "FINAL FEATURE_LENGTHS")
  report_text = "\n".join(output_strs)
  print(report_text)
  return (dict(max_seq_lens), report_text)
