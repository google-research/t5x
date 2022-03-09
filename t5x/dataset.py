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

"""Generic dataset manager interface and TF Data/SeqIO implementation.

DatasetManager and DatasetIterator are interfaces for feeding a data for
different tasks types such as train and eval. A DatasetManager is a collection
of DatasetIterator while a DatasetIterator is provides data in `get_batch`
method.

While an implementation can be as simple as simply wrapping tf.data.Dataset, it
can support more advanced features such as streaming and curriculum learning
with information encoded in `ids`.
"""

import abc
import collections
import dataclasses
import typing
from typing import Generic, Mapping, Optional, Sequence, Tuple, Type, TypeVar

import seqio

if typing.TYPE_CHECKING:
  import tensorflow as tf


@dataclasses.dataclass
class ElementSpec(dict):
  name: Optional[str]
  shape: Tuple[int, ...]
  dtype: "tf.dtypes.Dtype"


T = TypeVar("T")


class DatasetIterator(abc.ABC, collections.abc.Iterator, Generic[T]):
  """Iterator that provides data.

  For more advanced usage of the iterator, such as curriculum learning, one can
  encode information about previous state as `ids` and implement a custom
  `get_batch`.
  """

  def __iter__(self):
    return self

  def __next__(self) -> Sequence[T]:
    return self.get_batch()

  @abc.abstractmethod
  def get_batch(
      self,
      ids: Optional[Sequence[str]] = None,
  ) -> Sequence[T]:
    pass

  @property
  @abc.abstractmethod
  def element_spec(self) -> Mapping[str, ElementSpec]:
    pass


class TensorFlowDatasetIterator(DatasetIterator):
  """DatasetIterator implementation of tf.data."""

  _dataset: "tf.data.Dataset"

  def __init__(self, dataset: "tf.data.Dataset"):
    self._dataset = dataset
    self._iterator = None

  def _get_iterator(self):
    if self._iterator is None:
      self._iterator = iter(self._dataset)
    return self._iterator

  def get_batch(self, ids=None):
    # ids unused: does not vary based on previous training actions
    del ids

    return next(self._get_iterator())

  @property
  def element_spec(self) -> Mapping[str, ElementSpec]:
    return self._dataset.element_spec

  def repeat(self):
    # For compatibility with tf.data.Dataset.
    self._dataset = self._dataset.repeat()
    return self

  def take(self, num_data: int):
    # For compatibility with tf.data.Dataset.
    self._dataset = self._dataset.take(num_data)
    return self

  def as_numpy_iterator(self):
    return self._dataset.as_numpy_iterator()


class DatasetManager(abc.ABC):
  """Collection of DatasetIterator accessible with a unique name."""

  @abc.abstractmethod
  def get_iterator(self, name: str) -> DatasetIterator:
    pass

  @property
  @abc.abstractmethod
  def batch_size(self) -> int:
    pass


@dataclasses.dataclass
class DataLayout:
  batch_size: int
  shard_id: int
  num_shards: int


class SeqIoDatasetManager(DatasetManager):
  """SeqIO implementation of DatasetManage.

  A `name` of a DatasetIterator can be a mixture, a task, or a subtask.
  """

  def __init__(
      self,
      *,
      task_feature_lengths: Mapping[str, int],
      split: str,
      shuffle: bool,
      data_layout: DataLayout,
      feature_converter_cls: Type[seqio.FeatureConverter],
      seed: Optional[int] = None,
      use_custom_packing_ops: Optional[bool] = False,
      use_cached: bool = False,
      pack: bool = False,
      num_epochs: Optional[int] = None,
      start_example_from: Optional[int] = None,
  ):
    self._task_feature_lengths = task_feature_lengths
    self._split = split
    self._shuffle = shuffle
    self._use_cached = use_cached
    self._pack = pack
    self._seed = seed
    self._use_custom_packing_ops = use_custom_packing_ops
    self._data_layout = data_layout
    self._feature_converter_cls = feature_converter_cls
    self._num_epochs = num_epochs
    self._start_example_from = start_example_from
    super().__init__()

  def get_iterator(self, name) -> TensorFlowDatasetIterator:
    # Check that name == dataset_cfg.mixture_or_task or name is a valid subtask
    # Returns a Task or Mixture object
    mixture_or_task = seqio.get_mixture_or_task(name)
    shard_info = seqio.ShardInfo(
        index=self._data_layout.shard_id,
        num_shards=self._data_layout.num_shards,
    )


    ds = seqio.get_dataset(
        mixture_or_task_name=name,
        task_feature_lengths=self._task_feature_lengths,
        dataset_split=self._split,
        shuffle=self._shuffle,
        num_epochs=self._num_epochs,
        feature_converter=self._feature_converter_cls(
            pack=self._pack,
            use_custom_packing_ops=self._use_custom_packing_ops,
        ),  # pytype: disable=not-instantiable
        shard_info=shard_info,
        use_cached=self._use_cached,
        seed=self._seed,
    )
    ds = ds.batch(self.batch_size, drop_remainder=True)
    return TensorFlowDatasetIterator(ds)

  @property
  def batch_size(self) -> int:
    return self._data_layout.batch_size
