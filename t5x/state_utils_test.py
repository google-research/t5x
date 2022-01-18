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

"""Tests for state_utils."""

import re

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from t5x import state_utils


class StateUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          state_dict={"a": {
              "b": 2,
              "c": 3
          }},
          intersect_state_dict={
              "a": {
                  "b": 4
              },
              "d": 5
          },
          expect_state={"a": {
              "b": 2
          }}))
  def test_intersect_state(self, state_dict, intersect_state_dict,
                           expect_state):
    actual_state = state_utils.intersect_state(state_dict, intersect_state_dict)
    self.assertEqual(actual_state, expect_state)

  @parameterized.parameters(
      dict(
          state_dict={"a": {
              "b": 2,
              "c": 3
          }},
          merge_state_dict={
              "a": {
                  "b": 4
              },
              "d": 5
          },
          expect_state={
              "a": {
                  "b": 2,
                  "c": 3
              },
              "d": 5
          }))
  def test_merge_state(self, state_dict, merge_state_dict, expect_state):
    actual_state = state_utils.merge_state(state_dict, merge_state_dict)
    self.assertEqual(actual_state, expect_state)

  def test_tensorstore_leaf(self):
    leaf = {
        "driver": "zarr",
        "kvstore": {
            "driver": "gfile",
            "path": "target.bias"
        },
        "metadata": {
            "chunks": [4, 1],
            "compressor": {
                "id": "gzip",
                "level": 1
            },
            "dtype": "<f4",
            "shape": [4, 1]
        },
    }
    self.assertTrue(state_utils.tensorstore_leaf(None, leaf))

  def test_tensorstore_leaf_missing_entries(self):
    """Don't turn a module with parameter called `driver` into a leaf."""
    leaf = {
        "driver": np.ones([4, 4]),
    }
    self.assertFalse(state_utils.tensorstore_leaf(None, leaf))

  def test_tensorstore_leaf_dtype_and_transform(self):
    leaf = {
        "driver": "zarr",
        "dtype": "float32",
        "kvstore": {
            "driver": "gfile",
            "path": "target.bias"
        },
        "metadata": {
            "chunks": [4, 1],
            "compressor": {
                "id": "gzip",
                "level": 1
            },
            "dtype": "<f4",
            "shape": [4, 1]
        },
        "transform": {
            "input_exclusive_max": [[4], [1]],
            "input_inclusive_min": [0, 0]
        }
    }
    self.assertTrue(state_utils.tensorstore_leaf(None, leaf))

  def test_flatten_state_dict(self):
    result = state_utils.flatten_state_dict({
        "target": {
            "a": {
                "b": 3
            }
        },
        "tensorstore": {
            "driver": "foo",
            "kvstore": "baz",
            "metadata": "baz",
        }
    })
    self.assertEqual(
        result, {
            "target/a/b": 3,
            "tensorstore": {
                "driver": "foo",
                "kvstore": "baz",
                "metadata": "baz"
            }
        })

  def test_apply_assignment_map_basic(self):
    assignment_map = [
        (re.compile("foo/bar"), "fandangle"),
        (re.compile("foo/baz"), "fandangle"),
    ]
    result = state_utils.apply_assignment_map(
        ckpt_optimizer_state={
            "fandangle": 1234,
            "food": 31,
        },
        optimizer_state={
            "asfoo": {
                "bar": None
            },
            "foo": {
                "bar": {
                    "baz": None
                },
                "baz": None,
            },
        },
        assignment_map=assignment_map,
        require_all_rules_match=False,
    )
    self.assertEqual(result, {
        "food": 31,
        "foo": {
            "baz": 1234
        },
    })

  def test_apply_assignment_map_glob(self):
    assignment_map = [
        (re.compile("(.*)foo/bar.*"), r"\1fandangle"),
        (re.compile("foo/baz"), "fandangle"),
    ]
    result = state_utils.apply_assignment_map(
        ckpt_optimizer_state={
            "fandangle": 1234,
            "food": 31,
            "asfandangle": 47,
        },
        optimizer_state={
            "asfoo": {
                "bar": None
            },
            "foo": {
                "bar": {
                    "qux": None
                },
                "baz": None,
            },
        },
        assignment_map=assignment_map,
        require_all_rules_match=True,
    )
    self.assertEqual(
        result, {
            "food": 31,
            "asfoo": {
                "bar": 47
            },
            "foo": {
                "bar": {
                    "qux": 1234
                },
                "baz": 1234,
            },
        })

  def test_apply_assignment_map_single_unmapped(self):
    assignment_map = []
    result = state_utils.apply_assignment_map(
        ckpt_optimizer_state={"fandangle": 1234},
        optimizer_state={"fandangle": None},
        assignment_map=assignment_map,
        require_all_rules_match=True,
    )
    self.assertEqual(result, {"fandangle": 1234})

  @parameterized.parameters(
      # Unmatched ckpt/c param implicitly matched.
      dict(
          ckpt_optimizer_state={
              "ckpt": {
                  "a": {
                      "b": 1
                  },
                  "c": 2
              },
          },
          optimizer_state={
              "target": {
                  "a": {
                      "b": 3
                  },
              },
          },
          assignment_map=((r"target/a/(.*)", r"ckpt/a/\1"),),
          expect_state={
              "target": {
                  "a": {
                      "b": 1
                  },
              },
              "ckpt": {
                  "c": 2
              }
          }),
      # Explicitly skipped param: target/c
      dict(
          ckpt_optimizer_state={
              "ckpt": {
                  "a": {
                      "b": 1
                  },
              },
          },
          optimizer_state={
              "target": {
                  "a": {
                      "b": 3
                  },
                  "c": 4
              },
          },
          assignment_map=((r"target/a/(.*)", r"ckpt/a/\1"), (r"target/c",
                                                             None)),
          expect_state={"target": {
              "a": {
                  "b": 1
              },
          }}),
  )
  def test_apply_assignment_map_partial_initialization(self,
                                                       ckpt_optimizer_state,
                                                       optimizer_state,
                                                       assignment_map,
                                                       expect_state):
    assignment_map = [(re.compile(k), v) for (k, v) in assignment_map]
    actual_state = state_utils.apply_assignment_map(
        ckpt_optimizer_state=ckpt_optimizer_state,
        optimizer_state=optimizer_state,
        assignment_map=assignment_map,
        require_all_rules_match=True)
    self.assertEqual(actual_state, expect_state)

  def test_get_name_tree(self):
    state_dict = {"a": {"b": {"c": 0, "d": {"e": 1}}}, "f": {}}

    self.assertEqual(
        state_utils.get_name_tree(state_dict),
        {"a": {
            "b": {
                "c": "a/b/c",
                "d": {
                    "e": "a/b/d/e"
                }
            }
        }})

    self.assertEqual(
        state_utils.get_name_tree(state_dict, keep_empty_nodes=True), {
            "a": {
                "b": {
                    "c": "a/b/c",
                    "d": {
                        "e": "a/b/d/e"
                    }
                }
            },
            "f": "f"
        })

if __name__ == "__main__":
  absltest.main()
