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

"""Tests for binary_search."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from t5x import binary_search

_INT32_MIN = np.iinfo(np.int32).min
_INT32_MAX = np.iinfo(np.int32).max


class BinarySearchTest(absltest.TestCase):

  def test_int32_bsearch(self):
    a = jnp.asarray([
        1,
        43,
        79,
        2048,
        0,
        2047,
        _INT32_MIN,
        _INT32_MIN + 1,
        _INT32_MAX,
        _INT32_MAX - 1,
    ],
                    dtype=jnp.int32)

    def predicate(x):
      return x > a

    r = binary_search.int32_bsearch(a.shape, predicate)
    np.testing.assert_array_equal(a, r)

  def test_int32_bsearch_extreme_predicates(self):

    def predicate_false(x):
      return jnp.full_like(x, False)

    np.testing.assert_array_equal(
        jnp.asarray([_INT32_MAX]),
        binary_search.int32_bsearch((1,), predicate_false))

    def predicate_true(x):
      return jnp.full_like(x, True)

    np.testing.assert_array_equal(
        jnp.asarray([_INT32_MIN]),
        binary_search.int32_bsearch((1,), predicate_true))

  def test_float32_bsearch(self):
    a = jnp.asarray([1.23, 0.0, -0.0, 105.4, -1024, 4.3], dtype=jnp.float32)

    def predicate(x):
      return x > a

    c = binary_search.float32_bsearch(a.shape, predicate)
    # Given that the predicate is based on floating point '>' as implemented by
    # JAX, we need our equality test to be based on floating point '==' as
    # implemented by JAX, rather than np.testing.assert_array_equal.
    #
    # Some corner cases on subnormal numbers may be different, depending on what
    # platform we run on.
    self.assertTrue(jnp.all(a == c), f'a={a}, c={c}')

  def test_topk_mask(self):
    mask = -1e10
    x = jnp.asarray([
        [1.4, 7.9, -4.3, 100, 71, 6, -1e4],
        [8.3, 1.2, 1.3, 1.2, 1.2, 9.7, -100],
    ])

    # Using exact equality here, because topk_mask guarantees it: it is just
    # masking some things, not doing arithmetic on the array.
    np.testing.assert_array_equal(
        jnp.asarray([
            [mask, mask, mask, 100, mask, mask, mask],
            [mask, mask, mask, mask, mask, 9.7, mask],
        ]),
        binary_search.topk_mask(x, 1, mask),
    )
    np.testing.assert_array_equal(
        jnp.asarray([
            [mask, mask, mask, 100, 71, mask, mask],
            [8.3, mask, mask, mask, mask, 9.7, mask],
        ]),
        binary_search.topk_mask(x, 2, mask),
    )
    np.testing.assert_array_equal(
        jnp.asarray([
            [mask, 7.9, mask, 100, 71, mask, mask],
            [8.3, mask, 1.3, mask, mask, 9.7, mask],
        ]),
        binary_search.topk_mask(x, 3, mask),
    )
    np.testing.assert_array_equal(
        jnp.asarray([
            [mask, 7.9, mask, 100, 71, 6, mask],
            [8.3, 1.2, 1.3, 1.2, 1.2, 9.7, mask],
        ]),
        binary_search.topk_mask(x, 4, mask),
    )
    np.testing.assert_array_equal(
        jnp.asarray([
            [1.4, 7.9, mask, 100, 71, 6, mask],
            [8.3, 1.2, 1.3, 1.2, 1.2, 9.7, mask],
        ]),
        binary_search.topk_mask(x, 5, mask),
    )
    np.testing.assert_array_equal(
        jnp.asarray([
            [1.4, 7.9, -4.3, 100, 71, 6, mask],
            [8.3, 1.2, 1.3, 1.2, 1.2, 9.7, mask],
        ]),
        binary_search.topk_mask(x, 6, mask),
    )
    np.testing.assert_array_equal(
        jnp.asarray([
            [1.4, 7.9, -4.3, 100, 71, 6, -1e4],
            [8.3, 1.2, 1.3, 1.2, 1.2, 9.7, -100],
        ]),
        binary_search.topk_mask(x, 7, mask),
    )

  def test_topp_mask(self):
    probs = jnp.asarray([
        [0.0, 0.7, 0.04, 0.06, 0.2, 0.0],
        [0.0, 0.2, 0.2, 0.2, 0.3, 0.1],
    ])
    logits = jnp.log(probs)
    np.testing.assert_allclose(jax.nn.softmax(logits), probs)
    mask = -1e10

    # Using exact equality here, because topp_mask guarantees it: it is just
    # masking some things, not doing arithmetic on the array.
    np.testing.assert_array_equal(
        jnp.asarray([
            [mask, jnp.log(0.7), mask, mask, mask, mask],
            [mask, mask, mask, mask, jnp.log(0.3), mask],
        ]),
        binary_search.topp_mask(logits, 0.1, mask),
    )
    np.testing.assert_array_equal(
        jnp.asarray([
            [mask, jnp.log(0.7), mask, mask, mask, mask],
            [mask, mask, mask, mask, jnp.log(0.3), mask],
        ]),
        binary_search.topp_mask(logits, 0.3, mask),
    )
    np.testing.assert_array_equal(
        jnp.asarray([
            [mask, jnp.log(0.7), mask, mask, mask, mask],
            [
                mask,
                jnp.log(0.2),
                jnp.log(0.2),
                jnp.log(0.2),
                jnp.log(0.3), mask
            ],
        ]),
        binary_search.topp_mask(logits, 0.4, mask),
    )
    np.testing.assert_array_equal(
        jnp.asarray([
            [mask, jnp.log(0.7), mask, mask,
             jnp.log(0.2), mask],
            [
                mask,
                jnp.log(0.2),
                jnp.log(0.2),
                jnp.log(0.2),
                jnp.log(0.3), mask
            ],
        ]),
        binary_search.topp_mask(logits, 0.8, mask),
    )
    np.testing.assert_array_equal(
        jnp.asarray([
            [mask, jnp.log(0.7), mask,
             jnp.log(0.06),
             jnp.log(0.2), mask],
            [
                mask,
                jnp.log(0.2),
                jnp.log(0.2),
                jnp.log(0.2),
                jnp.log(0.3),
                jnp.log(0.1)
            ],
        ]),
        binary_search.topp_mask(logits, 0.95, mask),
    )


if __name__ == '__main__':
  absltest.main()
