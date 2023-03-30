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

r"""Tool to convert a T5/MeshTF checkpoint to T5X.

While T5X can be load these checkpoints on-the-fly, the process can be slow
for very large checkpoints. For frequently used checkpoints, it's recommended to
convert them once to a T5X checkpoint.

Example usage:

CUDA_VISIBLE_DEVICES=""
python -m t5x.convert_tf_checkpoint \
 --gin_file=t5x/examples/t5/t5_1_0/small.gin\
 --gin.convert_checkpoint.model=%MODEL\
 --gin.convert_checkpoint.tf_checkpoint_path=\
\"gs://t5-data/pretrained_models/small/model.ckpt-1000000\"\
 --gin.convert_checkpoint.output_dir=\"/tmp/t5x_checkpoints/t5_small\"\
 --logtostderr
"""
import jax
import jax.numpy as jnp
from t5x import checkpoints
from t5x import models
from t5x import partitioning
from t5x import train_state as train_state_lib


def convert_checkpoint(model: models.BaseModel,
                       tf_checkpoint_path: str,
                       output_dir: str,
                       save_dtype: jnp.dtype = jnp.float32,
                       concurrent_gb: int = 16):
  """Converts a TensorFlow checkpoint to a P5X checkpoint.

  Args:
    model:
    tf_checkpoint_path: Path to a TensorFlow checkpoint to convert.
    output_dir: Path to a directory to write the converted checkpoint.
    save_dtype: What dtype to store the target parameters as.
    concurrent_gb: Number of gigabtes of parameters to convert in parallel.
      Actual RAM usage may be 4X this number.
  """

  def initialize_train_state(rng):
    initial_variables = model.get_initial_variables(  # pytype: disable=wrong-arg-types  # jax-array
        rng=rng,
        input_shapes={
            'encoder_input_tokens': (1, 1),
            'decoder_input_tokens': (1, 1)
        })
    return train_state_lib.FlaxOptimTrainState.create(model.optimizer_def,
                                                      initial_variables)

  train_state = jax.eval_shape(initialize_train_state, jax.random.PRNGKey(0))

  partitioner = partitioning.PjitPartitioner(1)

  checkpointer = checkpoints.Checkpointer(
      train_state, partitioner, output_dir, save_dtype=jnp.dtype(save_dtype))

  checkpointer.convert_from_tf_checkpoint(
      tf_checkpoint_path, concurrent_gb=concurrent_gb)


if __name__ == '__main__':
  # pylint:disable=g-import-not-at-top
  from absl import flags
  import gin
  from t5x import gin_utils
  # pylint:disable=g-import-not-at-top

  FLAGS = flags.FLAGS

  jax.config.parse_flags_with_absl()

  flags.DEFINE_multi_string(
      'gin_file',
      default=None,
      help='Path to gin configuration file. Multiple paths may be passed and '
      'will be imported in the given order, with later configurations  '
      'overriding earlier ones.')

  flags.DEFINE_multi_string(
      'gin_bindings', default=[], help='Individual gin bindings')

  flags.DEFINE_list(
      'gin_search_paths',
      default=['t5x/configs'],
      help='Comma-separated list of gin config path prefixes to be prepended '
      'to suffixes given via `--gin_file`. If a file appears in. Only the '
      'first prefix that produces a valid path for each suffix will be '
      'used.')

  def main(_):
    """True main function."""
    convert_checkpoint_using_gin = gin.configurable(convert_checkpoint)

    gin_utils.parse_gin_flags(FLAGS.gin_search_paths, FLAGS.gin_file,
                              FLAGS.gin_bindings)
    # Get gin-configured version of `convert_checkpoint`.
    convert_checkpoint_using_gin()

  gin_utils.run(main)
