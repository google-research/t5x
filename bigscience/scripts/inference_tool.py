import functools
import os
from typing import Sequence

import tensorflow as tf

import seqio.preprocessors
from seqio import PrefixLMFeatureConverter, LMFeatureConverter, autoregressive_inputs, \
    non_padding_position
from t5.data import get_default_vocabulary
import jax
from t5x.checkpoints import PartitionSpec

from t5x import utils, partitioning

from t5x.models import DecoderOnlyModel, EncoderDecoderModel

# Automatically search for gin files relative to the T5X package.
_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
]

"""
First download checkpoint locally.
`gsutil -m cp -r ${REMOTE_CHECKPOINT} ${LOCAL_CHECKPOINT}

Run this script via:
`PYTHONPATH=$(T5X_DIR)/bigscience/gins python ${T5X_DIR}/bigscience/scripts/inference_tool.py \
  --gin_file="bigscience/gins/c_dec_xxl.gin"
  --gin_file="bigscience/gins/inference_tool.gin"
  --gin.MODEL_DIR="'${LOCAL_MODEL_DIR}'" 
  --gin.INITIAL_CHECKPOINT_PATH="'${LOCAL_CHECKPOINT}'"
"""


def get_new_batch():
    print("Please write a new prompt")
    text = input()
    return {"text": [text]}


def infer(
        model: DecoderOnlyModel,
        restore_checkpoint_config: utils.RestoreCheckpointConfig,
        partitioner: partitioning.ModelBasedPjitPartitioner
):
    # Load model
    batch_size = 1
    vocabulary = get_default_vocabulary()
    output_features = {
        "text": seqio.Feature(vocabulary=vocabulary, add_eos=True),
    }
    # restore_checkpoint_config = utils.RestoreCheckpointConfig()

    if restore_checkpoint_config.path.startswith("gs://"):
        print(f"Please download the checkpoint locally first. "
              f"`gsutil cp {restore_checkpoint_config.path} $CHECKPOINT_PATH`. "
              f"You can then run the script by add `gin.utils.RestoreCheckpointConfig=\"'$CHECKPOINT_PATH'\"`")
        return

    # Initialize optimizer from the existing checkpoint.
    if isinstance(model, DecoderOnlyModel):
        feature_lengths = {
            "decoder_target_tokens": 626,
            "decoder_input_tokens": 626,
            "decoder_segment_ids": 626,
            "decoder_causal_attention": 626,
            "targets": 626
        }
    elif isinstance(model, EncoderDecoderModel):
        feature_lengths = {
            "encoder_input_tokens": 626,
            "decoder_target_tokens": 626,
            "decoder_input_tokens": 626,
            "encoder_segment_ids": 626,
            "encoder_positions": 626,
            "decoder_segment_ids": 626,
            "decoder_positions": 626,
            "decoder_loss_weights": 626,
            "targets": 626
        }
    else:
        raise NotImplementedError

    input_shapes = {
        k: (batch_size, l) for k, l in feature_lengths.items()
    }

    train_state_initializer = utils.TrainStateInitializer(
        optimizer_def=model.optimizer_def,
        init_fn=model.get_initial_variables,
        input_shapes=input_shapes,
        partitioner=partitioner)
    train_state_axes = train_state_initializer.train_state_axes

    train_state = train_state_initializer.from_checkpoint([restore_checkpoint_config])
    if train_state is None:
        print(f"Could not load checkpoint. {restore_checkpoint_config}")
        return

    # Compile the model only once.
    predict_fn = partitioner.partition(
        model.predict_batch,
        in_axis_resources=(train_state_axes.params, PartitionSpec('data', )),
        out_axis_resources=PartitionSpec('data', ))

    predict_fn = functools.partial(predict_fn, train_state.params)

    # model type: c_dec, nc_dec, enc_dec
    if isinstance(model, DecoderOnlyModel):
        if model._inputs_bidirectional_attention:
            model_type = "nc_dec"
        else:
            model_type = "c_dec"
    elif isinstance(model, EncoderDecoderModel):
        model_type = "enc_dec"
    else:
        raise NotImplementedError

    while True:
        # Create batch interactively
        batch = get_new_batch()
        prompts = batch["text"]
        assert len(prompts) == batch_size
        # Tokenize
        tokenized_batch = seqio.preprocessors.tokenize_impl(batch, output_features, copy_pretokenized=False)

        # Convert feature to model input
        preprocessed_batch = preprocess_lm(tokenized_batch, model_type, feature_lengths, vocabulary)

        # print(preprocessed_batch)

        # Run decoding algorithm
        predictions = predict_fn(preprocessed_batch)

        # Print results in string.
        print(f"Prompt: {prompts[0]}")
        print(vocabulary.decode([int(elt) for elt in predictions[0]]))


# ----- Helpers

# Copied from seqio
def concat_and_add_masks(features):
    inputs = features["inputs"]
    targets = features["targets"]

    # Width of the "inputs" portion in the concatenated sequence.
    width = tf.size(inputs)
    inputs_width = tf.fill([tf.size(inputs) + tf.size(targets)], width)

    # Width with an extra position to the right in the inputs mask. See
    # docstring for details.
    inputs_width_add_pos = tf.fill([tf.size(inputs) + tf.size(targets)],
                                   width + 1)

    return {
        "targets": tf.concat([inputs, targets], axis=-1),
        "inputs_width": inputs_width,
        "inputs_width_add_pos": inputs_width_add_pos
    }


# Copied from seqio
def convert_enc_dec_example(
        features):
    # targets_segment_id is present only for a packed dataset.
    decoder_input_tokens = autoregressive_inputs(
        features["targets"],
        sequence_id=features.get("targets_segment_ids", None))

    d = {"encoder_input_tokens": features["inputs"],
         "decoder_target_tokens": features["targets"],
         "decoder_input_tokens": decoder_input_tokens,
         # Loss is computed for all but the padding positions.
         "decoder_loss_weights": non_padding_position(features["targets"])}

    return d


def preprocess_lm(tokenized_batch, model_type, feature_lengths, vocabulary):
    # Only support batch size 1
    assert all([value.shape[0] == 1 for value in tokenized_batch.values()])
    assert "text" in tokenized_batch

    # Format + pad target for generation
    sequence_length = tokenized_batch["text"][0].shape[0]

    if model_type == "c_dec" or model_type == "nc_dec":
        feature_converter = PrefixLMFeatureConverter(pack=False)
        pad_target_length = feature_lengths["decoder_target_tokens"] - sequence_length
        example_single_elt = {
            "inputs": tokenized_batch["text"][0],
            "targets": tf.ones(pad_target_length, dtype=tokenized_batch["text"].dtype) * vocabulary.pad_id
        }

        example_single_elt = concat_and_add_masks(example_single_elt)
        preprocessed_single_elt = feature_converter._convert_example(example_single_elt)

        # In case of `model_type = "c_dec", it's okay to use prefixLMFeatureConverter as `decoder_causal_attention` will
        # get ignored by the model as we feed `inputs_bidirectional_attention = False` to the model.
    elif model_type == "enc_dec":
        pad_target_length = feature_lengths["decoder_target_tokens"]
        example_single_elt = {
            "inputs": tokenized_batch["text"][0],
            "targets": tf.ones(pad_target_length, dtype=tokenized_batch["text"].dtype) * vocabulary.pad_id
        }
        preprocessed_single_elt = convert_enc_dec_example(example_single_elt)
    else:
        raise NotImplementedError

    return {
        k: v[None, ...].numpy() for k, v in preprocessed_single_elt.items()
    }


if __name__ == "__main__":
    from absl import app
    from absl import flags
    import gin
    from t5x import gin_utils

    FLAGS = flags.FLAGS

    jax.config.parse_flags_with_absl()

    flags.DEFINE_multi_string(
        'gin_file',
        default=None,
        help='Path to gin configuration file. Multiple paths may be passed and '
             'will be imported in the given order, with later configurations  '
             'overriding earlier ones.')

    flags.DEFINE_multi_string(
        'gin_bindings', default=[], help='Individual gin bindings.')

    flags.DEFINE_list(
        'gin_search_paths',
        default=['.'],
        help='Comma-separated list of gin config path prefixes to be prepended '
             'to suffixes given via `--gin_file`. If a file appears in. Only the '
             'first prefix that produces a valid path for each suffix will be '
             'used.')

    flags.DEFINE_string(
        'tfds_data_dir', None,
        'If set, this directory will be used to store datasets prepared by '
        'TensorFlow Datasets that are not available in the public TFDS GCS '
        'bucket. Note that this flag overrides the `tfds_data_dir` attribute of '
        'all `Task`s.')


    def main(argv: Sequence[str]):
        """Wrapper for pdb post mortems."""
        _main(argv)


    def _main(argv: Sequence[str]):
        """True main function."""
        if len(argv) > 1:
            raise app.UsageError('Too many command-line arguments.')

        if FLAGS.tfds_data_dir:
            seqio.set_tfds_data_dir_override(FLAGS.tfds_data_dir)

        # Create gin-configurable version of `eval`.
        evaluate_using_gin = gin.configurable(infer)

        gin_utils.parse_gin_flags(
            # User-provided gin paths take precedence if relative paths conflict.
            FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
            FLAGS.gin_file,
            FLAGS.gin_bindings)
        evaluate_using_gin()


    gin_utils.run(main)

# PYTHONPATH=$(pwd)/bigscience/gins python3 bigscience/scripts/inference_tool.py --gin_file="bigscience/gins/nc_dec_xxl.gin" --gin_file="bigscience/gins/inference_tool.gin" --gin.MODEL_DIR="'/home/thomas/model_dir'" --gin.INITIAL_CHECKPOINT_PATH="'/home/thomas/checkpoints/nc_dec_c4_prefix_lm/checkpoint_420000'"
# PYTHONPATH=$(pwd)/bigscience/gins python3 bigscience/scripts/inference_tool.py --gin_file="bigscience/gins/c_dec_xxl.gin" --gin_file="bigscience/gins/inference_tool.gin" --gin.MODEL_DIR="'/home/thomas/model_dir'" --gin.INITIAL_CHECKPOINT_PATH="'/home/thomas/checkpoints/c_dec_c4_full_lm/checkpoint_420000'"
# PYTHONPATH=$(pwd)/bigscience/gins python3 bigscience/scripts/inference_tool.py --gin_file="bigscience/gins/enc_dec_xxl.gin" --gin_file="bigscience/gins/inference_tool.gin" --gin.MODEL_DIR="'/home/thomas/model_dir'" --gin.INITIAL_CHECKPOINT_PATH="'/home/thomas/checkpoints/enc_dec_c4_prefix_lm/checkpoint_32768'"
