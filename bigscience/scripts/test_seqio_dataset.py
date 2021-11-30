from t5x import models
from t5x import utils
import tensorflow as tf
from ..gins import task

def main():
    ds = utils.get_dataset(
        utils.DatasetConfig(
            "c4_v220_full_lm",
            task_feature_lengths={"targets": 626},
            split="train",
            batch_size=2048,
            shuffle=False,
            seed=None,
            use_cached=True,
            pack=True,
            use_custom_packing_ops=False,
            use_memory_cache=False,
        ),
        0,
        1,
        models.DecoderOnlyModel.FEATURE_CONVERTER_CLS
    )
    first_element = next(iter(ds))
    print(first_element)

    # This should output `dict_keys(['decoder_target_tokens', 'decoder_input_tokens', 'decoder_loss_weights', 'decoder_segment_ids', 'decoder_positions'])`
    print(first_element.keys())

    print(first_element["decoder_target_tokens"])
    print(first_element["decoder_loss_weights"])

    # This should all output `tf.Tensor([2048  626], shape=(2,), dtype=int32)`
    print(tf.shape(first_element["decoder_target_tokens"]))
    print(tf.shape(first_element["decoder_input_tokens"]))
    print(tf.shape(first_element["decoder_loss_weights"]))
    print(tf.shape(first_element["decoder_segment_ids"]))
    print(tf.shape(first_element["decoder_positions"]))

if __name__ == "__main__":
    main()