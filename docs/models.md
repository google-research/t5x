# Models


This page lists the available pre-trained T5 models. To use a pre-trained model,
you need a Gin config file that defines the model params, and the model
checkpoint to load from. For your convenience, TensorFlow checkpoints and Gin
configs for common T5 pre-trained models have been made available for use in
T5X. Following is a list of these pre-trained models and their Gin and
checkpoint locations.

+   All checkpoints:
    [`gs://t5-data/pretrained_models/t5x/`](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/)
+   All Gin files:
    [`t5x/configs/models/`](https://github.com/google-research/t5x/tree/main/t5x/configs/)

### Public Research Models

#### T5 Checkpoints

These are the checkpoints used in the paper [Exploring the Limits of Transfer
Learning with a Unified Text-to-Text
Transformer](https://arxiv.org/abs/1910.10683). They are encoder-decoder models
trained on [C4](https://www.tensorflow.org/datasets/catalog/c4) with a denoising
objective.

Model    | Gin File Location                                                              | Checkpoint Location
-------- | ------------------------------------------------------------------------------ | -------------------
T5 Small | [t5_small.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_0/small.gin) | [gs://t5-data/pretrained_models/t5x/t5_small/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_small)
T5 Base  | [t5_base.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_0/base.gin)   | [gs://t5-data/pretrained_models/t5x/t5_base/checkpoint_999900](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_base)
T5 Large | [t5_large.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_0/large.gin) | [gs://t5-data/pretrained_models/t5x/t5_large/checkpoint_1000700](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_large)
T5 3B    | [t5_3B.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_0/3B.gin)       | [gs://t5-data/pretrained_models/t5x/t5_3B/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_3B)
T5 11B   | [t5_11B.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_0/11B.gin)     | [gs://t5-data/pretrained_models/t5x/t5_11B/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_11B)

#### T5 1.1 Checkpoints

These are similar to the models from [Exploring the Limits of Transfer Learning
with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683), but
with the following improvements:

*   GEGLU activation in feed-forward hidden layer, rather than ReLU - see
    https://arxiv.org/abs/2002.05202 .
*   Dropout was turned off in pre-training (quality win). Dropout should be
    re-enabled during fine-tuning.
*   Pre-trained on C4 only without mixing in the downstream tasks.
*   no parameter sharing between embedding and classifier layer
*   "xl" and "xxl" replace "3B" and "11B". The model shapes are a bit
    different - larger d_model and smaller num_heads and d_ff.

For English-language, sequence-to-sequence-style tasks (ones where the goal is
to map from an input text sequence to a target sequence) these are usually the
best models to fine-tune.

Model        | Gin File Location                                                                  | Checkpoint Location
------------ | ---------------------------------------------------------------------------------- | -------------------
T5 1.1 Small | [t5_1_1/small.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_1/small.gin) | [gs://t5-data/pretrained_models/t5x/t5_1_1_small/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_small)
T5 1.1 Base  | [t5_1_1/base.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_1/base.gin)   | [gs://t5-data/pretrained_models/t5x/t5_1_1_base/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_base)
T5 1.1 Large | [t5_1_1_large.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_1/large.gin) | [gs://t5-data/pretrained_models/t5x/t5_1_1_large/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_large)
T5 1.1 XL    | [t5_1_1_xl.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_1/xl.gin)       | [gs://t5-data/pretrained_models/t5x/t5_1_1_xl/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_xl)
T5 1.1 XXL   | [t5_1_1_xxl.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_1/xxl.gin)     | [gs://t5-data/pretrained_models/t5x/t5_1_1_xxl/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_xxl)

#### T5 1.1 LM-Adapted Checkpoints

These "LM-adapted" models are initialized from T5 1.1 (above) and trained for an
additional 100K steps on the LM objective discussed in the
[T5 paper](https://arxiv.org/abs/1910.10683). This adaptation improves the
ability of the model to be used for
[prompt tuning](https://arxiv.org/abs/2104.08691). These checkpoints were also
used within the BigScience [T0](https://arxiv.org/abs/2110.08207) project.

Model                | Gin File Location                                                                                                   | Checkpoint Location
-------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------
T5 1.1 LM-100K Small | [t5_1_1_small.gin](https://github.com/google-research/t5x/tree/main/t5x/google/examples/flaxformer_t5/configs/models/t5_1_1_small.gin) | [t5_1_1_lm100k_small/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_lm100k_small)
T5 1.1 LM-100K Base  | [t5_1_1_base.gin](https://github.com/google-research/t5x/tree/main/t5x/google/examples/flaxformer_t5/configs/models/t5_1_1_base.gin)   | [t5_1_1_lm100k_base/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_lm100k_base)
T5 1.1 LM-100K Large | [t5_1_1_large.gin](https://github.com/google-research/t5x/tree/main/t5x/google/examples/flaxformer_t5/configs/models/t5_1_1_large.gin) | [t5_1_1_lm100k_large/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_lm100k_large)
T5 1.1 LM-100K XL    | [t5_1_1_xl.gin](https://github.com/google-research/t5x/tree/main/t5x/google/examples/flaxformer_t5/configs/models/t5_1_1_xl.gin)       | [t5_1_1_lm100k_xl/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_lm100k_xl)
T5 1.1 LM-100K XXL   | [t5_1_1_xxl.gin](https://github.com/google-research/t5x/tree/main/t5x/google/examples/flaxformer_t5/configs/models/t5_1_1_xxl.gin)     | [t5_1_1_lm100k_xxl/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_lm100k_xxl)

#### MT5 Checkpoints

These are the checkpoints used in the paper
[mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer](https://aclanthology.org/2021.naacl-main.41/).
They are encoder-decoder models trained on
[multilingual C4](https://www.tensorflow.org/datasets/catalog/c4#c4multilingual)
with a denoising objective. These are the best checkpoints to fine-tune for
non-English sequence-to-sequence tasks.

Model     | Gin File Location                                                            | Checkpoint Location
--------- | ---------------------------------------------------------------------------- | -------------------
MT5 Small | [mt5/small.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/mt5/small.gin) | [gs://t5-data/pretrained_models/t5x/mt5_small/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/mt5_small)
MT5 Base  | [mt5/base.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/mt5/base.gin)   | [gs://t5-data/pretrained_models/t5x/mt5_base/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/mt5_base)
MT5 Large | [mt5/large.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/mt5/large.gin) | [gs://t5-data/pretrained_models/t5x/mt5_large/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/mt5_large)
MT5 XL    | [mt5/xl.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/mt5/xl.gin)       | [gs://t5-data/pretrained_models/t5x/mt5_xl/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/mt5_xl)
MT5 XXL   | [mt5/xxl.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/mt5/xxl.gin)     | [gs://t5-data/pretrained_models/t5x/mt5_xxl/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/mt5_xxl)

#### LongT5 Checkpoints

These are the checkpoints used in the paper
[LongT5: Efficient Text-to-Text Transformer for Long Sequences](https://arxiv.org/abs/2112.07916).
They are encoder-decoder models trained on
[C4](https://www.tensorflow.org/datasets/catalog/c4) using the PEGASUS Principle
Sentences Generation objective. These are the recommended checkpoints to
fine-tune for long input sequence tasks.

##### LongT5 Local Attention Checkpoints

The checkpoints below use local attention, which uses a sliding window to reduce
training time from quadratic (with regards to input length) to linear. These are
the recommended checkpoints to use for faster training/inference time.

Model                        | Gin File Location                                                                                                                     | Checkpoint Location
---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------
LongT5 Local Attention Base  | [longt5/models/longt5_1_1_base.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/longt5/models/longt5_1_1_base.gin)   | [gs://t5-data/pretrained_models/t5x/longt5/local_base/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/longt5/local_base)
LongT5 Local Attention Large | [longt5/models/longt5_1_1_large.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/longt5/models/longt5_1_1_large.gin) | [gs://t5-data/pretrained_models/t5x/longt5/local_large/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/longt5/local_large)

##### LongT5 Transient Global Attention Checkpoints

The checkpoints below use transient global attention, which introduces global
tokens at each encoder layer to allow tokens to interact with each other at
longer distances. These are the recommended checkpoints to use for increased
performance on long input sequence tasks.

Model        | Gin File Location                                                                                                                                                | Checkpoint Location
------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------
LongT5 Base  | [longt5/models/longt5_1_1_transient_base.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/longt5/models/longt5_1_1_transient_global_base.gin)   | [gs://t5-data/pretrained_models/t5x/longt5/tglobal_base/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/longt5/tglobal_base)
LongT5 Large | [longt5/models/longt5_1_1_transient_large.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/longt5/models/longt5_1_1_transient_global_large.gin) | [gs://t5-data/pretrained_models/t5x/longt5/tglobal_large/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/longt5/tglobal_large)
LongT5 XL    | [longt5/models/longt5_1_1_transient_xl.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/longt5/models/longt5_1_1_transient_global_xl.gin)       | [gs://t5-data/pretrained_models/t5x/longt5/tglobal_xl/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/longt5/tglobal_xl)

