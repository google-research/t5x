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
    [`t5x/configs/models/`](https://github.com/google-research/t5x/blob/main/t5x/configs/)

### Selecting a model:

Publicly Available Models:

Model             | Use Case
---------------------------------------     | ---------------------------------------------------
[T5 1.1](#t5-11-checkpoints)                | Improved T5, recommended for most research. English only.
[T5](#t5-checkpoints)                       | The original T5 work for reproducibility. English only.
[T5 1.1 LM-Adapted](#t5-11-lm-adapted-checkpoints)| Trained for 100k additional steps on the LM objective, per [prompt tuning paper](https://arxiv.org/abs/2104.08691).
[mT5](#mt5-checkpoints)                     | Multilingual T5. Recommended for multilingual research. Note that at smaller scales (at least through XL), mT5 performance is lower than T5 on English tasks.
[mT5 LM-Adapted](#mt5-lm-adapted-checkpoints)| Trained for 100k additional steps on the LM objective, per [zero-shot cross-lingual generation (XGen) paper](https://arxiv.org/abs/2205.12647).
[umT5](#umt5-checkpoints)                   | umT5, an updated mT5 model trained using a more uniform language distribution, per [the UniMax paper](https://openreview.net/forum?id=kXwdL1cWOAi).
[ByT5](#byt5-checkpoints)                   | ByT5. A "token-free" model that uses UTF-8 bytes for input and output. Recommended for tasks involving word-internal phenomena such as spelling, pronunciation, or morphology.
[LongT5](#longt5-checkpoints)               | TBD
[MoE](#mixture-of-experts-moe-checkpoints)  | Useful for MoE experimentation.
[Flan-T5](#flan-t5-checkpoints)  | General purpose T5 checkpoints for few-shot and finetuning. We recommend Flan-T5 over vanilla T5 and T5 LM-adapted


### Public Research Models

#### T5 Checkpoints

These are the checkpoints used in the paper [Exploring the Limits of Transfer
Learning with a Unified Text-to-Text
Transformer](https://arxiv.org/abs/1910.10683). They are encoder-decoder models
pre-trained on [C4](https://www.tensorflow.org/datasets/catalog/c4) with a "span
corruption" denoising objective, in addition to a mixture of downstream tasks
including: GLUE, SuperGLUE, CNN/Daily Mail, SQuAD, and WMT.

**Vocabulary:**
[cc_all.32000.100extra](https://console.cloud.google.com/storage/browser/t5-data/vocabs/cc_all.32000.100extra)

Model    | Gin File Location                                                              | Checkpoint Location
-------- | ------------------------------------------------------------------------------ | -------------------
T5 Small | [t5_small.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_0/small.gin) | [gs://t5-data/pretrained_models/t5x/t5_small/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_small)
T5 Base  | [t5_base.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_0/base.gin)   | [gs://t5-data/pretrained_models/t5x/t5_base/checkpoint_999900](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_base)
T5 Large | [t5_large.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_0/large.gin) | [gs://t5-data/pretrained_models/t5x/t5_large/checkpoint_1000700](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_large)
T5 3B    | [t5_3B.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_0/3B.gin)       | [gs://t5-data/pretrained_models/t5x/t5_3B/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_3B)
T5 11B   | [t5_11B.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_0/11B.gin)     | [gs://t5-data/pretrained_models/t5x/t5_11B/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_11B)

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

**Vocabulary:**
[cc_all.32000.100extra](https://console.cloud.google.com/storage/browser/t5-data/vocabs/cc_all.32000.100extra)

Model        | Gin File Location                                                                  | Checkpoint Location
------------ | ---------------------------------------------------------------------------------- | -------------------
T5 1.1 Small | [t5_1_1/small.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/small.gin) | [gs://t5-data/pretrained_models/t5x/t5_1_1_small/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_small)
T5 1.1 Base  | [t5_1_1/base.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/base.gin)   | [gs://t5-data/pretrained_models/t5x/t5_1_1_base/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_base)
T5 1.1 Large | [t5_1_1_large.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/large.gin) | [gs://t5-data/pretrained_models/t5x/t5_1_1_large/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_large)
T5 1.1 XL    | [t5_1_1_xl.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/xl.gin)       | [gs://t5-data/pretrained_models/t5x/t5_1_1_xl/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_xl)
T5 1.1 XXL   | [t5_1_1_xxl.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/xxl.gin)     | [gs://t5-data/pretrained_models/t5x/t5_1_1_xxl/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_xxl)

#### T5 1.1 LM-Adapted Checkpoints

These "LM-adapted" models are initialized from T5 1.1 (above) and trained for an
additional 100K steps on the LM objective discussed in the
[T5 paper](https://arxiv.org/abs/1910.10683). This adaptation improves the
ability of the model to be used for
[prompt tuning](https://arxiv.org/abs/2104.08691). These checkpoints were also
used within the BigScience [T0](https://arxiv.org/abs/2110.08207) project.

**Vocabulary:**
[cc_all.32000.100extra](https://console.cloud.google.com/storage/browser/t5-data/vocabs/cc_all.32000.100extra)

Model                | Gin File Location                                                                                                   | Checkpoint Location
-------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------
T5 1.1 LM-100K Small | [t5_1_1_small.gin](https://github.com/google-research/t5x/blob/main/t5x/google/examples/flaxformer_t5/configs/models/t5_1_1_small.gin) | [t5_1_1_lm100k_small/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_lm100k_small)
T5 1.1 LM-100K Base  | [t5_1_1_base.gin](https://github.com/google-research/t5x/blob/main/t5x/google/examples/flaxformer_t5/configs/models/t5_1_1_base.gin)   | [t5_1_1_lm100k_base/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_lm100k_base)
T5 1.1 LM-100K Large | [t5_1_1_large.gin](https://github.com/google-research/t5x/blob/main/t5x/google/examples/flaxformer_t5/configs/models/t5_1_1_large.gin) | [t5_1_1_lm100k_large/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_lm100k_large)
T5 1.1 LM-100K XL    | [t5_1_1_xl.gin](https://github.com/google-research/t5x/blob/main/t5x/google/examples/flaxformer_t5/configs/models/t5_1_1_xl.gin)       | [t5_1_1_lm100k_xl/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_lm100k_xl)
T5 1.1 LM-100K XXL   | [t5_1_1_xxl.gin](https://github.com/google-research/t5x/blob/main/t5x/google/examples/flaxformer_t5/configs/models/t5_1_1_xxl.gin)     | [t5_1_1_lm100k_xxl/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_lm100k_xxl)


#### mT5 Checkpoints

These are the checkpoints used in the paper
[mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer](https://aclanthology.org/2021.naacl-main.41/).
They are encoder-decoder models trained on
[multilingual C4](https://www.tensorflow.org/datasets/catalog/c4#c4multilingual)
with a denoising objective. These are the best checkpoints to fine-tune for
non-English sequence-to-sequence tasks.

**Vocabulary:**
[mc4.250000.100extra](https://console.cloud.google.com/storage/browser/t5-data/vocabs/mc4.250000.100extra)

Model     | Gin File Location                                                            | Checkpoint Location
--------- | ---------------------------------------------------------------------------- | -------------------
mT5 Small | [mt5/small.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/mt5/small.gin) | [gs://t5-data/pretrained_models/t5x/mt5_small/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/mt5_small)
mT5 Base  | [mt5/base.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/mt5/base.gin)   | [gs://t5-data/pretrained_models/t5x/mt5_base/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/mt5_base)
mT5 Large | [mt5/large.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/mt5/large.gin) | [gs://t5-data/pretrained_models/t5x/mt5_large/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/mt5_large)
mT5 XL    | [mt5/xl.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/mt5/xl.gin)       | [gs://t5-data/pretrained_models/t5x/mt5_xl/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/mt5_xl)
mT5 XXL   | [mt5/xxl.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/mt5/xxl.gin)     | [gs://t5-data/pretrained_models/t5x/mt5_xxl/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/mt5_xxl)

#### mT5 LM-Adapted Checkpoints

These are the checkpoints released as part of the [zero-shot cross-lingual generation (XGen) paper](https://arxiv.org/abs/2205.12647).

These "LM-adapted" models are initialized from mT5 (above) and trained for an
additional 100K steps on the LM objective discussed in the [T5 paper](https://arxiv.org/abs/1910.10683).

This adaptation improves the ability of the model to be used for [prompt tuning](https://arxiv.org/abs/2104.08691).

**Vocabulary:**
[mc4.250000.100extra](https://console.cloud.google.com/storage/browser/t5-data/vocabs/mc4.250000.100extra)

Model                | Gin File Location                                                                                                   | Checkpoint Location
-------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------
mT5 LM-Adapted Small | [mt5/small.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/mt5/small.gin) | [mt5_lm_adapted/small/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/mt5_lm_adapted/small/checkpoint_1100000)
mT5 LM-Adapted Base | [mt5/base.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/mt5/base.gin) | [mt5_lm_adapted/base/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/mt5_lm_adapted/base/checkpoint_1100000)
mT5 LM-Adapted Large | [mt5/large.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/mt5/large.gin) | [mt5_lm_adapted/large/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/mt5_lm_adapted/large/checkpoint_1100000)
mT5 LM-Adapted XL | [mt5/xl.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/mt5/xl.gin) | [mt5_lm_adapted/xl/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/mt5_lm_adapted/xl/checkpoint_1100000)
mT5 LM-Adapted XXL | [mt5/xxl.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/mt5/xxl.gin) | [mt5_lm_adapted/xxl/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/mt5_lm_adapted/xxl/checkpoint_1100000)

#### umT5 Checkpoints

These are the checkpoints described in the paper [UniMax: Fairer and More
Effective Language Sampling for Large-Scale Multilingual
Pretraining](https://openreview.net/forum?id=kXwdL1cWOAi). umT5 is similar to
mT5 (see above); both are multilingual encoder-decoder models ranging from 300M
to 13B parameters, trained on the mC4 corpus using a denoising objective. umT5
is trained on a fresher version of the mC4 corpus (3.1.0), and with a more
uniform language balancing strategy.

**Vocabulary:**
[umt5.256000](https://console.cloud.google.com/storage/browser/t5-data/vocabs/umt5.256000)

Model                | Gin File Location                                                                                                   | Checkpoint Location
-------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------
umT5 Small | [umt5/small.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/umt5/small.gin) | [umt5/small/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/umt5/small/checkpoint_1000000)
umT5 Base | [umt5/base.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/umt5/base.gin) | [umt5/base/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/umt5/base/checkpoint_1000000)
umT5 XL | [umt5/xl.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/umt5/xl.gin) | [umt5/xl/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/umt5/xl/checkpoint_1000000)
umT5 XXL | [umt5/xxl.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/umt5/xxl.gin) | [umt5/xxl/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/umt5/xxl/checkpoint_1000000)

#### ByT5 Checkpoints

These are the checkpoints used in the paper [ByT5: Towards a Token-Free Future
with Pre-trained Byte-to-Byte Models](https://aclanthology.org/2022.tacl-1.17/).
They are similar to mT5 (above), but are "token-free", processing text as raw
UTF-8 bytes, as opposed to using a pretrained subword vocabulary. These models
are more robust to character-level noise, and outperform parameter-matched mT5
models in many settings, particularly on word-level tasks sensitive to spelling,
pronunciation, or morphology. However inference is significantly slower, up to
10x depending on the task.

**Vocabulary:** None

Model     | Gin File Location                                                            | Checkpoint Location
--------- | ---------------------------------------------------------------------------- | -------------------
ByT5 Small | [byt5/small.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/byt5/small.gin) | [gs://t5-data/pretrained_models/t5x/byt5_small/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/byt5_small)
ByT5 Base  | [byt5/base.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/byt5/base.gin)   | [gs://t5-data/pretrained_models/t5x/byt5_base/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/byt5_base)
ByT5 Large | [byt5/large.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/byt5/large.gin) | [gs://t5-data/pretrained_models/t5x/byt5_large/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/byt5_large)
ByT5 XL    | [byt5/xl.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/byt5/xl.gin)       | [gs://t5-data/pretrained_models/t5x/byt5_xl/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/byt5_xl)
ByT5 XXL   | [byt5/xxl.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/byt5/xxl.gin)     | [gs://t5-data/pretrained_models/t5x/byt5_xxl/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/byt5_xxl)

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

**Vocabulary:**
[cc_all.32000.100extra](https://console.cloud.google.com/storage/browser/t5-data/vocabs/cc_all.32000.100extra)

Model                        | Gin File Location                                                                                                                     | Checkpoint Location
---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------
LongT5 Local Attention Base  | [longt5/models/longt5_1_1_base.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/longt5/models/longt5_1_1_base.gin)   | [gs://t5-data/pretrained_models/t5x/longt5/local_base/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/longt5/local_base)
LongT5 Local Attention Large | [longt5/models/longt5_1_1_large.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/longt5/models/longt5_1_1_large.gin) | [gs://t5-data/pretrained_models/t5x/longt5/local_large/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/longt5/local_large)

##### LongT5 Transient Global Attention Checkpoints

The checkpoints below use transient global attention, which introduces global
tokens at each encoder layer to allow tokens to interact with each other at
longer distances. These are the recommended checkpoints to use for increased
performance on long input sequence tasks.

**Vocabulary:**
[cc_all.32000.100extra](https://console.cloud.google.com/storage/browser/t5-data/vocabs/cc_all.32000.100extra)

Model        | Gin File Location                                                                                                                                                | Checkpoint Location
------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------
LongT5 Base  | [longt5/models/longt5_1_1_transient_base.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/longt5/models/longt5_1_1_transient_global_base.gin)   | [gs://t5-data/pretrained_models/t5x/longt5/tglobal_base/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/longt5/tglobal_base)
LongT5 Large | [longt5/models/longt5_1_1_transient_large.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/longt5/models/longt5_1_1_transient_global_large.gin) | [gs://t5-data/pretrained_models/t5x/longt5/tglobal_large/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/longt5/tglobal_large)
LongT5 XL    | [longt5/models/longt5_1_1_transient_xl.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/longt5/models/longt5_1_1_transient_global_xl.gin)       | [gs://t5-data/pretrained_models/t5x/longt5/tglobal_xl/checkpoint_1000000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/longt5/tglobal_xl)


#### Mixture of Experts (MoE) Checkpoints

These MoE checkpoints need to be used with T5X MoE overrides -- specifically,
the MoeTrainer and the MoePjitPartitioner. For example, for fine-tuning, use the
[MoE fine-tune run config](https://github.com/google-research/t5x/blob/main/t5x/contrib/moe/configs/runs/finetune.gin).


##### Converted Mesh Tensorflow checkpoints

These are the checkpoints from the
[Switch Transformer model](https://arxiv.org/abs/2101.03961).

**Vocabulary:**
[cc_all.32000.100extra](https://console.cloud.google.com/storage/browser/t5-data/vocabs/cc_all.32000.100extra)

Model                                    | Gin File Location                                                                                            | Checkpoint Location
---------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------
Switch Transformer Base 8 Experts        | [switch_base.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/moe/models/switch_base.gin)   | [gs://t5-data/pretrained_models/t5x/moe/switch_classic/base/e8/checkpoint_500100](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/moe/switch_classic/base/e8)
Switch Transformer Base 16 Experts       | [switch_base.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/moe/models/switch_base.gin)   | [gs://t5-data/pretrained_models/t5x/moe/switch_classic/base/e16/checkpoint_550000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/moe/switch_classic/base/e16)
Switch Transformer Base 32 Experts       | [switch_base.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/moe/models/switch_base.gin)   | [gs://t5-data/pretrained_models/t5x/moe/switch_classic/base/e32/checkpoint_550000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/moe/switch_classic/base/e32)
Switch Transformer Base 64 Experts       | [switch_base.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/moe/models/switch_base.gin)   | [gs://t5-data/pretrained_models/t5x/moe/switch_classic/base/e64/checkpoint_550000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/moe/switch_classic/base/e64)
Switch Transformer Base 128 Experts      | [switch_base.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/moe/models/switch_base.gin)   | [gs://t5-data/pretrained_models/t5x/moe/switch_classic/base/e128/checkpoint_550000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/moe/switch_classic/base/e128)
Switch Transformer Base 256 Experts      | [switch_base.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/moe/models/switch_base.gin)   | [gs://t5-data/pretrained_models/t5x/moe/switch_classic/base/e256/checkpoint_550000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/moe/switch_classic/base/e256)
Switch Transformer Large 128 Experts     | [switch_large.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/moe/models/switch_large.gin) | [gs://t5-data/pretrained_models/t5x/moe/switch_classic/large/e128/checkpoint_483100](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/moe/switch_classic/large/e128)
Switch Transformer XXL 128 Experts       | [switch_xxl.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/moe/models/switch_xxl.gin)     | [gs://t5-data/pretrained_models/t5x/moe/switch_classic/xxl/e128/checkpoint_634600](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/moe/switch_classic/xxl/e128)
Switch Transformer C 2048 Experts (1.6T) | [switch_c.gin](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/moe/models/switch_c.gin)         | [gs://t5-data/pretrained_models/t5x/moe/switch_classic/c/e2048/checkpoint_611800](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/moe/switch_classic/c/e2048)





#### Flan-T5 Checkpoints

These are the checkpoints released as part of the paper [Scaling
Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416). They were
initialized from the [T5 1.1 LM-Adapted](#t5-11-lm-adapted-checkpoints) and instruction-finetuned.

They significantly outperform the LM-adapted checkpoints. For example,
Flan-T5-XXL outperforms T5-LM-XXL by 26.6% absolute on the normalized average
score. It even outperforms a much larger PaLM 62B model on [BigBench
Hard](https://arxiv.org/abs/2210.09261) a
set of challenging BigBench benchmark.

Unlike the vanilla T5 checkpoints, these can be directly used for
few-shot prompting as well as standard finetuning. See [Chung et al. 2022](https://arxiv.org/abs/2210.11416) for details.

Model                | Gin File Location                                                                                                   | Checkpoint Location
-------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------
Flan-T5 Small | [t5_1_1/small.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/small.gin) | [gs://t5-data/pretrained_models/t5x/flan_t5_small/checkpoint_1198000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/flan_t5_small/checkpoint_1198000)
Flan-T5 Base  | [t5_1_1/base.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/base.gin)   | [gs://t5-data/pretrained_models/t5x/flan_t5_base/checkpoint_1184000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/flan_t5_base/checkpoint_1184000)
Flan-T5 Large | [t5_1_1_large.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/large.gin) | [gs://t5-data/pretrained_models/t5x/flan_t5_large/checkpoint_1164000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/flan_t5_large/checkpoint_1164000)
Flan-T5 XL    | [t5_1_1_xl.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/xl.gin)       | [gs://t5-data/pretrained_models/t5x/flan_t5_xl/checkpoint_1138000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/flan_t5_xl/checkpoint_1138000)
Flan-T5 XXL   | [t5_1_1_xxl.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/xxl.gin)     | [gs://t5-data/pretrained_models/t5x/flan_t5_xxl/checkpoint_1114000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/flan_t5_xxl/checkpoint_1114000)




