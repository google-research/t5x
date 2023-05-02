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

"""T5_1_1 Model Configurations, similar to t5x/examples/t5/t5_1_1/*.gin."""
from typing import Optional

import fiddle as fdl
import seqio
from t5x import adafactor
from t5x import models
from t5x import optimizers
from t5x.examples.t5 import network

Z_LOSS = 0.0001
LABEL_SMOOTHING = 0.0
# NOTE: When fine-tuning the public T5 checkpoints (trained in T5 MeshTF)
# the loss normalizing factor should be set to pretraining batch_size *
# target_token_length.
LOSS_NORMALIZING_FACTOR = None


def vocabulary() -> fdl.Buildable[seqio.SentencePieceVocabulary]:
  return fdl.Config(
      seqio.SentencePieceVocabulary,
      sentencepiece_model_file=(
          'gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model'
      ),
  )


def optimizer() -> fdl.Buildable[optimizers.OptimizerDef]:
  return fdl.Config(
      adafactor.Adafactor,
      decay_rate=0.8,
      step_offset=0,
      logical_factor_rules=fdl.Config(
          adafactor.standard_logical_factor_rules,
      ),
  )


def model(
    config: fdl.Buildable[network.T5Config],
    z_loss: float = Z_LOSS,
    label_smoothing: float = LABEL_SMOOTHING,
    loss_normalizing_factor: Optional[float] = LOSS_NORMALIZING_FACTOR,
) -> fdl.Buildable[models.BaseTransformerModel]:
  return fdl.Config(
      models.EncoderDecoderModel,
      module=fdl.Config(
          network.Transformer,
          config=config,
      ),
      input_vocabulary=vocabulary(),
      output_vocabulary=vocabulary(),
      optimizer_def=optimizer(),
      z_loss=z_loss,
      label_smoothing=label_smoothing,
      loss_normalizing_factor=loss_normalizing_factor,
  )


def base_config(
    dropout_rate: Optional[float],
) -> fdl.Buildable[network.T5Config]:
  return fdl.Config(
      network.T5Config,
      # vocab size rounded to a multiple of 128 for TPU efficiency
      vocab_size=32128,
      dtype='bfloat16',
      emb_dim=768,
      num_heads=12,
      num_encoder_layers=12,
      num_decoder_layers=12,
      head_dim=64,
      mlp_dim=2048,
      mlp_activations=('gelu', 'linear'),
      dropout_rate=dropout_rate,
      logits_via_embedding=False,
  )


def small_config(
    dropout_rate: Optional[float],
) -> fdl.Buildable[network.T5Config]:
  config = base_config(dropout_rate=dropout_rate)
  return fdl.copy_with(
      config,
      emb_dim=512,
      num_heads=6,
      num_encoder_layers=8,
      num_decoder_layers=8,
      head_dim=64,
      mlp_dim=1024,
  )
