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
from absl import logging
from contextlib import contextmanager
import gin
import jax

try:
  from transformer_engine.common.recipe import DelayedScaling
  from transformer_engine.common.recipe import Format as FP8Format
  import transformer_engine.jax as te
  _IS_TRANSFORMER_ENGINE_INSTALLED = True

except ModuleNotFoundError as e:
  _IS_TRANSFORMER_ENGINE_INSTALLED = False


def _canonicalize_fp8_format(fp8_format):
  if not _IS_TRANSFORMER_ENGINE_INSTALLED:
    return None

  fp8_format = fp8_format.lower()
  if fp8_format in ['fp8_e4m3', 'fp8e4m3', 'e4m3']:
    return FP8Format.E4M3
  if fp8_format in ['fp8_e5m2', 'fp8e5m2', 'e5m2']:
    return FP8Format.E5M2
  if fp8_format in ['fp8_hybrid', 'fp8hybrid', 'hybrid']:
    return FP8Format.HYBRID
  raise ValueError('fp8_format must be one of [fp8_e4m3, fp8_e5m2, fp8_hybrid]'
    f'but the value is {fp8_format}')

@gin.configurable
class TransformerEngineConfig:
  def __init__(self, enabled=False, fp8_format='fp8_hybrid', margin=0., amax_history_len=1024):
    assert (_IS_TRANSFORMER_ENGINE_INSTALLED or (not enabled)), \
        'Attempt to run transformer engine FP8 without installing transformer_engine.'

    self.enabled = enabled
    self.fp8_format = _canonicalize_fp8_format(fp8_format)
    self.margin = margin
    self.amax_history_len = amax_history_len

  def __str__(self):
    return f"TransformerEngineConfig: enabled:{self.enabled}," \
           f" fp8_format: {self.fp8_format}, margin: {self.margin}," \
           f" amax_history_len: {self.amax_history_len}."


class TransformerEngineHelperBase:

  @staticmethod
  def is_fp8_enabled():
    raise NotImplementedError

  @staticmethod
  @contextmanager
  def fp8_autocast(te_config, dp_mesh_axis=None, tp_mesh_axis=None):
    raise NotImplementedError

  @staticmethod
  def extend_logical_axis_rules(rules):
    raise NotImplementedError

  @staticmethod
  def update_fp8_metas(grad_accum, flax_mutables):
    raise NotImplementedError

  @staticmethod
  def check_dataset_cfg(config):
    raise NotImplementedError

  @staticmethod
  def get_t5x_config(config):
    raise NotImplementedError

  @staticmethod
  def get_attn_mask(mask):
    raise NotImplementedError

  @staticmethod
  def get_encoder_layer(config, relative_embedding, name, original_cls):
    raise NotImplementedError

  @staticmethod
  def get_decoder_layer(config, relative_embedding, name, original_cls):
    raise NotImplementedError


class TENotInstalledHelper(TransformerEngineHelperBase):

  @staticmethod
  def is_fp8_enabled():
    return False

  @staticmethod
  @contextmanager
  def fp8_autocast(te_config, dp_mesh_axis=None, tp_mesh_axis=None):
    try:
      yield
    finally:
      pass

  @staticmethod
  def extend_logical_axis_rules(rules):
    return rules

  @staticmethod
  def update_fp8_metas(grad_accum, flax_mutables):
    return flax_mutables

  @staticmethod
  def check_dataset_cfg(config):
    pass

  @staticmethod
  def get_t5x_config(config):
    assert not config.transpose_batch_sequence, \
        "Only allow transpose_batch_sequence when Transformer Engine is installed."
    return config

  @staticmethod
  def get_attn_mask(mask):
    return mask

  @staticmethod
  def get_encoder_layer(config, relative_embedding, name, original_cls):
    return original_cls(config=config,
        relative_embedding=relative_embedding, name=name)

  @staticmethod
  def get_decoder_layer(config, relative_embedding, name, original_cls):
    return original_cls(config=config,
        relative_embedding=relative_embedding, name=name)


class TEInstalledHelper(TransformerEngineHelperBase):

  @staticmethod
  def is_fp8_enabled():
    return te.fp8.FP8Helper.is_fp8_enabled()

  @staticmethod
  @contextmanager
  def fp8_autocast(te_config, dp_mesh_axis="data", tp_mesh_axis="model"):
    delay_scaling = DelayedScaling(margin=te_config.margin,
                                   fp8_format=te_config.fp8_format,
                                   amax_history_len=te_config.amax_history_len,
                                   amax_compute_algo="max")
    try:
      with te.fp8_autocast(enabled=te_config.enabled, fp8_recipe=delay_scaling,
                           sharding_resource=te.ShardingResource(dp_mesh_axis, tp_mesh_axis)):
        yield
    finally:
        pass

  @staticmethod
  def extend_logical_axis_rules(rules):
    # Apply fp8_autocast to correctly set sharding_resource up.
    with TEInstalledHelper.fp8_autocast(TransformerEngineConfig()):
      return te.extend_logical_axis_rules(rules)

  @staticmethod
  def update_fp8_metas(grad_accum, flax_mutables):
    update_coll = te.update_collections(grad_accum, flax_mutables)
    # As the suggestion of FP8 training, updating FP8 scales as frequent as possible.
    update_coll = te.update_fp8_metas(update_coll)
    return update_coll

  @staticmethod
  def check_dataset_cfg(config):
    assert not config.pack, \
        "Transformer Engine does not support dataset.packing, please turn it off."

  @staticmethod
  def get_t5x_config(config):
    return config

  @staticmethod
  def get_attn_mask(mask):
    # Invert T5X's mask by 0->1, and 1->0
    mask_ = mask
    mask_ = 1 - mask_.astype(jax.numpy.uint8)
    return mask_

  @staticmethod
  def get_encoder_layer(config, relative_embedding, name, original_cls):
    hidden_dropout_dims = (-3,) if config.transpose_batch_sequence else(-2,)
    return te.TransformerLayer(
        hidden_size=config.num_heads*config.head_dim,
        mlp_hidden_size=config.mlp_dim,
        layernorm_type="rmsnorm",
        num_attention_heads=config.num_heads,
        hidden_dropout=config.dropout_rate,
        hidden_dropout_dims = hidden_dropout_dims,
        attention_dropout=config.dropout_rate,
        mlp_activations=config.mlp_activations,
        transpose_batch_sequence=config.transpose_batch_sequence,
        float32_attention_logits=config.float32_attention_logits,
        scale_attn_logits=config.scale_attn_logits,
        scaled_query_init=True,
        fuse_qkv_params=config.fuse_qkv_params,
        relative_embedding=relative_embedding,
        dtype=config.dtype, layer_type=te.TransformerLayerType.ENCODER, name=name)

  @staticmethod
  def get_decoder_layer(config, relative_embedding, name, original_cls):
    hidden_dropout_dims = (-3,) if config.transpose_batch_sequence else(-2,)
    return te.TransformerLayer(
        hidden_size=config.num_heads*config.head_dim,
        mlp_hidden_size=config.mlp_dim,
        layernorm_type="rmsnorm",
        num_attention_heads=config.num_heads,
        hidden_dropout=config.dropout_rate,
        hidden_dropout_dims = hidden_dropout_dims,
        attention_dropout=config.dropout_rate,
        mlp_activations=config.mlp_activations,
        transpose_batch_sequence=config.transpose_batch_sequence,
        float32_attention_logits=config.float32_attention_logits,
        scale_attn_logits=config.scale_attn_logits,
        scaled_query_init=True,
        fuse_qkv_params=config.fuse_qkv_params,
        relative_embedding=relative_embedding,
        dtype=config.dtype, layer_type=te.TransformerLayerType.DECODER, name=name)


class TransformerEngineHelper(TransformerEngineHelperBase):

  @staticmethod
  def get_helper():
    if _IS_TRANSFORMER_ENGINE_INSTALLED:
      return TEInstalledHelper
    return TENotInstalledHelper

  @staticmethod
  def is_fp8_enabled():
    return TransformerEngineHelper.get_helper().is_fp8_enabled()

  @staticmethod
  @contextmanager
  def fp8_autocast(te_config, dp_mesh_axis="data", tp_mesh_axis="model"):
    try:
      with TransformerEngineHelper.get_helper().fp8_autocast(te_config, dp_mesh_axis, tp_mesh_axis):
        yield
    finally:
        pass

  @staticmethod
  def extend_logical_axis_rules(rules):
    return TransformerEngineHelper.get_helper().extend_logical_axis_rules(rules)

  @staticmethod
  def update_fp8_metas(grad_accum, flax_mutables):
    return TransformerEngineHelper.get_helper().update_fp8_metas(grad_accum, flax_mutables)

  @staticmethod
  def check_dataset_cfg(config):
    return TransformerEngineHelper.get_helper().check_dataset_cfg(config)

  @staticmethod
  def get_t5x_config(config):
    return TransformerEngineHelper.get_helper().get_t5x_config(config)

  @staticmethod
  def get_attn_mask(mask):
    return TransformerEngineHelper.get_helper().get_attn_mask(mask)

  @staticmethod
  def get_encoder_layer(config, relative_embedding, name, original_cls):
    return TransformerEngineHelper.get_helper().get_encoder_layer(config, relative_embedding, name, original_cls)

  @staticmethod
  def get_decoder_layer(config, relative_embedding, name, original_cls):
    return TransformerEngineHelper.get_helper().get_decoder_layer(config, relative_embedding, name, original_cls)
