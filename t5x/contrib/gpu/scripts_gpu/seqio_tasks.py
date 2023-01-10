# Copyright (c) 2022-2023 The T5x Authors
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
"""Add Tasks to registry."""
# TODO(adarob): Switch to seqio.Task.

import functools

import seqio
import t5.data
from t5.data import postprocessors
from t5.data import preprocessors
from t5.data.glue_utils import get_glue_metric
from t5.data.glue_utils import get_glue_postprocess_fn
from t5.data.glue_utils import get_glue_text_preprocessor
from t5.data.glue_utils import get_super_glue_metric
from t5.evaluation import metrics
import tensorflow_datasets as tfds
import sys

import t5x.contrib.gpu.scripts_gpu.tfds_pile

TaskRegistry = seqio.TaskRegistry

DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(vocabulary=t5.data.get_default_vocabulary(),
                      add_eos=True,
                      required=False),
    "targets":
        seqio.Feature(vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}

# ================================== The Pile ====================================
# NVIDIA JAX T5 pretraining
TaskRegistry.add("the_pile_span_corruption",
                 source=seqio.TfdsDataSource(tfds_name="ThePile:1.0.0"),
                 preprocessors=[
                     functools.partial(preprocessors.rekey,
                                       key_map={
                                           "inputs": None,
                                           "targets": "text"
                                       }),
                     seqio.preprocessors.tokenize,
                     seqio.CacheDatasetPlaceholder(),
                     preprocessors.span_corruption,
                     seqio.preprocessors.append_eos_after_trim,
                 ],
                 output_features=DEFAULT_OUTPUT_FEATURES,
                 metric_fns=[])

# =================================== GLUE =====================================
for b in tfds.text.glue.Glue.builder_configs.values():
  TaskRegistry.add("glue_%s_v2" % b.name,
                   source=seqio.TfdsDataSource(
                       tfds_name="glue/%s:2.0.0" % b.name,
                       splits=["test"] if b.name == "ax" else None),
                   preprocessors=[
                       get_glue_text_preprocessor(b),
                       seqio.preprocessors.tokenize,
                       seqio.CacheDatasetPlaceholder(),
                       seqio.preprocessors.append_eos_after_trim,
                   ],
                   metric_fns=get_glue_metric(b.name),
                   output_features=DEFAULT_OUTPUT_FEATURES,
                   postprocess_fn=get_glue_postprocess_fn(b))
