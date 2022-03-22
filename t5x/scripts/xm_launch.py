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

r"""XManager launcher for t5x.

Read about XManager:
https://github.com/deepmind/xmanager

Usage:
xmanager xm_launch.py -- \
  --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin" \
  --model_dir=gs://$GOOGLE_CLOUD_BUCKET_NAME/t5x/$(date +%Y%m%d) \
  --tfds_data_dir=gs://$GOOGLE_CLOUD_BUCKET_NAME/t5x/data
"""

import collections
import os
import sys
import tempfile
from typing import Any, Dict

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_local
from xmanager.contrib import copybara

_CLONE_GITHUB = flags.DEFINE_bool(
    'clone_github',
    False,
    'If True, clone t5x/ from GitHub. Otherwise, use the local version.',
)
_COPYBARA_CONFIG = flags.DEFINE_string(
    'copybara_config',
    None,
    'Copybara config to use. See https://github.com/google/copybara '
    'If None, the local t5x directory will be copied with no modifications.',
)
_COPYBARA_WORKFLOW = flags.DEFINE_string(
    'copybara_workflow',
    'local',
    'Copybara workflow to apply with --copybara_config',
)
_COPYBARA_ORIGIN = flags.DEFINE_string(
    'copybara_origin',
    '..',
    'Copybara origin folder to apply with --copybara_config',
)

_TPU_CORES = flags.DEFINE_integer(
    'tpu_cores',
    8,
    'Number of TPU cores to run. There will be a new worker every 8 cores. '
    'TPU types: https://cloud.google.com/tpu/docs/types-zones#types',
)
_MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    None,
    'Model dir to save logs, ckpts, etc. in "gs://model_dir" format.',
)
_TFDS_DATA_DIR = flags.DEFINE_string(
    'tfds_data_dir',
    None,
    'Data dir to save the processed dataset in "gs://data_dir" format.',
)
_SEQIO_CACHE_DIRS = flags.DEFINE_list(
    'seqio_additional_cache_dirs',
    [],
    'Comma separated directories in "gs://cache_dir" format to search for cached Tasks in addition to defaults.',
)


@xm.run_in_asyncio_loop
async def main(_, gin_args: Dict[str, Any]):
  name = 't5x'
  async with xm_local.create_experiment(experiment_title=name) as experiment:
    # TODO(chenandrew) Vertex Tensorboard is not supported for TPUs.
    # https://github.com/deepmind/xmanager/issues/11
    # vertex = xm_local.vertex_client()
    # tensorboard_name = await vertex.get_or_create_tensorboard(name)
    # tensorboard = xm_local.TensorboardCapability(
    #     name=tensorboard_name,
    #     base_output_directory=_MODEL_DIR.value)
    tensorboard = None
    executor = xm_local.Vertex(
        requirements=xm.JobRequirements(tpu_v2=_TPU_CORES.value),
        tensorboard=tensorboard,
    )

    # The t5x/ root directory.
    path = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
    if _COPYBARA_CONFIG.value:
      copybara_destination = os.path.join(tempfile.mkdtemp(), 't5x')
      path = copybara.run_workflow(_COPYBARA_CONFIG.value,
                                   _COPYBARA_WORKFLOW.value,
                                   _COPYBARA_ORIGIN.value, copybara_destination)

    if _CLONE_GITHUB.value:
      copy_t5x = [
          'RUN git clone --branch=main https://github.com/google-research/t5x',
      ]
    else:
      copy_t5x = [f'COPY {os.path.basename(path)}/ t5x']

    [executable] = experiment.package([
        xm.python_container(
            executor.Spec(),
            path=path,
            base_image='gcr.io/deeplearning-platform-release/base-cpu',
            docker_instructions=[
                *copy_t5x,
                'WORKDIR t5x',
                'RUN python3 -m pip install -e ".[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html',
            ],
            entrypoint=xm.CommandList([
                f'export MODEL_DIR=\'"{_MODEL_DIR.value}/logs"\'',
                f'export TFDS_DATA_DIR={_TFDS_DATA_DIR.value}',
                'export SEQIO_CACHE_DIRS={}'.format(','.join(
                    _SEQIO_CACHE_DIRS.value)),
                'export T5X_DIR=.',
                ('python3 ${T5X_DIR}/t5x/train.py '
                 '--gin.MODEL_DIR=${MODEL_DIR} '
                 '--tfds_data_dir=${TFDS_DATA_DIR} '
                 '--seqio_additional_cache_dirs=${SEQIO_CACHE_DIRS}'),
            ]),
        ),
    ])
    args = []
    for k, l in gin_args.items():
      for v in l:
        args.append(f'--{k}={v}')
    experiment.add(xm.Job(executable=executable, executor=executor, args=args))


def _split_gin_args(argv, prefix='--gin'):
  """Separates absl and gin args into separate lists."""
  other_args = [argv[0]]
  gin_args = collections.defaultdict(list)
  for arg in argv[1:]:
    if arg.startswith(prefix):
      k, v = arg[len('--'):].split('=', maxsplit=1)
      gin_args[k].append(v)
    else:
      other_args.append(arg)
  return other_args, gin_args


if __name__ == '__main__':
  _other_args, _gin_args = _split_gin_args(sys.argv)
  app.run(lambda argv: main(argv, _gin_args), _other_args)
