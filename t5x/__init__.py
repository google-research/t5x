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

"""Import API modules."""

import t5x.adafactor
import t5x.checkpoints
import t5x.decoding
import t5x.gin_utils
import t5x.losses
import t5x.models
import t5x.partitioning
import t5x.state_utils
import t5x.train_state
import t5x.trainer
import t5x.utils

# Version number.
from t5x.version import __version__

# TODO(adarob): Move clients to t5x.checkpointing and rename
# checkpoints.py to checkpointing.py
checkpointing = t5x.checkpoints
