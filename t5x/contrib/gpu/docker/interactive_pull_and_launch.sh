#!/bin/bash

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

set -x

CONTAINER=${1}
echo $CONTAINER
docker pull $CONTAINER

DATASET_PATH=${2} 

## !! Uncomment this to add a custom path to workspace dir !!##
## By default `.../T5X/t5x/workspace` is selected
# WORKSPACE_PATH=<ADD CUSTOM PATH TO `workspace` dir>

nvidia-docker run -ti --net=host --ipc=host -v ${PWD}:/t5x_home -v ${DATASET_PATH}:/t5x_home/datasets -v ${WORKSPACE_PATH:-${PWD}/workspace}:/t5x_home/workspace --privileged $CONTAINER /bin/bash
set +x
