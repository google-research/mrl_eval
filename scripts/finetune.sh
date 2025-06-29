# Copyright 2025 The Google Research Authors.
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

#!/bin/bash
set -e

HOME=$1 # the full path prefix to the project directory (THIS_PATH/mrl_eval)
T5X_DIR=$2 # path to the cloned t5x repo
TFDS_DATA_DIR=$3 # e.g. "gs://my_bucket/my_data"
MODEL_DIR=$4 # e.g. "gs://my_bucket/my_model"
GIN_PATH_IN_PROJECT=$5 # ft config, e.g."models/gin/finetune_mt5_xl_hesentiment.gin"

PROJECT_DIR=${HOME}

export PYTHONPATH=${PROJECT_DIR}
echo --gin_search_paths="${PROJECT_DIR}" \
  --gin_file="${PROJECT_DIR}"/"${GIN_PATH_IN_PROJECT}" \
  --gin.MODEL_DIR="${MODEL_DIR}" \
  --tfds_data_dir="${TFDS_DATA_DIR}"
  
python3 "${T5X_DIR}"/t5x/train.py \
  --gin_search_paths="${PROJECT_DIR}" \
  --gin_file="${PROJECT_DIR}"/"${GIN_PATH_IN_PROJECT}" \
  --gin.MODEL_DIR="${MODEL_DIR}" \
  --tfds_data_dir="${TFDS_DATA_DIR}"