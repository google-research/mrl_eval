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
name=$1
gin_path_in_project=$2
ckpt=$3

bucket_name=$GOOGLE_CLOUD_BUCKET_NAME
t5x_root=$T5X_DIR
root_for_project_dir=$PROJECT_DIR_ROOT

data_dir=gs://$bucket_name/t5x/data
model_dir=gs://$bucket_name/t5x/"${name}"/$(date +%Y%m%d)

export MODEL_DIR=$model_dir
export TFDS_DATA_DIR=$data_dir

python3 "${t5x_root}"/t5x/scripts/xm_launch.py \
  --gin_file="${gin_path_in_project}" \
  --tfds_data_dir="${TFDS_DATA_DIR}" \
  --run_mode="eval" \
  --name="${name}" \
  --model_dir="${model_dir}" \
  --project_dirs="${root_for_project_dir}"/"mrl_eval","${root_for_project_dir}"/"mrl_eval_data" \
  --pip_install="rouge_score","immutabledict","Levenshtein","numpy","pandas","scikit-learn" \
  --gin.EVAL_OUTPUT_DIR="'gs://$GOOGLE_CLOUD_BUCKET_NAME/t5x/${name}/eval/${ckpt}'" \
  --gin.CHECKPOINT_PATH="'${ckpt}'" \
