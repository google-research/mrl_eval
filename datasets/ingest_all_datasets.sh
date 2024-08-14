# Copyright 2024 The Google Research Authors.
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
datasets=("nemo" "heq" "hesum" "hesentiment" "hebnli")
save_tfrecord=$1


for dataset in "${datasets[@]}"; do 
    echo "${dataset}"
    if [[ "$save_tfrecord" = "save_tfrecord" ]]; then
        python -m mrl_eval.datasets.ingest_dataset --dataset "$dataset" --save_tfrecord
    else
        python -m mrl_eval.datasets.ingest_dataset --dataset "$dataset"
    fi

    if [[ $? -ne 0 ]]; then 
        echo "Error occurred while ingesting dataset: $dataset"
        exit 1 
    fi
done
