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
raw_data_dir=mrl_eval_data

echo "NEMO"
for split in "train" "dev" "test"; do
  for level in "morph" "token-single"; do
      wget -P ${raw_data_dir}/nemo https://github.com/OnlpLab/NEMO-Corpus/raw/main/data/spmrl/gold/${level}_gold_${split}.bmes
  done
done

echo "HeQ"
for split in "train" "val" "test"; do
  wget -P ${raw_data_dir}/heq https://github.com/NNLP-IL/Hebrew-Question-Answering-Dataset/raw/main/data/${split}.json
done

echo "HebNLI"
for split in "train" "val" "test"; do
  wget -P ${raw_data_dir}/hebnli https://huggingface.co/datasets/HebArabNlpProject/HebNLI/resolve/main/HebNLI_${split}.jsonl
done

echo "HeSentiment"
for split in "train" "val" "test"; do
  wget -P ${raw_data_dir}/hesentiment https://huggingface.co/datasets/HebArabNlpProject/HebrewSentiment/resolve/main/HebSentiment_${split}.jsonl
done

echo "HeSum"
wget -P ${raw_data_dir}/hesum https://github.com/OnlpLab/HeSum/raw/main/data/train/train.csv
wget -P ${raw_data_dir}/hesum https://github.com/OnlpLab/HeSum/raw/main/data/dev/validation.csv
wget -P ${raw_data_dir}/hesum https://github.com/OnlpLab/HeSum/raw/main/data/test/test.csv
echo "Done"