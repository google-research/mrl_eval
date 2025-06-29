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
raw_data_dir=mrl_eval_data

echo "NEMO"
for split in "train" "dev" "test"; do
  for level in "morph" "token-single"; do
      wget -P ${raw_data_dir}/nemo https://github.com/OnlpLab/NEMO-Corpus/raw/main/data/spmrl/gold/${level}_gold_${split}.bmes
  done
done

echo "HeQ"
mkdir -p ${raw_data_dir}/heq
mkdir -p ${raw_data_dir}/heq_question_gen

for split in "train" "val" "test"; do
  wget -O "${raw_data_dir}/heq/${split}.json" "https://github.com/NNLP-IL/Hebrew-Question-Answering-Dataset/raw/main/data/data%20v1.1/${split}%20v1.1.json"
  cp ${raw_data_dir}/heq/${split}.json ${raw_data_dir}/heq_question_gen/${split}.json
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

echo "ArTyDiQA"
mkdir -p ${raw_data_dir}/artydiqa
mkdir -p ${raw_data_dir}/artydiqa_question_gen
wget -P ${raw_data_dir}/artydiqa https://github.com/google-research-datasets/artydiqa/raw/main/qa.zip
unzip -j ${raw_data_dir}/artydiqa/qa.zip -d ${raw_data_dir}/artydiqa
wget -P ${raw_data_dir}/artydiqa https://github.com/google-research-datasets/artydiqa/raw/main/qg.zip
unzip -j ${raw_data_dir}/artydiqa/qg.zip -d ${raw_data_dir}/artydiqa_question_gen

echo "ArQ"
mkdir -p ${raw_data_dir}/arq_MSA
mkdir -p ${raw_data_dir}/arq_spoken
mkdir -p ${raw_data_dir}/arq_MSA_question_gen
mkdir -p ${raw_data_dir}/arq_spoken_question_gen

for variant in "spoken" "MSA"; do
  for split in "train" "val" "test"; do
    wget -P ${raw_data_dir}/arq_${variant} https://huggingface.co/datasets/HebArabNlpProject/ArQ/resolve/main/${variant}_${split}.json
    # cp to question gen directory
    cp ${raw_data_dir}/arq_${variant}/${variant}_${split}.json ${raw_data_dir}/arq_${variant}_question_gen/${variant}_${split}.json
  done
done

echo "HebCo"
wget -P ${raw_data_dir}/hebco https://github.com/IAHLT/coref/raw/refs/heads/master/train_val_test/coref-5-heb_train.jsonl
wget -P ${raw_data_dir}/hebco https://github.com/IAHLT/coref/raw/refs/heads/master/train_val_test/coref-5-heb_test.jsonl
wget -P ${raw_data_dir}/hebco https://github.com/IAHLT/coref/raw/refs/heads/master/train_val_test/coref-5-heb_val.jsonl

echo "HebSummaries"
mkdir -p ${raw_data_dir}/hebsummaries
for split in "train" "val" "test"; do
  wget -P ${raw_data_dir}/hebsummaries https://huggingface.co/datasets/HebArabNlpProject/HebSummaries/resolve/main/${split}.jsonl
done

echo "IahltNER"
mkdir -p ${raw_data_dir}/iahlt_ner
for split in "train" "val" "test"; do
  wget -P ${raw_data_dir}/iahlt_ner https://huggingface.co/datasets/HebArabNlpProject/arabic-iahlt-NER/resolve/main/iahlt_ner_${split}.jsonl
done

echo "ArCoref"
mkdir -p ${raw_data_dir}/arcoref
for split in "train" "val" "test"; do
  wget -P ${raw_data_dir}/arcoref https://huggingface.co/datasets/HebArabNlpProject/ArabCoRef/resolve/main/arcoref_${split}.jsonl
done

echo "ArSentiment"
mkdir -p ${raw_data_dir}/arsentiment
for split in "train" "val" "test"; do
  wget -P ${raw_data_dir}/arsentiment https://huggingface.co/datasets/HebArabNlpProject/ArabicSentimentDataSet/resolve/main/arsentiment_${split}.jsonl
done

echo "ArXLSum"
wget -P ${raw_data_dir}/ar_xlsum https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/arabic_XLSum_v2.0.tar.bz2
tar -xvf ${raw_data_dir}/ar_xlsum/arabic_XLSum_v2.0.tar.bz2 -C ${raw_data_dir}/ar_xlsum
echo "Done"

echo "ArabicNLI"
for split in "train" "validation" "test"; do
  wget -P ${raw_data_dir}/arabic_nli https://huggingface.co/datasets/facebook/xnli/resolve/main/ar/${split}-00000-of-00001.parquet
done