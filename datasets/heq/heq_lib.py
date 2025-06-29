# coding=utf-8
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

"""Preprocessing the HeQ dataset and writing it to storage."""

from collections.abc import Mapping, Sequence
import functools
import pathlib
from typing import Any, Union

import immutabledict
import tensorflow as tf

from mrl_eval.datasets import constants
from mrl_eval.datasets import dataset_lib
from mrl_eval.evaluation import metrics
from mrl_eval.utils import io_utils

TextField = str
AnswerPositionField = int

# HeQ format
AnswerField = dict[str, Union[list[TextField], list[AnswerPositionField]]]
RawExample = dict[str, Union[TextField, AnswerField]]
FeatureMap = dict[str, Union[tf.train.Feature, list[tf.train.Feature]]]
RawDataset = dataset_lib.RawDataset

NULL_ANSWER_TEXT = "לא ניתן לענות על השאלה על סמך ההקשר."
NULL_ANSWER_TEXT_START = -1


class HeQ(dataset_lib.Dataset):
  """Implementation of dataset_lib.Dataset for HeQ.

  This class implements transformation from raw dataset files
  (https://github.com/NNLP-IL/Hebrew-Question-Answering-Dataset) into
  tensforflow records.
  """

  @property
  def dataset_name(self):
    return constants.HEQ

  @property
  def raw_files(self):
    return {"train": "train.json", "val": "val.json", "test": "test.json"}

  @property
  def metrics(self):
    """Returns the metrics to be calculated for this dataset."""
    return [
        metrics.em,
        metrics.f1,
        functools.partial(metrics.tlnls, null_answer_text=NULL_ANSWER_TEXT),
    ]

  def map_to_feature(self, ex):
    """mapping RawExample into a FeatureMap as expected for TFRecord."""

    # Disabling pytype here as it can't infer correct type for the dict values.
    # pytype: disable=attribute-error

    feature = {
        "id": self.bytes_feature([ex["id"].encode()]),
        "title": self.bytes_feature([ex["title"].encode()]),
        "context": self.bytes_feature([ex["context"].encode()]),
        "question": self.bytes_feature([ex["question"].encode()]),
        "answers/text": self.bytes_feature(
            [text.encode() for text in ex["answers"]["text"]]
        ),
        "answers/answer_start": self.int64_feature(
            ex["answers"]["answer_start"]
        ),
    }
    # pytype: enable=attribute-error

    return feature

  def name_to_features(self):
    return immutabledict.immutabledict({
        "id": tf.io.FixedLenFeature([], tf.string),
        "title": tf.io.FixedLenFeature([], tf.string),
        "question": tf.io.FixedLenFeature([], tf.string),
        "context": tf.io.FixedLenFeature([], tf.string),
        "answers/text": tf.io.FixedLenSequenceFeature(
            [], tf.string, allow_missing=True
        ),
        "answers/answer_start": tf.io.FixedLenSequenceFeature(
            [], tf.int64, allow_missing=True
        ),
    })

  def _flatten_raw_data(self, raw_data):

    skip_impossible = self.dataset_name == constants.HEQ_QUESTION_GEN

    flatten_data = []
    for doc in raw_data:
      for paragraph in doc["paragraphs"]:
        for qa in paragraph["qas"]:

          if qa["is_impossible"]:
            if skip_impossible:
              continue

            else:
              answer_texts = [NULL_ANSWER_TEXT]
              answer_texts_start = [NULL_ANSWER_TEXT_START]

          else:
            answer_texts = [ans["text"] for ans in qa["answers"]]

            # the raw dataset files contain both answer_start and answer.start
            answer_start_key = (
                "answer_start"
                if "answer_start" in qa["answers"][0]
                else "answer.start"
            )

            answer_texts_start = [
                ans[answer_start_key] for ans in qa["answers"]
            ]

          flatten_data.append({
              "id": qa["id"],
              "title": doc["title"],
              "context": paragraph["context"],
              "question": qa["question"],
              "answers": {
                  "text": answer_texts,
                  "answer_start": answer_texts_start,
              },
          })

    return flatten_data

  def process_raw_examples(self, dataset):
    """Flatten the raw data to a list of examples."""
    return self._flatten_raw_data(dataset)

  def read_raw_data_file(self, file_path):
    return io_utils.read_json(file_path)["data"]

  def get_inputs(self, example):
    return (
        "הקשר: "
        + example["context"]
        + "\n"
        + "שאלה: "
        + example["question"]
    )

  def get_outputs(self, example):
    return example["answers"]["text"]

  def get_example_id(self, example):
    return example["id"]


class HeQQuestionGen(HeQ):
  """Implementation of dataset_lib.Dataset for HeQ for the question generation task."""

  @property
  def dataset_name(self):
    return constants.HEQ_QUESTION_GEN

  @property
  def metrics(self):
    return [metrics.rouge]

  def get_inputs(self, example):
    return (
        "הקשר: "
        + example["context"]
        + "\n"
        + "תשובה: "
        + example["answers"]["text"][0]
    )

  def get_outputs(self, example):
    return str(example["question"])

  @property
  def dataset_dir(self):
    """Returns the directory of the raw files.

    The HeQ question generation task uses the HeQ QA task.
    """
    return pathlib.Path(constants.BASE_PATH) / constants.HEQ_QUESTION_GEN
