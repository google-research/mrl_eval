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

"""Preprocessing the ArQ dataset and writing it to storage."""

from collections.abc import Mapping, Sequence
import functools
import pathlib
import re
from typing import Any, Union

import immutabledict
import tensorflow as tf

from mrl_eval.datasets import constants
from mrl_eval.datasets import dataset_lib
from mrl_eval.evaluation import metrics
from mrl_eval.utils import io_utils


TextField = str
AnswerPositionField = int

# ArQ format
AnswerField = dict[str, Union[list[TextField], list[AnswerPositionField]]]
RawExample = dict[str, Union[TextField, AnswerField]]
FeatureMap = dict[str, Union[tf.train.Feature, list[tf.train.Feature]]]
RawDataset = dataset_lib.RawDataset

NULL_ANSWER_TEXT = "لا يمكن الإجابة على السؤال من النص."
NULL_ANSWER_TEXT_START = -1


class ArQ(dataset_lib.Dataset):
  """Implementation of dataset_lib.Dataset for ArQ.

  This class implements transformation from raw ArQ dataset files
  (https://huggingface.co/datasets/HebArabNlpProject/ArQ) into tensorflow
  records.
  """

  def __init__(self, variant = "spoken"):
    if variant not in ["spoken", "MSA"]:
      raise ValueError(
          f"Unsupported ArQ variant: {variant}. Supported variants are 'spoken'"
          " and 'MSA'."
      )
    self._variant = variant

  @property
  def dataset_name(self):
    return (
        constants.ARQ_SPOKEN if self._variant == "spoken" else constants.ARQ_MSA
    )

  @property
  def raw_files(self):
    return {
        "train": f"{self._variant}_train.json",
        "val": f"{self._variant}_val.json",
        "test": f"{self._variant}_test.json",
    }

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
        "context": self.bytes_feature([ex["context"].encode()]),
        "title": self.bytes_feature([ex["title"].encode()]),
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
        "question": tf.io.FixedLenFeature([], tf.string),
        "context": tf.io.FixedLenFeature([], tf.string),
        "title": tf.io.FixedLenFeature([], tf.string),
        "answers/text": tf.io.FixedLenSequenceFeature(
            [], tf.string, allow_missing=True
        ),
        "answers/answer_start": tf.io.FixedLenSequenceFeature(
            [], tf.int64, allow_missing=True
        ),
    })

  def _get_answer_candidates(self, raw_example):
    """Returns answer annotations from all available contexts."""
    if not raw_example["answerable question"]:
      # Example cannot be answered. Return NULL answer.
      return {
          "text": [NULL_ANSWER_TEXT],
          "answer_start": [NULL_ANSWER_TEXT_START],
      }
    answers = []
    answers_start = []
    for context_key in ["context", "context 2", "context 3", "context 4"]:
      context = raw_example[context_key]
      if context:
        search_result = re.search(r".*(\*([^\*]*)\*).*", context, re.DOTALL)
        if search_result is None:
          raise ValueError(f"Could not extract answer from context={context}")
        answers.append(search_result.group(2))
        answers_start.append(search_result.start(1))
    return {"text": answers, "answer_start": answers_start}

  def _process_raw_example(self, raw_example):
    return {
        "id": raw_example["id_question"],
        "context": raw_example["context_clean"],
        "title": "",  # For compatiblility with SQuAD format.
        "question": raw_example["question"],
        "answers": self._get_answer_candidates(raw_example),
    }

  def process_raw_examples(self, dataset):
    """Extracts relevant fields and put them in SQuAD format."""

    # For question generation task, we skip unanswerable examples.
    skip_unanswerable = self.dataset_name in (
        constants.ARQ_SPOKEN_QUESTION_GEN,
        constants.ARQ_MSA_QUESTION_GEN,
    )

    processed_data = []
    for raw_example in dataset:
      # Ignore rejected examples.
      if raw_example["q quality"] == "rejected":
        continue
      if skip_unanswerable and not raw_example["answerable question"]:
        continue
      processed_data.append(self._process_raw_example(raw_example))

    return processed_data

  def read_raw_data_file(self, file_path):
    return io_utils.read_json(file_path)

  def get_inputs(self, example):
    return (
        "النص: "
        + example["context"]
        + "\n"
        + "السؤال: "
        + example["question"]
    )

  def get_outputs(self, example):
    return example["answers"]["text"]

  def get_example_id(self, example):
    return example["id"]


class ArQQuestionGen(ArQ):
  """Implementation of dataset_lib.Dataset for ArQ for the question generation task."""

  @property
  def dataset_name(self):
    return (
        constants.ARQ_SPOKEN_QUESTION_GEN
        if self._variant == "spoken"
        else constants.ARQ_MSA_QUESTION_GEN
    )

  @property
  def metrics(self):
    return [metrics.rouge]

  @property
  def dataset_dir(self):
    """Returns the directory of the raw files.

    The ArQ question generation task uses the ArQ QA task.
    """
    return pathlib.Path(constants.BASE_PATH) / self.dataset_name

  def get_inputs(self, example):
    return (
        "النص: "
        + str(example["context"])
        + "\n"
        + "الجواب: "
        + str(self._get_answer_candidates(example)["text"])
    )

  def get_outputs(self, example):
    return example["question"]
