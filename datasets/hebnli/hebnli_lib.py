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

"""Preprocessing the NLI dataset and writing it to storage."""

from collections.abc import Mapping, Sequence
import copy
import pathlib
from typing import Any, Union

import immutabledict
import tensorflow as tf

from mrl_eval.datasets import constants
from mrl_eval.datasets import dataset_lib
from mrl_eval.evaluation import metrics
from mrl_eval.utils import io_utils


RawExample = dict[str, Union[int, str]]
FeatureMap = dict[str, tf.train.Feature]
RawDataset = dataset_lib.RawDataset

_EN_TO_HE_LABEL_MAPPING = immutabledict.immutabledict(
    {"entailment": "היסק", "contradiction": "סתירה", "neutral": "ניטרלי"}
)


class HebNLI(dataset_lib.Dataset):
  """Implementation of the Dataset class for HebNLI.

  This class transforms from raw dataset files into TensorFlow records.
  """

  HEB_LABEL_NAME = "label_in_hebrew"
  HEB_FIRST_SENT_NAME = "heb_sent_1"
  HEB_SECOND_SENT_NAME = "heb_sent_2"

  @property
  def dataset_name(self):
    return constants.HEBNLI

  @property
  def raw_files(self):
    return {
        split: f"HebNLI_{split}.jsonl" for split in ["train", "val", "test"]
    }

  @property
  def metrics(self):
    """Returns the metrics to be calculated for this dataset."""
    return [
        metrics.accuracy,
        metrics.get_macro_f1_fn(list(_EN_TO_HE_LABEL_MAPPING.values())),
    ]

  def map_to_feature(self, ex):
    # Disabling pytype here as it can't infer correct type for the dict values.
    # pytype: disable=attribute-error
    feature = {
        "id": self.bytes_feature([ex["pairID"].encode()]),
        self.HEB_FIRST_SENT_NAME: self.bytes_feature(
            [ex["translation1"].encode()]
        ),
        self.HEB_SECOND_SENT_NAME: self.bytes_feature(
            [ex["translation2"].encode()]
        ),
        self.HEB_LABEL_NAME: self.bytes_feature(
            [ex[self.HEB_LABEL_NAME].encode()]
        ),
    }
    # pytype: enable=attribute-error

    return feature

  def name_to_features(self):
    return immutabledict.immutabledict({
        "id": tf.io.FixedLenFeature([], tf.string),
        self.HEB_FIRST_SENT_NAME: tf.io.FixedLenFeature([], tf.string),
        self.HEB_SECOND_SENT_NAME: tf.io.FixedLenFeature([], tf.string),
        self.HEB_LABEL_NAME: tf.io.FixedLenFeature([], tf.string),
    })

  def read_raw_data_file(self, file_path):
    return io_utils.read_jsonl(file_path)

  def process_raw_examples(self, dataset):

    processed_dataset = []
    for raw_example in dataset:
      example = copy.deepcopy(raw_example)
      example["id"] = raw_example["pairID"]

      # The test file containes example that are labeld in their hebrew version
      # 'hebrew_label', the train and val files we use 'original_label'.
      label_key = (
          "hebrew_label" if "hebrew_label" in raw_example else "original_label"
      )
      label = raw_example[label_key]
      if label == "-":  # Tie in annotations, discarding example.
        continue
      example[self.HEB_LABEL_NAME] = self._translate_label(
          raw_example[label_key]
      )

      processed_dataset.append(example)
    return processed_dataset

  def get_inputs(self, example):
    return (
        "משפט 1:"
        + example["translation1"]
        + "\n"
        + "משפט 2:"
        + example["translation2"]
    )

  def get_outputs(self, example):
    return example[self.HEB_LABEL_NAME]

  def get_example_id(self, example):
    return example["id"]

  def _translate_label(self, label):
    label_stripped = label.strip()
    if label_stripped not in _EN_TO_HE_LABEL_MAPPING:
      raise ValueError(
          f"label should be in {_EN_TO_HE_LABEL_MAPPING.keys()}."
          f" Received {label_stripped}"
      )
    return _EN_TO_HE_LABEL_MAPPING[label_stripped]
