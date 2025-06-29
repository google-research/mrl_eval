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

"""Preprocessing the  Sentiment Analysis dataset and writing it to storage."""

from collections.abc import Callable, Mapping
import copy
import pathlib
from typing import Any, Union
from immutabledict import immutabledict
import tensorflow as tf
from mrl_eval.datasets import constants
from mrl_eval.datasets import dataset_lib
from mrl_eval.evaluation import metrics
from mrl_eval.utils import io_utils

RawExample = dict[str, Union[int, str]]
FeatureMap = dict[str, tf.train.Feature]
RawDataset = dataset_lib.RawDataset


class HeSentiment(dataset_lib.Dataset):
  """Implementation of the Dataset class for Amram Sentiment Analysis.

  This class transforms from raw dataset files into TensorFlow records.
  """

  SEED = 42
  HEBREW_LABEL_NAME = "sentiment"
  TEXT_NAME = "text"

  def __init__(self):
    super().__init__()
    self._en_to_he_label_mapping = {
        "Positive": "חיובי",
        "Negative": "שלילי",
        "Neutral": "ניטרלי",
    }

  @property
  def dataset_name(self):
    return constants.HESENTIMENT

  @property
  def raw_files(self):
    return {
        split: f"HebSentiment_{split}.jsonl"
        for split in ["train", "val", "test"]
    }

  @property
  def metrics(self):
    """Returns the metrics to be calculated for this dataset."""
    return [
        metrics.accuracy,
        metrics.get_macro_f1_fn(list(self._en_to_he_label_mapping.values())),
    ]

  def map_to_feature(self, ex):
    # Disabling pytype here as it can't infer correct type for the dict values.
    # pytype: disable=attribute-error
    feature = {
        "id": self.bytes_feature([ex["id"].encode()]),
        self.TEXT_NAME: self.bytes_feature([ex[self.TEXT_NAME].encode()]),
        self.HEBREW_LABEL_NAME: self.bytes_feature(
            [ex[self.HEBREW_LABEL_NAME].encode()]
        ),
    }
    # pytype: enable=attribute-error

    return feature

  def name_to_features(self):
    return immutabledict({
        "id": tf.io.FixedLenFeature([], tf.string),
        self.TEXT_NAME: tf.io.FixedLenFeature([], tf.string),
        self.HEBREW_LABEL_NAME: tf.io.FixedLenFeature([], tf.string),
    })

  def _translate_labels(self, raw_dataset):
    """Translates the sentiment labels from English to Hebrew.

    Since we aim to evaluate models in Hebrew, we should not expect them to
    generate answers in English.

    Args:
      raw_dataset: a list of raw examples.

    Returns:
      The dataset with the translated labels.
    """

    dataset_with_hebrew_labels = []
    for raw_example in raw_dataset:
      new_example = copy.deepcopy(raw_example)
      new_example[self.HEBREW_LABEL_NAME] = self._translate_sentiment(
          raw_example["tag_ids"]
      )
      dataset_with_hebrew_labels.append(new_example)
    return dataset_with_hebrew_labels

  def read_raw_data_file(self, file_path):
    return io_utils.read_jsonl(file_path)

  def process_raw_examples(self, dataset):
    return self._translate_labels(dataset)

  def _translate_sentiment(self, sentiment):
    sentiment_stripped = sentiment.strip()
    if sentiment_stripped not in self._en_to_he_label_mapping:
      raise ValueError(
          f"sentiment should be in {self._en_to_he_label_mapping.keys()}."
          f" Received {sentiment_stripped}"
      )
    return self._en_to_he_label_mapping[sentiment_stripped]

  def get_inputs(self, example):
    return str(example[self.TEXT_NAME])

  def get_outputs(self, example):
    return example[self.HEBREW_LABEL_NAME]

  def get_example_id(self, example):
    return example["id"]
