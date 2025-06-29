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

from collections.abc import Callable, Mapping, Sequence
import copy
import pathlib
from typing import Any, Union
import immutabledict
import pandas as pd
import tensorflow as tf

from mrl_eval.datasets import constants
from mrl_eval.datasets import dataset_lib
from mrl_eval.evaluation import metrics
from mrl_eval.utils import io_utils

RawExample = dict[str, Union[int, str]]
FeatureMap = dict[str, tf.train.Feature]
RawDataset = dataset_lib.RawDataset


class ArSentiment(dataset_lib.Dataset):
  """Implementation of the Dataset class for Arabic Sentiment Analysis."""

  LABEL_KEY = "sentiment"
  TEXT_KEY = "text"

  def __init__(self):
    super().__init__()
    self._en_to_ar_label_name = {
        "Positive": "إيجابي",
        "Negative": "سلبي",
        "Neutral": "محايد",
        "Complex": "معقد",
    }

  @property
  def dataset_name(self):
    return constants.ARSENTIMENT

  @property
  def raw_files(self):
    return {
        split: f"arsentiment_{split}.jsonl"
        for split in ["train", "val", "test"]
    }

  @property
  def metrics(self):
    """Returns the metrics to be calculated for this dataset."""
    return [
        metrics.accuracy,
        metrics.get_macro_f1_fn(list(self._en_to_ar_label_name.values())),
    ]

  def map_to_feature(self, ex):
    # Disabling pytype here as it can't infer correct type for the dict values.
    # pytype: disable=attribute-error
    feature = {
        "id": self.bytes_feature([str(ex["id"]).encode()]),
        self.TEXT_KEY: self.bytes_feature([ex[self.TEXT_KEY].encode()]),
        self.LABEL_KEY: self.bytes_feature([ex[self.LABEL_KEY].encode()]),
    }
    # pytype: enable=attribute-error

    return feature

  def name_to_features(self):
    return immutabledict.immutabledict({
        "id": tf.io.FixedLenFeature([], tf.string),
        self.TEXT_KEY: tf.io.FixedLenFeature([], tf.string),
        self.LABEL_KEY: tf.io.FixedLenFeature([], tf.string),
    })

  def _translate_labels(self, raw_dataset):
    """Translates the sentiment labels from English to Arabic, dropping items with null labels.

    Since we aim to evaluate models in Arabic, we should not expect them to
    generate answers in Arabic.

    Args:
      raw_dataset: a list of raw examples.

    Returns:
      The dataset with the translated labels.
    """

    dataset_with_arabic_labels = []
    for raw_example in raw_dataset:
      if pd.isna(raw_example[self.LABEL_KEY]):
        continue

      new_example = copy.deepcopy(raw_example)
      new_example[self.LABEL_KEY] = self._translate_sentiment(
          raw_example["sentiment"]
      )
      dataset_with_arabic_labels.append(new_example)
    return dataset_with_arabic_labels

  def read_raw_data_file(self, file_path):
    return io_utils.read_jsonl(file_path)

  def process_raw_examples(self, dataset):
    return self._translate_labels(dataset)

  def _translate_sentiment(self, sentiment):
    sentiment_stripped = sentiment.strip()
    if sentiment_stripped not in self._en_to_ar_label_name:
      raise ValueError(
          f"sentiment should be in {self._en_to_ar_label_name.keys()}."
          f" Received {sentiment_stripped}"
      )
    return self._en_to_ar_label_name[sentiment_stripped]

  def get_inputs(self, example):
    return str(example[self.TEXT_KEY])

  def get_outputs(self, example):
    return example[self.LABEL_KEY]

  def get_example_id(self, example):
    return example["id"]
