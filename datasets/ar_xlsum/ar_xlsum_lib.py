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

"""Preprocessing the HeSum dataset and writing it to storage."""

from collections.abc import Callable, Mapping, Sequence
import pathlib
from typing import Any

import immutabledict
import tensorflow as tf

from mrl_eval.datasets import constants
from mrl_eval.datasets import dataset_lib
from mrl_eval.evaluation import metrics
from mrl_eval.utils import io_utils


RawExample = dict[str, str]
FeatureMap = dict[str, tf.train.Feature]
RawDataset = dataset_lib.RawDataset


class ArXLSum(dataset_lib.SummarizationDataset):
  """Implementation of the Dataset class for the Arabic XLSum dataset.

  This class transforms from raw dataset files into TensorFlow records.
  """

  article = "text"
  summary = "summary"
  ID = "id"

  @property
  def dataset_name(self):
    return constants.AR_XLSUM

  @property
  def raw_files(self):
    return {
        "train": "arabic_train.jsonl",
        "val": "arabic_val.jsonl",
        "test": "arabic_test.jsonl",
    }

  @property
  def metrics(self):
    """Returns the metrics to be calculated for this dataset."""
    return [metrics.rouge]

  def map_to_feature(self, ex):
    # Disabling pytype here as it can't infer correct type for the dict values.
    # pytype: disable=attribute-error
    feature = {
        "id": self.bytes_feature([ex["id"].encode()]),
        self.article: self.bytes_feature([ex[self.article].encode()]),
        self.summary: self.bytes_feature([ex[self.summary].encode()]),
    }
    # pytype: enable=attribute-error

    return feature

  def name_to_features(self):
    return immutabledict.immutabledict({
        "id": tf.io.FixedLenFeature([], tf.string),
        self.article: tf.io.FixedLenFeature([], tf.string),
        self.summary: tf.io.FixedLenFeature([], tf.string),
    })

  def read_raw_data_file(self, file_path):
    """Reads a raw data file and returns a list of examples.

    Args:
      file_path: The path to the file to read.

    Returns:
      A list of examples, where each example is a dictionary mapping feature
      names to values.
    """
    data = io_utils.read_jsonl(file_path)
    return data

  def get_inputs(self, example):
    return example[self.article]

  def get_outputs(self, example):
    return example[self.summary]

  def get_example_id(self, example):
    return example[self.ID]

  def process_raw_examples(self, dataset):
    return dataset
