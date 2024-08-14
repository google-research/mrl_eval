# coding=utf-8
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

"""Preprocessing the NLI dataset and writing to the storage."""

from collections.abc import Callable, Mapping
import copy
import pathlib
from typing import Any, Dict, List

import immutabledict
import tensorflow as tf

from mrl_eval.datasets import constants
from mrl_eval.datasets import dataset_lib
from mrl_eval.evaluation import metrics
from mrl_eval.utils import io_utils


RawExample = Dict[str, str]
FeatureMap = Dict[str, tf.train.Feature]
RawDataset = dataset_lib.RawDataset


class HeSum(dataset_lib.Dataset):
  """Implementation of the Dataset class for HebNLI.

  This class transforms from raw dataset files into TensorFlow records.
  """

  ARTICLE = "article"
  SUMMARY = "summary"

  @property
  def dataset_name(self):
    return constants.HESUM

  @property
  def raw_files(self):
    return {"train": "train.csv", "val": "validation.csv", "test": "test.csv"}

  @property
  def metrics(self):
    """Returns the metrics to be calculated for this dataset."""
    return [metrics.rouge]

  def map_to_feature(self, ex):
    # Disabling pytype here as it can't infer correct type for the dict values.
    # pytype: disable=attribute-error
    feature = {
        "id": self.bytes_feature([ex["id"].encode()]),
        self.ARTICLE: self.bytes_feature([ex[self.ARTICLE].encode()]),
        self.SUMMARY: self.bytes_feature([ex[self.SUMMARY].encode()]),
    }
    # pytype: enable=attribute-error

    return feature

  def name_to_features(self):
    return immutabledict.immutabledict({
        "id": tf.io.FixedLenFeature([], tf.string),
        self.ARTICLE: tf.io.FixedLenFeature([], tf.string),
        self.SUMMARY: tf.io.FixedLenFeature([], tf.string),
    })

  def read_raw_data_file(self, file_path):
    """Reads a raw data file and returns a list of examples.

    Args:
      file_path: The path to the file to read.

    Returns:
      A list of examples, where each example is a dictionary mapping feature
      names to values.
    """
    df = io_utils.read_csv(file_path)
    return df.to_dict("records")

  def _add_example_ids(self, dataset):
    dataset_with_ids = []
    for i, raw_example in enumerate(dataset):
      example = copy.deepcopy(raw_example)
      example["id"] = f"hs_{i}"
      dataset_with_ids.append(example)
    return dataset_with_ids

  def process_raw_examples(self, dataset):
    return self._add_example_ids(dataset)

  def _get_target(self, example):
    return example[self.SUMMARY]
