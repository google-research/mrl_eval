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

r"""Preprocessing the HeBSummaries dataset and writing it to storage."""

from collections.abc import Callable, Mapping, Sequence
import copy
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


class HebSummaries(dataset_lib.SummarizationDataset):
  """Implementation of the Dataset class for the HebSummaries dataset.

  This class transforms from raw dataset files into TensorFlow records.
  """

  article = "text_raw"
  summary = "summary"

  @property
  def dataset_name(self):
    return constants.HEBSUMMARIES

  @property
  def raw_files(self):
    return {
        "train": "train.jsonl",
        "val": "val.jsonl",
        "test": "test.jsonl",
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
    return io_utils.read_jsonl(file_path)

  def get_inputs(self, example):
    return example[self.article]

  def get_outputs(self, example):
    return example[self.summary]

  def get_example_id(self, example):
    return example["id"]

  def _add_example_ids(self, dataset):
    dataset_with_ids = []
    for i, raw_example in enumerate(dataset):
      example = copy.deepcopy(raw_example)
      example["id"] = f"hs_{i}"
      dataset_with_ids.append(example)
    return dataset_with_ids

  def process_raw_examples(self, dataset):
    return self._add_example_ids(dataset)
