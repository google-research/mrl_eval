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

"""Preprocessing the ASAS dataset and writing it to storage."""

from collections.abc import Callable, Mapping, Sequence
import pathlib
from typing import Any

import immutabledict
import tensorflow as tf

from mrl_eval.datasets import constants
from mrl_eval.datasets import dataset_lib
from mrl_eval.evaluation import metrics
from mrl_eval.utils import io_utils


RawExample = dict[str, Any]
FeatureMap = dict[str, tf.train.Feature]
RawDataset = dataset_lib.RawDataset


class ASAS(dataset_lib.SummarizationDataset):
  """Implementation of the Dataset class for the Arabic ASAS dataset.

  This class transforms from raw dataset files into TensorFlow records.
  """

  article = "text"
  summary = "summary"
  ID = "id"
  MISSING_TEXT = "ERROR: Failed to process."

  @property
  def dataset_name(self):
    return constants.ASAS

  @property
  def raw_files(self):
    return {
        "train": "train.jsonl",
        "val": "dev.jsonl",
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

  def _get_summary(self, example):
    summary_sentences = [
        item["final_summary_sentence"] for item in example["annotations"]
    ]
    summary = " ".join(summary_sentences)
    return summary

  def process_raw_examples(self, dataset):
    processed_dataset = []
    for i, raw_example in enumerate(dataset):
      summary = self._get_summary(raw_example)
      text = raw_example["full_text"]
      if not text or text == self.MISSING_TEXT:
        print(f"Skipping example {i} because text is missing")
        continue

      elif len(summary) > len(text):
        print(f"Skipping example {i} because summary is longer than text")
        continue

      example = {
          self.ID: f"ASAS_{i}",
          self.article: text,
          self.summary: summary,
      }
      processed_dataset.append(example)

    return processed_dataset
