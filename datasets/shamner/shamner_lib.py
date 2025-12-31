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

"""Preprocessing the ShamNER dataset and writing it to storage."""

from collections.abc import Mapping, Sequence
import logging
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

# create type aliases
SpanLabel = str
SpanIndex = int
Span = tuple[SpanIndex, SpanIndex, SpanLabel]
SpanList = list[Span]
RawSpan = dict[str, Union[int, str]]


class ShamNER(dataset_lib.Dataset):
  """Implementation of the Dataset class for ShamNER."""

  LABEL_KEY = "label"
  RAW_SPAN_KEY = "spans"
  TEXT_KEY = "text"
  ID_FIELD = "id"

  @property
  def dataset_name(self):
    return constants.SHAMNER

  @property
  def raw_files(self):
    files = {"train": "train-00000-of-00001.parquet",
             "val": "validation-00000-of-00001.parquet",
             "test": "test-00000-of-00001.parquet"}
    return files

  @property
  def metrics(self):
    return [metrics.token_level_span_f1]

  def map_to_feature(self, ex):
    # Disabling pytype here as it can't infer correct type for the dict values.
    # pytype: disable=attribute-error
    feature = {
        self.ID_FIELD: self.bytes_feature([str(ex[self.ID_FIELD]).encode()]),
        self.TEXT_KEY: self.bytes_feature([ex[self.TEXT_KEY].encode()]),
        self.LABEL_KEY: self.bytes_feature([ex[self.LABEL_KEY].encode()]),
    }
    # pytype: enable=attribute-error

    return feature

  def name_to_features(self):
    return immutabledict.immutabledict({
        self.ID_FIELD: tf.io.FixedLenFeature([], tf.string),
        self.TEXT_KEY: tf.io.FixedLenFeature([], tf.string),
        self.LABEL_KEY: tf.io.FixedLenFeature([], tf.string),
    })

  def read_raw_data_file(self, file_path):
    dataset = io_utils.read_parquet(file_path)
    return dataset.to_dict(orient="records")

  def _insert_labels(self, text, spans):
    """Inserts labels into the text based on the provided span indices.

    Args:
      text: The text to insert labels into.
      spans: A list of tuples (start, end, label) where:
      - start: starting index (inclusive)
      - end: ending index (exclusive)
      - label: the label name as a string

    Returns:
      The text with labels inserted into it.

    For example:
      text = "البروفسور, محمود خليل"
      labels = [[0, 9, "TTL"], [11, 21, "PER"]]

    Will return:
      "[TTL البروفسور], [PER محمود خليل]"
    """
    # Sort labels by starting index
    sorted_spans = sorted(spans, key=lambda x: x[0])
    result = []
    last_index = 0

    for start, end, label in sorted_spans:
      # append text before the current entity
      result.append(text[last_index:start])
      # append labeled entity
      result.append(f"[{label} " + text[start:end] + "]")
      last_index = end

    # append remaining text
    result.append(text[last_index:])
    return "".join(result)

  def _replace_brackets(self, text):
    """Replaces square brackets (reserved for entity markers) with parentheses in original input text."""
    return text.replace("[", "(").replace("]", ")")

  def _trim_ws(self, text, spans):
    """Removes whitespace from the start and end of each span."""
    trimmed_spans = []
    for start, end, label in spans:
      # Keep track of original end to not go past it
      original_end = end

      while start < original_end and text[start].isspace():
        start += 1

      while end > start and text[end - 1].isspace():
        end -= 1

      # Only add the span if it's still valid (i.e., not empty)
      if start < end:
        trimmed_spans.append((start, end, label))

    return trimmed_spans

  def _has_overlapping_spans(self, spans):
    """Checks if any spans in the list overlap."""
    # Sort spans by starting index
    sorted_spans = sorted(spans, key=lambda x: x[0])
    for i in range(len(sorted_spans) - 1):
      if sorted_spans[i][1] > sorted_spans[i + 1][0]:
        return True
    return False

  def _has_invalid_spans(self, text, spans):
    """Checks if any spans in the list are invalid."""
    for start, end, _ in spans:
      if not (0 <= start <= end <= len(text)):
        return True
    return False

  def process_raw_examples(self, dataset):
    processed_dataset = []
    for idx, raw_example in enumerate(dataset):
      text = raw_example[self.TEXT_KEY]
      spans = raw_example[self.RAW_SPAN_KEY]
      id_key = f"Shamner_{idx}"

      spans = [(item["start"], item["end"], item["label"]) for item in spans]

      if self._has_invalid_spans(text, spans):
        logging.warning("Skipping example %s due to invalid spans.", id_key)
        continue

      if self._has_overlapping_spans(spans):
        logging.warning("Skipping example %s due to overlapping spans.", id_key)
        continue

      text = self._replace_brackets(text)
      spans = self._trim_ws(text, spans)
      text_with_labels = self._insert_labels(text, spans)

      processed_dataset.append({
          self.ID_FIELD: id_key,
          self.TEXT_KEY: text,
          self.LABEL_KEY: text_with_labels,
      })
    return processed_dataset

  def get_inputs(self, example):
    return str(example[self.TEXT_KEY])

  def get_outputs(self, example):
    return example[self.LABEL_KEY]

  def get_example_id(self, example):
    return example[self.ID_FIELD]
