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

"""Preprocessing the Arabic IAHLT NER dataset and writing it to storage."""

from collections.abc import Mapping, Sequence
import pathlib
import re
import string
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


class IahltNer(dataset_lib.Dataset):
  """Implementation of the Dataset class for Arabic IAHLT NER."""

  LABEL_KEY = "label"
  TEXT_KEY = "text"
  ID_FIELD = "id"

  def __init__(self, separate_punctuation = False):
    super().__init__()
    self._separate_punctuation = separate_punctuation

  @property
  def dataset_name(self):
    return constants.IAHLT_NER

  @property
  def raw_files(self):
    return {
        split: f"iahlt_ner_{split}.jsonl" for split in ["train", "val", "test"]
    }

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
    dataset = io_utils.read_jsonl(file_path)
    return dataset

  def _insert_labels(
      self, text, labels
  ):
    """Inserts labels into the text based on the provided span indices.

    Args:
      text: The text to insert labels into.
      labels: A list of tuples (start, end, label) where:
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
    sorted_labels = sorted(labels, key=lambda x: x[0])
    result = []
    last_index = 0

    for start, end, label in sorted_labels:
      if not (0 <= start <= end <= len(text)):
        raise ValueError(
            f"Invalid span [{start}, {end}] for text of length {len(text)}."
        )

      # append text before the current entity
      result.append(text[last_index:start])
      # append labeled entity
      result.append(f"[{label} " + text[start:end] + "]")
      last_index = end

    # append remaining text
    result.append(text[last_index:])
    return "".join(result)

  def _separate_punctuation_tokens(self, text):
    """Inserts spaces around punctuation so that each punctuation mark becomes a separate token.

    Args:
      text: The text to separate punctuation from.

    Returns:
      The text with punctuation separated from the words around it.

    For example:
      "[TTL البروفسور], [PER محمود خليل]"

    Will return:
      "[TTL البروفسور] , [PER محمود خليل]"
    """

    arabic_punctuation = "،؛؟"
    # exclude square brackets from punctuation characters
    punctuation_chars = (
        "".join(ch for ch in string.punctuation if ch not in "[]")
        + arabic_punctuation
    )

    pattern = f"([{re.escape(punctuation_chars)}])"
    # add a space before and after punctuation character
    text_with_spaces = re.sub(pattern, r" \1 ", text)
    # remove extra spaces
    return " ".join(text_with_spaces.split())

  def _replace_brackets(self, text):
    """Replaces square brackets (used for entity markers) with parentheses in input text."""
    return text.replace("[", "(").replace("]", ")")

  def process_raw_examples(self, dataset):
    processed_dataset = []
    for raw_example in dataset:
      text = raw_example[self.TEXT_KEY]
      labels = raw_example[self.LABEL_KEY]
      id_key = raw_example[self.ID_FIELD]

      text = self._replace_brackets(text)
      text_with_labels = self._insert_labels(text, labels)
      if self._separate_punctuation:
        text_with_labels = self._separate_punctuation_tokens(text_with_labels)
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
