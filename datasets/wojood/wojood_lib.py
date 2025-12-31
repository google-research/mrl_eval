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

"""Preprocessing the Wojood dataset and writing it to storage.
"""

import enum
import pathlib
from typing import Any, Mapping, Union

from absl import logging
import immutabledict
import pandas as pd
import tensorflow as tf

from mrl_eval.datasets import constants
from mrl_eval.datasets import dataset_lib
from mrl_eval.evaluation import metrics

Tokens = list[str]
Tag = str
Tags = list[Tag]
RawExample = dict[str, Union[int, str, Tokens, Tags]]
FeatureMap = dict[str, tf.train.Feature]
RawDataset = dataset_lib.RawDataset


class Variant(enum.Enum):
  """Variant of the dataset."""

  SPOKEN = "spoken"
  MSA = "msa"
  FULL = "full"


def targets_as_entity_markers_formulation(ex):
  """Formulate example where entities are denoted with markers.

  Output is represented as the input with additional entities markers that wrap
  each entity and its corresponding type.

  Output should be formulated in the following way:
  Word1 [LabelName1 Word2 Word3] ...

  For example:
    for the input:
      Barack Obama was born in Honolulu
    We create the following targets:
      [PER Barack Obama] was born in [GPE Honolulu]

  Args:
    ex: a RawExample is a list of RawExample.

  Returns:
    The transformed output from RawExample to str. If the tags are not formatted
    as IOB, returns None.
  """

  tokens, tags = ex["tokens"], ex["tags"]
  if not tags:
    return None
  if not tokens:
    return None
  if len(tokens) != len(tags):
    return None
  result_parts, current_span_tokens, current_span_type = [], [], None

  for token, tag in zip(tokens, tags):
    if tag.startswith("B-"):
      if current_span_tokens:
        result_parts.append(
            f"[{current_span_type} {' '.join(current_span_tokens)}]"
        )
      current_span_tokens, current_span_type = [token], tag[2:]
    elif tag.startswith("I-"):
      tag_type = tag[2:]
      if current_span_type == tag_type:
        current_span_tokens.append(token)
      else:
        return None
    else:  # 'O' tag
      if current_span_tokens:
        result_parts.append(
            f"[{current_span_type} {' '.join(current_span_tokens)}]"
        )
      current_span_tokens, current_span_type = [], None
      result_parts.append(token)

  if current_span_tokens:
    result_parts.append(
        f"[{current_span_type} {' '.join(current_span_tokens)}]"
    )
  return " ".join(result_parts)


class Wojood(dataset_lib.Dataset):
  """Wojood dataset.

  This class provides methods to load, process, and prepare the Wojood dataset
  for training and evaluation.

  Attributes:
    variant: The variant of the dataset to use (e.g., spoken, MSA, full).
  """

  def __init__(self, variant):
    super().__init__()
    self._variant = variant

  @property
  def raw_files(self):
    sub_dir = "Wojood1_1_flat"
    return {
        "train": f"{sub_dir}/train.csv",
        "dev": f"{sub_dir}/val.csv",
        "test": f"{sub_dir}/test.csv",
    }

  def _validate_tags(self, tags):
    return all([t[0] in "IOB" for t in tags])

  def _extract_tokens_and_tags(
      self, df
  ):
    """Extracts tokens and tags from a dataframe.

    This function groups the dataframe by `global_sentence_id`, validates the
    sentence length, filters by variant (spoken or MSA), and extracts the tokens
    and tags for each sentence.

    Args:
      df: The input dataframe containing token and tag information.

    Returns:
      A list of dictionaries, where each dictionary represents a sentence with
      its tokens, tags, subcorpus, and id.
    """
    dataset = []
    for name, group in df.groupby("global_sentence_id"):
      sentence_len = len(group)
      # Sanity check. Verify sentence length equal to max token position.
      if sentence_len != group["token_position"].max():
        logging.error(
            "Sentence length %d != max token position %d",
            sentence_len,
            group["token_position"].max(),
        )
        continue
      subcorpus = group["Sub_corpus"].iloc[0]
      dialect_subcorpus = ["Curras", "Lebanese"]
      if self._variant == Variant.SPOKEN:
        if subcorpus not in dialect_subcorpus:
          continue
      elif self._variant == Variant.MSA:
        if subcorpus in dialect_subcorpus: continue
      sentence = {
          "tokens": [],
          "tags": [],
          "subcorpus": subcorpus,
          "id": f"wojood_{name}",
      }
      for i in range(1, sentence_len + 1):
        token_data = group[group["token_position"] == i].iloc[0]
        sentence["tokens"].append(token_data["token"])
        sentence["tags"].append(token_data["tags"])
      if self._validate_tags(sentence["tags"]):
        dataset.append(sentence)
      else:
        logging.error(
            "Invalid tag. Skipped sentence id: %s, tags: %s",
            name,
            sentence["tags"],
        )
    return dataset

  def read_raw_data_file(self, path):
    df = pd.read_csv(path)
    return self._extract_tokens_and_tags(df)

  def process_raw_examples(
      self, dataset
  ):
    """Processes raw examples by formulating targets as entity markers.

    This function iterates through the input dataset, applies the
    `targets_as_entity_markers_formulation` function to each example, and
    creates a new dataset with "id", "inputs", and "targets_as_entity_markers"
    keys. It skips examples where the formulation fails.

    Args:
      dataset: The input dataset containing raw examples with tokens and tags.

    Returns:
      A new dataset with processed examples, where targets are formulated as
      entity markers.
    """
    processed_dataset = []
    for example in dataset:
      targets_as_entity_markers = targets_as_entity_markers_formulation(example)
      if not targets_as_entity_markers:
        logging.error(
            "Failed to formulate targets as entity markers for example: %s",
            example,
        )
        continue
      processed_dataset.append({
          "id": example["id"],
          "inputs": " ".join(example["tokens"]),
          "targets_as_entity_markers": targets_as_entity_markers,
      })
    return processed_dataset

  def get_inputs(self, example):
    """Returns the input to give to the model for a given example."""
    return example["inputs"]

  def get_outputs(self, example):
    """Returns the expected response for a given example."""
    return example["targets_as_entity_markers"]

  def get_example_id(self, example):
    return example["id"]

  def map_to_feature(self, ex):
    """mapping RawExample into a FeatureMap as expected for TFRecord."""

    # Disabling pytype here as it can't infer correct type for the dict values.
    # pytype: disable=attribute-error
    feature = {
        "id": self.bytes_feature([ex["id"].encode()]),
        "inputs": self.bytes_feature([ex["inputs"].encode()]),
        "targets_as_entity_markers": self.bytes_feature(
            [ex["targets_as_entity_markers"].encode()]
        ),
    }
    # pytype: enable=attribute-error

    return feature

  def name_to_features(self):
    return immutabledict.immutabledict({
        "id": tf.io.FixedLenFeature([], tf.string),
        "inputs": tf.io.FixedLenFeature([], tf.string),
        "targets_as_entity_markers": tf.io.FixedLenFeature([], tf.string),
    })

  @property
  def metrics(self):
    return [metrics.token_level_span_f1]

  @property
  def dataset_name(self):
    if self._variant == Variant.SPOKEN:
      return constants.WOJOOD_SPOKEN
    elif self._variant == Variant.MSA:
      return constants.WOJOOD_MSA
    else:
      return constants.WOJOOD_FULL
