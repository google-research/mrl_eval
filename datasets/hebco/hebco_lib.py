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

"""Implementation of the Dataset class for HebCo.

https://github.com/IAHLT/coref/tree/master This class transforms from raw
dataset files into TensorFlow records.
"""

from collections.abc import Callable, Mapping, Sequence
import copy
import pathlib
from typing import Any, Union

import immutabledict
import tensorflow as tf

from mrl_eval.datasets import constants
from mrl_eval.datasets import dataset_lib
from mrl_eval.evaluation import metrics
from mrl_eval.utils import dataset_utils
from mrl_eval.utils import io_utils

Tokens = list[str]
Tag = str
Tags = list[Tag]
ClustersType = list[dict[str, Any]]
RawExample = dict[str, Union[int, str, Tokens, Tags, ClustersType]]
FeatureMap = dict[str, tf.train.Feature]
RawDataset = dataset_lib.RawDataset


class Hebco(dataset_lib.Dataset):
  """Implementation of the Dataset class for HebCo.

  This class transforms from raw dataset files into TensorFlow records.
  """

  TEXT_FIELD = "heb_doc"
  TARGET_FIELD = "doc_corefs"
  ID_FIELD = "id"

  _TEXT = "text"
  _CLUSTERS = "clusters"
  _METADATA = "metadata"
  _MENTIONS = "mentions"

  INNER_SEP = "<S>"
  OUTER_SEP = "<N>"
  WORD_SEP = " "

  ZWS = "\u200B"

  def __init__(
      self,
      char_limit = 1500 * 3,  # ~num_tokens * avg_chars_per_token
      index_text = True,
      index_targets = True,
      drop_singleton_clusters = False,
  ):
    super().__init__()
    self._index_text = index_text
    self._index_targets = index_targets
    self._text_parser = dataset_utils.CorefParser(
        index_text=index_text, index_targets=index_targets
    )
    self._char_limit = char_limit
    self._drop_singleton_clusters = drop_singleton_clusters

  @property
  def dataset_name(self):
    return constants.HEBCO

  @property
  def raw_files(self):
    """Files templates."""
    return {
        "train": "coref-5-heb_train.jsonl",
        "val": "coref-5-heb_val.jsonl",
        "test": "coref-5-heb_test.jsonl",
    }

  @property
  def metrics(self):
    return [
        metrics.get_em_cluster_matching_f1_fn(
            get_parse_string_representation(
                outer_sep=self.OUTER_SEP, inner_sep=self.INNER_SEP
            )
        )
    ]

  def map_to_feature(self, ex):
    # Disabling pytype here, can't infer correct type for the dict values.
    # pytype: disable=attribute-error
    feature = {
        "id": self.bytes_feature([ex["id"].encode()]),
        self.TEXT_FIELD: self.bytes_feature([ex[self.TEXT_FIELD].encode()]),
        self.TARGET_FIELD: self.bytes_feature([ex[self.TARGET_FIELD].encode()]),
    }
    # pytype: enable=attribute-error

    return feature

  def name_to_features(self):
    return immutabledict.immutabledict({
        "id": tf.io.FixedLenFeature([], tf.string),
        self.TEXT_FIELD: tf.io.FixedLenFeature([], tf.string),
        self.TARGET_FIELD: tf.io.FixedLenFeature([], tf.string),
    })

  def read_raw_data_file(self, file_path):
    return io_utils.read_jsonl(file_path)

  def process_raw_examples(self, dataset):
    processed_dataset = []
    for raw_example in dataset:
      # Removing unecesary white spaces and adding a single space between words
      standardized_example = self.standardize_text_and_adjust_clusters(
          raw_example
      )
      # Indexing words in the text if needed, adjusting char limit accordingly
      adjusted_char_limit = self._text_parser.get_character_limit_data(
          standardized_example[self._TEXT], self._char_limit
      )["non_indexed_text_length"]

      if self._is_empty_target(standardized_example, adjusted_char_limit):
        continue
      example = {}
      example[self.ID_FIELD] = standardized_example["doc_key"]
      example[self.TEXT_FIELD] = self._get_text(standardized_example)
      example[self.TARGET_FIELD] = self._prep_target_from_raw(
          standardized_example, adjusted_char_limit
      )
      processed_dataset.append(example)
    return processed_dataset

  def _get_text(self, standardized_example):
    processed_text = self._text_parser.get_text(
        standardized_example[self._TEXT], self._char_limit
    )
    processed_text = (processed_text).replace(self.ZWS, "")
    return processed_text

  def adjust_cluster_indices(
      self, clusters, index_mapping
  ):
    adjusted_clusters = []
    for cluster in clusters:
      adjusted_mentions = []
      for mention in cluster[self._MENTIONS]:
        start, end, tags = mention  # maitaining HebCo format (start, end, tags)
        new_start = index_mapping[start] if start in index_mapping else start
        new_end = index_mapping[end] if end in index_mapping else end
        adjusted_mentions.append([new_start, new_end, tags])
      if adjusted_mentions:
        adjusted_clusters.append({
            self._METADATA: cluster[self._METADATA],
            self._MENTIONS: adjusted_mentions,
        })
    return adjusted_clusters

  def _is_whitespace(self, ch):
    return ch.isspace() and ch != self.ZWS  # Exclude ZWS

  def standardize_text_and_adjust_clusters(
      self, example
  ):

    example = copy.deepcopy(example)
    text = example[self._TEXT]
    clusters = example[self._CLUSTERS]

    # Create a mapping from original indices to new indices
    orig_to_new_index_mapping = {}

    new_text = []
    new_index = 0
    char_index = 0
    text_length = len(text)

    while char_index < text_length:
      if self._is_whitespace(text[char_index]):
        char_index += 1
        if new_text and new_text[-1] != self.WORD_SEP:
          new_text.append(self.WORD_SEP)
          orig_to_new_index_mapping[char_index - 1] = (
              new_index  # mapping the last whitespace to a space
          )
          new_index += 1
          continue
      else:
        orig_to_new_index_mapping[char_index] = new_index
        new_text.append(text[char_index])
        char_index += 1
        new_index += 1

    # remove trailing space if present
    if new_text and new_text[-1] == self.WORD_SEP:
      new_text.pop()

    standardized_text = "".join(new_text)

    # Adjust clusters using the index mapping
    adjusted_clusters = self.adjust_cluster_indices(
        clusters, orig_to_new_index_mapping
    )
    example[self._CLUSTERS] = adjusted_clusters
    example[self._TEXT] = standardized_text
    return example

  def _is_empty_target(self, example, char_limit):
    """Returns True if the target is empty."""
    num_eligible_clusters = 0
    for cluster in example[self._CLUSTERS]:
      num_mentions = 0
      for mention in cluster[self._MENTIONS]:
        if mention[1] <= char_limit:
          num_mentions += 1
      if (num_mentions == 1 and not self._drop_singleton_clusters) or (
          num_mentions > 1
      ):
        num_eligible_clusters += 1
    return num_eligible_clusters <= 0

  def _prep_target_from_raw(
      self,
      example,
      adjusted_char_limit = None,
  ):
    """Returns the sequence representation for the example clusters from raw examples."""
    text = example[self._TEXT]
    clusters_spans = []
    for cluster in example[self._CLUSTERS]:
      cluster_mentions = []
      for mention in cluster[self._MENTIONS]:
        cluster_mentions.append((mention[0], mention[1]))
      clusters_spans.append(cluster_mentions)
    filtered_clusters = self._text_parser.get_clusters_as_strings(
        text, clusters_spans, adjusted_char_limit
    )
    if self._drop_singleton_clusters:
      filtered_clusters = [
          cluster for cluster in filtered_clusters if len(cluster) > 1
      ]
    seq_output = self.OUTER_SEP.join(
        [self.INNER_SEP.join(mentions) for mentions in filtered_clusters]
    ).replace(self.ZWS, "")
    return seq_output

  def get_inputs(self, example):
    return str(example[self.TEXT_FIELD])

  def get_outputs(self, example):
    return example[self.TARGET_FIELD]

  def get_example_id(self, example):
    return example[self.ID_FIELD]


def get_parse_string_representation(
    outer_sep, inner_sep
):
  """Returns a function that parses a string representation of clusters into a list of clusters."""
  def parse_string_representation(str_representation):
    clusters = str_representation.split(outer_sep)
    res = []
    for cluster in clusters:
      res.append(cluster.split(inner_sep))
    return res

  return parse_string_representation
