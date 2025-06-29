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

"""Implementation of the Dataset class for ArCoref.
"""

from collections.abc import Callable, Mapping, Sequence
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


class ArCoref(dataset_lib.Dataset):
  """Implementation of the Dataset class for ArCoref.

  This class transforms from raw dataset files into TensorFlow records.
  """

  TEXT_FIELD = "ar_doc"
  TARGET_FIELD = "doc_corefs"
  ID_FIELD = "id"

  _TEXT = "text"
  _CLUSTERS = "clusters"
  _METADATA = "metadata"
  _MENTIONS = "mentions"

  INNER_SEP = "<S>"
  OUTER_SEP = "<N>"
  WORD_SEP = " "

  def __init__(
      self,
      char_limit = 3000 * 3,  # ~num_tokens * avg_chars_per_token
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
    return constants.ARCOREF

  @property
  def raw_files(self):
    """Files templates."""
    return {
        "train": "arcoref_train.jsonl",
        "val": "arcoref_val.jsonl",
        "test": "arcoref_test.jsonl",
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
      # Indexing words in the text if needed, adjusting char limit accordingly
      adjusted_char_limit = self._text_parser.get_character_limit_data(
          raw_example[self._TEXT], self._char_limit
      )["non_indexed_text_length"]

      if self._is_empty_target(raw_example, adjusted_char_limit):
        continue
      example = {}
      example[self.ID_FIELD] = raw_example["doc_key"]
      example[self.TEXT_FIELD] = self._get_text(raw_example)
      example[self.TARGET_FIELD] = self._prep_target_from_raw(
          raw_example, adjusted_char_limit
      )
      processed_dataset.append(example)
    return processed_dataset

  def _get_text(self, standardized_example):
    processed_text = self._text_parser.get_text(
        standardized_example[self._TEXT], self._char_limit
    )
    return processed_text

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
    )
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
