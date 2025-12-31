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

"""Implementation of a revised Dataset class for ArCoref with gold mentions.

The initial dataset format can be found in:
nlp/mrl_eval/datasets/arcoref/arcoref_lib.py
"""

from collections.abc import Mapping, Sequence
import pathlib
from typing import Any, Union

import immutabledict
import tensorflow as tf

from mrl_eval.datasets import constants
from mrl_eval.datasets import dataset_lib
from mrl_eval.datasets.arcoref import arcoref_lib
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


class ArCorefGoldMentions(dataset_lib.Dataset):
  """Implementation of a revised Dataset class for ArCoref with gold mentions.

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
      drop_singleton_clusters = False,
      merge_subword_mentions = False,
  ):
    super().__init__()
    self.markup_strategy = dataset_utils.BracketMarkupStrategy()
    self.target_format_strategy = (
        dataset_utils.AnaphorAntecedentTargetStrategy()
    )

    self._text_parser = dataset_utils.CorefWithGoldMentionsParser(
        markup_strategy=self.markup_strategy,
        target_format_strategy=self.target_format_strategy,
        merge_subword_mentions=merge_subword_mentions,
    )
    self._char_limit = char_limit
    self._drop_singleton_clusters = drop_singleton_clusters

  @property
  def dataset_name(self):
    return constants.ARCOREF_GOLD_MENTIONS

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
            self.target_format_strategy.parse_string_representation
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

  def _get_clusters_spans(
      self, example
  ):
    """Returns the clusters spans from the example."""
    clusters_spans = []
    for cluster in example[self._CLUSTERS]:
      cluster_mentions = []
      for mention in cluster[self._MENTIONS]:
        cluster_mentions.append((mention[0], mention[1]))
      clusters_spans.append(cluster_mentions)

    return clusters_spans

  def process_raw_examples(self, dataset):
    """Processes examples in the dataset into the revised ArCoref format.

    Args:
      dataset: The raw dataset to preprocess.

    Returns:
      The processed dataset.
    """
    processed_dataset = []
    for raw_example in dataset:
      # Indexing words in the text if needed, adjusting char limit accordingly
      adjusted_char_limit = self._text_parser.get_adjusted_char_limit(
          raw_example[self._TEXT], self._char_limit
      )

      if arcoref_lib.is_empty_target(
          example=raw_example,
          char_limit=adjusted_char_limit,
          drop_singleton_clusters=self._drop_singleton_clusters,
          clusters_key=self._CLUSTERS,
          mentions_key=self._MENTIONS,
      ):
        continue

      clusters_spans = self._get_clusters_spans(raw_example)
      # Drop clusters with only one mention if needed.
      if self._drop_singleton_clusters:
        clusters_spans = [
            cluster for cluster in clusters_spans if len(cluster) > 1
        ]

      parsed_example = self._text_parser.parse_example(
          text=raw_example[self._TEXT],
          clusters=clusters_spans,
          char_limit=self._char_limit,
      )
      example = {
          self.ID_FIELD: raw_example["doc_key"],
          self.TEXT_FIELD: parsed_example.input,
          self.TARGET_FIELD: parsed_example.label,
      }

      processed_dataset.append(example)

    return processed_dataset

  def get_inputs(self, example):
    return str(example[self.TEXT_FIELD])

  def get_outputs(self, example):
    return example[self.TARGET_FIELD]

  def get_example_id(self, example):
    return example[self.ID_FIELD]
