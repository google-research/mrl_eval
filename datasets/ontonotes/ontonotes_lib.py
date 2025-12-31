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

"""Implementation of Dataset class for OntoNotes."""

from collections.abc import Callable, Mapping, Sequence
import itertools
import pathlib
import re
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


def is_empty_target(
    clusters,
    char_limit,
    drop_singleton_clusters,
):
  """Returns True if there are no eligible clusters within the char limit.

  Args:
    clusters: A list of clusters, where each cluster is a list of mentions,
      and each mention is a pair of [start, end] word indices.
    char_limit: The character limit.
    drop_singleton_clusters: Whether to drop singleton clusters.
  """
  num_eligible_clusters = 0
  for cluster in clusters:
    num_mentions = 0
    for mention in cluster:
      if mention[1] <= char_limit:
        num_mentions += 1
    if (num_mentions == 1 and not drop_singleton_clusters) or (
        num_mentions > 1
    ):
      num_eligible_clusters += 1
  return num_eligible_clusters <= 0


class Ontonotes(dataset_lib.Dataset):
  """Implementation of a Dataset class for OntoNotes."""

  TEXT_FIELD = "ar_doc"
  TARGET_FIELD = "doc_corefs"
  ID_FIELD = "id"

  _SENTENCES = "sentences"
  _CLUSTERS = "clusters"

  INNER_SEP = "<S>"
  OUTER_SEP = "<N>"

  def __init__(
      self,
      char_limit = 3000 * 3,  # ~num_tokens * avg_chars_per_token
      index_text = True,
      index_targets = True,
      drop_singleton_clusters = False,
      drop_diacritics = True,
      word_separator = " ",
      sentence_separator = " ",
  ):
    super().__init__()
    self._text_parser = dataset_utils.CorefParser(
        index_text=index_text, index_targets=index_targets
    )
    self._char_limit = char_limit
    self._drop_singleton_clusters = drop_singleton_clusters
    self._drop_diacritics = drop_diacritics
    self._word_separator = word_separator
    self._sentence_separator = sentence_separator

  @property
  def dataset_name(self):
    return constants.ONTONOTES

  @property
  def raw_files(self):
    """Files templates."""
    return {
        "train": "train.arabic.jsonlines",
        "val": "dev.arabic.jsonlines",
        "test": "test.arabic.jsonlines",
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

  def read_raw_data_file(
      self, file_path
  ):
    return io_utils.read_jsonl(file_path)

  def _build_text_and_char_map(
      self,
      sentences,
      word_separator = " ",
      sentence_separator = ". "
  ):
    """Builds a single text string and maps each word to its starting character index.

    Args:
      sentences: A list of sentences, where each sentence is a list of words in
        the sentence.
      word_separator : The string to use as a separator between words.
        Defaults to " ".
      sentence_separator: The string to use as a separator between sentences.
        Defaults to ". ".

    Returns:
      tuple: A tuple containing the combined text, a list of all words, and a
        list mapping word indices to their starting character positions.
    """
    # Remove empty sentences.
    active_sentences = [s for s in sentences if s]

    # Put together the sentences into a single text string.
    sentence_texts = [word_separator.join(s) for s in active_sentences]
    text = sentence_separator.join(sentence_texts)
    all_words = list(itertools.chain.from_iterable(active_sentences))

    # Build a map of the starting character position of each word.
    word_start_char_map = []
    current_char_pos = 0
    word_sep_len = len(word_separator)
    sent_sep_len = len(sentence_separator)

    for i, sentence in enumerate(active_sentences):
      for j, word in enumerate(sentence):
        word_start_char_map.append(current_char_pos)
        current_char_pos += len(word)

        # Add the length of the separator that follows the word.
        is_last_word = j == len(sentence) - 1
        is_last_sentence = i == len(active_sentences) - 1

        if is_last_word and not is_last_sentence:
          current_char_pos += sent_sep_len  # End of a sentence
        elif not is_last_word:
          current_char_pos += word_sep_len  # Between words
        # No separator after the very last word of the text.

    return text, all_words, word_start_char_map

  def _merge_split_morphemes_in_sentence(
      self,
      sentence,
  ):
    """Merges split morphemes in a single sentence.

    Args:
      sentence: A list of words in a single sentence.

    Returns:
      A tuple containing:
        new_sentence: The list of unified words for this sentence.
        merged_counts_per_new_word: A list where each element is the
          count of old words that were merged into the new word at the
          corresponding index.
    """
    new_sentence = []
    merged_counts_per_new_word = []

    i = 0
    while i < len(sentence):
      word = sentence[i]

      # Check if a merge sequence *starts* here
      # The texts are split into morphemes or quasi-morphemes. There's a dash
      # immediately after and immediately before the morphemes that are split.
      if (word.endswith("-") and
          (i + 1) < len(sentence) and
          sentence[i + 1].startswith("-")):

        merged_word = word[:-1] + sentence[i + 1][1:]
        current_merge_count = 2
        i += 2  # We've processed two words

        # Inner loop for cases where the word is split into multiple
        # morphemes, e.g., ["A-", "-B-", "-C-", "-D"]
        while (merged_word.endswith("-") and
               (i < len(sentence)) and
               sentence[i].startswith("-")):
          # Continue merging
          merged_word = merged_word[:-1] + sentence[i][1:]
          current_merge_count += 1
          i += 1

        new_sentence.append(merged_word)
        merged_counts_per_new_word.append(current_merge_count)

      else:
        # Single word, no morpheme split
        new_sentence.append(word)
        merged_counts_per_new_word.append(1)
        i += 1

    return new_sentence, merged_counts_per_new_word

  def _build_word_index_map(
      self,
      merged_counts_per_new_word,
      current_new_word_index,
  ):
    """Builds the old-to-new word index map for a sentence.

    After merging morphemes, the old word indices no longer correspond to the
    new word indices. This function builds a map from the old word indices
    to the new word indices.

    Args:
      merged_counts_per_new_word: A list where each element is the count of old
        words that were merged into the new word at the corresponding index.
      current_new_word_index: The current new word index to start indexing this
        sentence from.

    Returns:
      A tuple containing:
        old_to_new_map_segment: The portion of the old-to-new index map
          for this sentence.
        next_new_word_index: The new global word index to use for the
          next sentence.
    """
    old_to_new_map_segment = []
    sentence_new_word_index = current_new_word_index

    # Iterate over the counts for each `new` word
    for count in merged_counts_per_new_word:
      # Map `count` number of `old` words to the `same`` new word index
      for _ in range(count):
        old_to_new_map_segment.append(sentence_new_word_index)
      # Move to the next `new` word index
      sentence_new_word_index += 1

    return old_to_new_map_segment, sentence_new_word_index

  def _unify_morphemes_and_build_map(
      self,
      sentences,
  ):
    """Unifies split morphemes and builds a map from old to new word indices.

    Args:
      sentences: The original list of sentences, with morphemes potentially
        split (e.g., ['وَ-', '-كانُوا']).

    Returns:
      A tuple containing:
        new_sentences: A list of sentences with morphemes unified.
        old_to_new_word_index_map: A flat list where the index is the
          `old` global word index and the value is the `new` global
          word index.
    """
    old_to_new_word_index_map = []
    new_sentences = []
    current_new_word_index = 0

    for sentence in sentences:
      if not sentence:
        new_sentences.append([])
        continue

      # Merge morphemes and get counts
      new_sentence, merged_counts_per_new_word = (
          self._merge_split_morphemes_in_sentence(sentence)
      )
      # Build the index map from the counts
      map_segment, current_new_word_index = self._build_word_index_map(
          merged_counts_per_new_word, current_new_word_index
      )

      new_sentences.append(new_sentence)
      old_to_new_word_index_map.extend(map_segment)

    return new_sentences, old_to_new_word_index_map

  def _remap_cluster_word_indices(
      self,
      clusters,
      old_to_new_word_index_map,
  ):
    """Converts clusters from old word indices to new word indices.

    Args:
      clusters: The original clusters using old (split) word indices.
      old_to_new_word_index_map: Map from old global word index to new
        global word index.

    Returns:
      New clusters using the new (unified) word indices. Duplicate mentions
      created during remapping are removed.
    """
    new_clusters = []
    for cluster in clusters:
      # Use a set to automatically handle duplicate mentions if any.
      new_cluster_mentions = set()
      for start_word_idx, end_word_idx in cluster:
        if (start_word_idx >= len(old_to_new_word_index_map) or
            end_word_idx >= len(old_to_new_word_index_map)):
          raise ValueError(
              "Word indices out of bounds. Old-to-new map is "
              f"{len(old_to_new_word_index_map)} long, but indices are "
              f"{start_word_idx} and {end_word_idx}."
          )

        new_start_word_idx = old_to_new_word_index_map[start_word_idx]
        new_end_word_idx = old_to_new_word_index_map[end_word_idx]
        new_cluster_mentions.add((new_start_word_idx, new_end_word_idx))

      if new_cluster_mentions:
        new_clusters.append(list(new_cluster_mentions))

    return new_clusters

  def _convert_clusters(
      self,
      clusters,
      all_words,
      word_start_char_map,
  ):
    """Converts word-based cluster indices to character-based indices.

    Args:
      clusters: A list of clusters, where each cluster is a list of mentions,
        and each mention is a pair of [start, end] word indices.
      all_words: A list of all words in the text.
      word_start_char_map: A list mapping word indices to their starting
        character positions.

    Returns:
       The new clusters with character-based indices.
    """
    new_clusters = []
    for cluster in clusters:
      new_cluster = []
      for start_word_idx, end_word_idx in cluster:
        start_char = word_start_char_map[start_word_idx]
        end_word = all_words[end_word_idx]
        end_char = word_start_char_map[end_word_idx] + len(end_word)
        # New mention span consists of the start and end character indices.
        new_cluster.append((start_char, end_char))

      new_clusters.append(new_cluster)

    return new_clusters

  def convert_to_char_indices(
      self,
      sentences,
      clusters,
  ):
    """Converts the span of the mentions to be based on character indices.

    The annotation of mentions in OntoNotes is based on word indices. This
    function converts the span of the mentions to be based on character indices.

    Args:
      sentences: A list of sentences, where each sentence is a list of words in
        the sentence.
      clusters: A list of clusters, where each cluster is a list of mentions,
        and each mention is a pair of [start, end] word indices.

    Returns:
      A tuple containing the combined text from all sentences and the
      coreference clusters with mentions updated to use character indices.
    """
    # Get the combined text and the word-to-character mapping.
    text, all_words, word_start_char_map = self._build_text_and_char_map(
        sentences, self._word_separator, self._sentence_separator
    )

    # Convert the cluster indices using the generated map.
    clusters_with_char_indices = self._convert_clusters(
        clusters, all_words, word_start_char_map
    )

    return text, clusters_with_char_indices

  def _remove_diacritics(
      self, sentences
  ):
    """Removes diacritics from words in the sentences."""
    pattern = (
        "["
        "\u064B-\u0652"  # Fathatan, Dammatan, Kasratan, Fatha, Damma, Kasra,
        # Shadda, Sukun
        "\u0670"  # Dagger Alif
        "\u06D6-\u06DC"  # Small High Ligatures
        "\u06DF-\u06E4"  # Small High Rounded Zero, etc.
        "\u06E7-\u06E8"  # Small High Upright Meem, etc.
        "\u06EA-\u06ED"  # Small Low Seen, etc.
        "\u0640"  # Tatweel
        "]"
    )
    new_sentences = []
    for sentence in sentences:
      new_sentence = []
      for word in sentence:
        new_sentence.append(re.sub(pattern, "", word))
      new_sentences.append(new_sentence)
    return new_sentences

  def _prep_target_from_clusters_spans(
      self,
      text,
      clusters_spans,
      adjusted_char_limit = None,
  ):
    """Returns the sequence representation for the example clusters from cluster spans."""
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

  def process_raw_examples(
      self, dataset
  ):
    """Processes examples in the dataset.

    Args:
      dataset: The raw dataset to preprocess.

    Returns:
      The processed dataset.
    """
    processed_dataset = []
    for raw_example in dataset:
      sentences = raw_example[self._SENTENCES]
      clusters = raw_example[self._CLUSTERS]

      # Unify split morphemes and get the index map.
      new_sentences, old_to_new_map = self._unify_morphemes_and_build_map(
          sentences
      )
      # Remap old cluster word indices to new word indices
      remapped_clusters = self._remap_cluster_word_indices(
          clusters, old_to_new_map
      )
      if self._drop_diacritics:
        new_sentences = self._remove_diacritics(new_sentences)
      text, clusters_with_char_indices = self.convert_to_char_indices(
          new_sentences, remapped_clusters
      )

      # Indexing words in the text if needed, adjusting char limit accordingly
      adjusted_char_limit = self._text_parser.get_character_limit_data(
          text, self._char_limit
      )["non_indexed_text_length"]

      if is_empty_target(
          clusters=clusters_with_char_indices,
          char_limit=adjusted_char_limit,
          drop_singleton_clusters=self._drop_singleton_clusters,
      ):
        continue

      example = {
          self.ID_FIELD: raw_example["doc_key"],
          self.TEXT_FIELD: self._text_parser.get_text(text, self._char_limit),
          self.TARGET_FIELD: self._prep_target_from_clusters_spans(
              text, clusters_with_char_indices, adjusted_char_limit
          ),
      }
      processed_dataset.append(example)

    return processed_dataset

  def get_inputs(self, example):
    return str(example[self.TEXT_FIELD])

  def get_outputs(self, example):
    return example[self.TARGET_FIELD]

  def get_example_id(self, example):
    return example[self.ID_FIELD]
