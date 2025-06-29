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

"""Utilities for processing datasets."""

from collections.abc import Mapping, Sequence
import re
from typing import Any, Optional


class CorefParser:
  """A class for parsing coreference data, annotating words with indices and adjusting clusters accordingly.

  Attributes:
    left_span_bracket: The left span bracket to use for indexing.
    right_span_bracket: The right span bracket to use for indexing.
    index_text: Whether to index the text.
    index_targets: Whether to index the targets.
  """

  _WORD_INDEX = 'index'
  _WORD_TEXT = 'text'
  _WORD_START = 'start'
  _WORD_END = 'end'
  _INDEXED_WORD_END = 'indexed_end'
  _WORD_SEP = ' '

  def __init__(
      self,
      left_span_bracket='[',
      right_span_bracket=']',
      index_text=False,
      index_targets=False,
  ):
    self.left_span_bracket = left_span_bracket
    self.right_span_bracket = right_span_bracket
    self.index_text = index_text
    self.index_targets = index_targets

  def extract_words(self, text):
    """Extract words from the text along with their indices and character positions.

    Handles multiple spaces or tabs between words.

    Args:
      text: The text to extract words from.

    Returns:
      A list of words, each represented by a dictionary with the following keys:
      index: The index of the word in the text.
      text: The text of the word.
      start: The start character position of the word in the text.
      end: The end character position of the word in the text.
    """
    words = []
    for idx, match in enumerate(
        re.finditer(r'\S+', text)
    ):  # for multisple spaces
      word_text = match.group()
      word_start = match.start()
      word_end = match.end()
      words.append({
          self._WORD_INDEX: idx,
          self._WORD_TEXT: word_text,
          self._WORD_START: word_start,
          self._WORD_END: word_end,
      })
    words = sorted(words, key=lambda x: x[self._WORD_INDEX])
    indexed_additions = 0
    for word in words:
      cur_addition = len(f'_{word[self._WORD_INDEX]}')
      word[self._INDEXED_WORD_END] = (
          word[self._WORD_END] + cur_addition + indexed_additions
      )
      indexed_additions += cur_addition
    return words

  def get_span_words(
      self, words, span_start, span_end
  ):
    """Get the words that are related to the span."""
    related_words = []
    for word in words:
      if word[self._WORD_END] < span_start or word[self._WORD_START] > span_end:
        continue  # No overlap
      else:
        related_words.append(word)
    return related_words

  def process_mention_span(
      self, words, span_start, span_end
  ):
    """Process a single span and return the formatted span segment string representation.

    Adjusted to always enclose the output in brackets.

    Args:
      words: A list of words, each represented by the same dictionary as
        returned by extract_words.
      span_start: The start character position of the span.
      span_end: The end character position of the span.

    Returns:
      The formatted span segment representation.
    """

    related_words = self.get_span_words(words, span_start, span_end)

    if not related_words:
      raise ValueError(
          f'Span [{span_start}, {span_end}] does not overlap with any word.'
      )

    span_representation = []
    was_opened = False
    was_closed = False
    for word in related_words:
      word_start = word[self._WORD_START]
      word_end = word[self._WORD_END]

      within_word_start = max(span_start, word_start) - word_start
      within_word_end = min(span_end, word_end) - word_start
      word_text = word[self._WORD_TEXT]

      before_span = word_text[:within_word_start]
      span_part = word_text[within_word_start:within_word_end]
      after_span = word_text[within_word_end:]

      if span_start > word_start:
        before_span += self.left_span_bracket
        was_opened = True
      if span_end <= word_end:
        span_part += self.right_span_bracket
        was_closed = True
      word_output = f'{before_span}{span_part}{after_span}'
      if self.index_targets:
        word_output = f'{word[self._WORD_INDEX]}_{word_output}'
      span_representation.append(word_output)

    cand = f'{self._WORD_SEP.join(span_representation)}'
    if not was_opened:
      cand = self.left_span_bracket + cand
    if not was_closed:
      cand = cand + self.right_span_bracket
    return cand

  def get_clusters_as_strings(
      self,
      text,
      clusters,
      char_limit = None,
  ):
    """Process multiple clusters in the text, returns the formatted cluster.

    Args:
      text: The text to process.
      clusters: The clusters to process.
      char_limit: The character limit to apply (character limit of the
        non-indexed text).

    Returns:
      The formatted clusters.
    """
    words = self.extract_words(text)
    if char_limit is not None:
      if char_limit <= 0:
        raise ValueError('char_limit must be a positive number.')
      clusters = self.remove_exceeding_mentions(clusters, char_limit)
    if clusters is None:
      return
    clusters_as_texts = []
    for mention_spans in clusters:
      span_outputs = []
      for span_start, span_end in mention_spans:
        span_output = self.process_mention_span(words, span_start, span_end)
        span_outputs.append(span_output)
      clusters_as_texts.append(span_outputs)
    return clusters_as_texts

  def get_text(self, text, char_limit = None):
    """Returns the formatted text (e.g. indexing words if needed)."""
    last_word_index = self.get_character_limit_data(text, char_limit)[
        'last_word_index'
    ]
    words = self.extract_words(text)
    return self._WORD_SEP.join(
        self._get_word_representation(word)
        for word in words
        if word[self._WORD_INDEX] <= last_word_index
    )

  def get_character_limit_data(
      self, text, char_limit = None
  ):
    """Returns the last word index that fits within the character limit and the length of the text up to that word."""
    words = self.extract_words(text)
    matching_word_idx = -1
    if (
        char_limit is not None
        and self._get_word_end_index(words[-1]) > char_limit
    ):
      for word_idx, word in enumerate(words):
        if self._get_word_end_index(word) >= char_limit:
          matching_word_idx = max(word_idx - 1, 0)
          break
    return {
        'last_word_index': words[matching_word_idx][self._WORD_INDEX],
        'non_indexed_text_length': words[matching_word_idx][self._WORD_END],
        'actual_text_length': self._get_word_end_index(
            words[matching_word_idx]
        ),
    }

  def _get_word_representation(
      self,
      word,
  ):
    """Returns the formatted word representation."""
    if self.index_text:
      return f'{word[self._WORD_INDEX]}_{word[self._WORD_TEXT]}'
    else:
      return word[self._WORD_TEXT]

  def _get_word_end_index(
      self,
      word,
  ):
    """Returns the formatted word representation."""
    if self.index_text:
      return word[self._INDEXED_WORD_END]
    else:
      return word[self._WORD_END]

  def remove_exceeding_mentions(
      self, clusters, char_limit
  ):
    """Removes mentions that exceed the character limit."""
    updated_clusters = []
    for cluster in clusters:
      filtered_mentions = [
          mention for mention in cluster if mention[1] <= char_limit
      ]
      if filtered_mentions:
        updated_clusters.append(filtered_mentions)
    if updated_clusters:
      return updated_clusters
    return
