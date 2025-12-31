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

import abc
import bisect
from collections.abc import Mapping, Sequence
import dataclasses
import re
from typing import Any, Optional


@dataclasses.dataclass
class CorefExample:
  """A dataclass for representing a coreference example."""
  input: str
  label: str


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
    ):  # for multiple spaces
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


class GoldMentionsMarkupStrategy(abc.ABC):
  """Abstract base class for coreference text markup strategies."""

  @abc.abstractmethod
  def _get_end_marker(self, mention_id):
    """Gets the end marker for a mention."""

  @abc.abstractmethod
  def _get_start_marker(self, mention_id):
    """Gets the start marker for a mention."""

  def annotate_text_with_events(
      self,
      text,
      sorted_events,
      mention_to_id_map
  ):
    """Uses the events to annotate the text with mentions.

    Args:
      text: The text to annotate.
      sorted_events: The events to annotate, sorted by order of appearance.
      mention_to_id_map: A mapping from mentions to unique indices.

    Returns:
      The text with mentions annotated according to the strategy.
    """
    result_parts = []
    last_pos = 0
    for pos, event_type, span in sorted_events:
      # Append the text slice from the last event to the current one.
      result_parts.append(text[last_pos:pos])

      mention_id = mention_to_id_map[span]
      if event_type == 'START':
        result_parts.append(self._get_start_marker(mention_id))
      elif event_type == 'END':
        result_parts.append(self._get_end_marker(mention_id))
      else:
        raise ValueError(f'Unsupported event type: {event_type}')

      last_pos = pos

    # Append the final slice of text after the last event.
    result_parts.append(text[last_pos:])

    return ''.join(result_parts)

  def _get_sorted_events_from_mentions(
      self,
      mentions,
  ):
    """Generates a sorted sequence of events from a list of mentions.

    This function transforms a list of mentions into a sequence of 'START' and
    'END' events, sorted by their position in the text. This is crucial for
    correctly handling nested and overlapping mentions when annotating text.

    Args:
      mentions: A sequence of tuples, where each tuple represents a mention
        and contains the start and end indices of the mention in the text.

    Returns:
      A sequence of tuples, where each tuple represents an event. The format
      is (position, event_type, span), where:
        - position: The character position of the event in the text.
        - event_type: A string, either 'START' or 'END', indicating the type
          of event.
        - span: The original mention span (start, end).
    """
    # Use events to correctly handle nested and overlapping mentions.
    events = []
    for span in mentions:
      start, end = span
      events.append((start, 'START', span))
      events.append((end, 'END', span))

    # Sort events to handle nesting correctly. The key ensures that at any
    # given character position, inner mentions are closed before outer ones,
    # and outer mentions are opened before inner ones.
    def sort_key(event):
      pos, event_type, span = event
      span_length = span[1] - span[0]

      # Process END events before START events at the same position.
      type_order = -1 if event_type == 'END' else 1

      # For ENDs, sort by length ascending (inner first), thus in for example:
      # `Bank of America`, the mention for `America` will be closed before the
      # one for `Bank of America`.
      # For STARTs, sort by length descending (outer first).
      length_order = span_length if event_type == 'END' else -span_length

      return (pos, type_order, length_order)
    events.sort(key=sort_key)

    return events

  def generate_annotated_text(
      self,
      text,
      mentions,
      mention_to_id_map
  ):
    """Generates the text with mentions annotated according to the strategy.

    Args:
      text: The text to annotate.
      mentions: The mentions to annotate.
      mention_to_id_map: A mapping from mentions to unique indices.

    Returns:
      The text with mentions annotated according to the strategy.
    """
    events = self._get_sorted_events_from_mentions(mentions)

    return self.annotate_text_with_events(text, events, mention_to_id_map)


class BracketMarkupStrategy(GoldMentionsMarkupStrategy):
  """Marks up mentions using customizable brackets, e.g., 1_[mention]."""

  def __init__(self, start_format = '{index}_[', end_format = ']'):
    """Initializes the bracket markup strategy.

    Args:
      start_format: The format string for the start bracket. Can contain an
        optional placeholder for the mention index.
      end_format: The format string for the end bracket. Can contain an
        optional placeholder for the mention index.
    """
    self.start_format = start_format
    self.end_format = end_format

  def _get_end_marker(self, mention_id):
    return self.end_format.format(index=mention_id)

  def _get_start_marker(self, mention_id):
    return self.start_format.format(index=mention_id)


class TargetFormatStrategy(abc.ABC):
  """Abstract base class for coreference target string formatting strategies."""

  @abc.abstractmethod
  def generate_target_string(
      self,
      clusters,
      mention_to_id_map
  ):
    """Generates the target string from clusters according to the strategy."""


class ClusterGroupsTargetStrategy(TargetFormatStrategy):
  """Formats the target as cluster groups, e.g., 1|2|4#3|8.

  Where | is an in-cluster separator and # is a between-clusters separator.
  """

  def __init__(
      self,
      in_cluster_separator = '|',
      between_clusters_separator = '#',
  ):
    self.in_cluster_separator = in_cluster_separator
    self.between_clusters_separator = between_clusters_separator

  def generate_target_string(
      self,
      clusters,
      mention_to_id_map,
  ):
    cluster_strings = []
    for cluster in clusters:
      if not cluster:
        continue
      mention_ids = sorted([mention_to_id_map[span] for span in cluster])
      cluster_strings.append(
          self.in_cluster_separator.join(map(str, mention_ids))
      )
    return self.between_clusters_separator.join(cluster_strings)

  def parse_string_representation(
      self, str_representation
  ):
    """Parses a string of pairs back into a list of clusters.

    This is the inverse of `generate_target_string`. It takes a string
    like "1|2|4|5|6|9#3|8" and reconstructs the original clusters,
    e.g., [['1', '2', '4', '5', '6', '9'], ['3', '8']].

    Args:
      str_representation: The string of anaphor-antecedent pairs.

    Returns:
      A list of clusters, where each cluster is a list of mention IDs.
    """
    clusters = str_representation.split(self.between_clusters_separator)
    res = []
    for cluster in clusters:
      res.append(cluster.split(self.in_cluster_separator))
    return res


class AnaphorAntecedentTargetStrategy(TargetFormatStrategy):
  """Formats the target as anaphor-antecedent pairs, e.g., "1,1#2,1#4,1#8,3".

  This strategy identifies the first mention in a cluster as the antecedent
  and all subsequent mentions as anaphors. It creates pairs of
  (anaphor_id, antecedent_id). The antecedent itself is paired with itself.
  """

  def __init__(
      self,
      pair_separator = ',',
      between_pairs_separator = '#',
      order_by_anaphor_id = True,
  ):
    self.pair_separator = pair_separator
    self.between_pairs_separator = between_pairs_separator
    self.order_by_anaphor_id = order_by_anaphor_id

  def generate_target_string(
      self,
      clusters,
      mention_to_id_map,
  ):
    all_pairs = []
    for cluster in clusters:
      # Get the IDs for mentions in the current cluster that are in the map.
      mention_ids = [
          mention_to_id_map[span]
          for span in cluster
          if span in mention_to_id_map
      ]
      if not mention_ids:
        continue

      # The antecedent is the first mention in the cluster (smallest ID).
      antecedent_id = min(mention_ids)

      # Create a pair (as a tuple) for every mention in the cluster.
      for anaphor_id in mention_ids:
        all_pairs.append((anaphor_id, antecedent_id))

    # Sort all pairs by the anaphor ID. If False, the pairs are grouped by
    # cluster, and are sorted within each cluster.
    if self.order_by_anaphor_id:
      all_pairs.sort()

    # Format the sorted pairs into strings.
    pair_strings = [
        f'{anaphor}{self.pair_separator}{antecedent}'
        for anaphor, antecedent in all_pairs
    ]

    return self.between_pairs_separator.join(pair_strings)

  def parse_string_representation(
      self, str_representation
  ):
    """Parses a string of pairs back into a list of clusters.

    This is the inverse of `generate_target_string`. It takes a string
    like "1,1#2,1#3,3#8,3" and reconstructs the original clusters,
    e.g., [['1', '2'], ['3', '8']].

    Args:
      str_representation: The string of anaphor-antecedent pairs.

    Returns:
      A list of clusters, where each cluster is a list of mention IDs.
    """
    if not str_representation:
      return []

    antecedent_to_anaphors = {}
    pairs = str_representation.split(self.between_pairs_separator)

    for pair_str in pairs:
      anaphor_id, antecedent_id = pair_str.split(self.pair_separator)
      if not anaphor_id or not antecedent_id:
        # Handle cases with malformed pairs, e.g., "1," or ",2"
        raise ValueError(
            f'Invalid pair string: {pair_str}. Each pair is expected to be a'
            ' sequence of anaphor, antecedent separated by'
            f' {self.pair_separator}'
        )

      if antecedent_id not in antecedent_to_anaphors:
        antecedent_to_anaphors[antecedent_id] = []
      antecedent_to_anaphors[antecedent_id].append(anaphor_id)

    # The values of the dictionary are the reconstructed clusters.
    return list(antecedent_to_anaphors.values())


class CorefWithGoldMentionsParser:
  """A class for parsing coreference data with gold mentions.

  It uses a markup strategy to annotate gold mentions in the text and a target
  strategy to format the target string. It includes an option to merge
  sub-word mentions into a single-word annotation.
  """

  def __init__(
      self,
      markup_strategy,
      target_format_strategy,
      merge_subword_mentions = False,
  ):
    """Initializes the parser with the markup and target strategies.

    Args:
      markup_strategy: The strategy for marking up mentions in the text.
      target_format_strategy: The strategy for formatting the target string.
      merge_subword_mentions: If True, mentions within a single word are
        merged into one annotation. If False, each mention is annotated
        separately.
    """
    self.markup_strategy = markup_strategy
    self.target_format_strategy = target_format_strategy
    self.merge_subword_mentions = merge_subword_mentions

  def get_adjusted_char_limit(
      self, text, char_limit = None,
  ):
    """Returns the adjusted character limit for the text.

    Iterates over words in the text and finds the last word that fits within the
    character limit.

    Args:
      text: The text to iterate over.
      char_limit: The character limit to use.
    """
    adjusted_limit = len(text)
    if char_limit is not None:
      for match in re.finditer(r'\S+', text):
        if match.end() <= char_limit:
          adjusted_limit = match.end()
        else:
          break

    return adjusted_limit

  def _filter_exceeding_mentions(
      self,
      clusters,
      adjusted_char_limit,
  ):
    """Filters mentions that exceed the character limit.

    Args:
      clusters: The clusters to filter.
      adjusted_char_limit: The adjusted character limit.

    Returns:
      The valid clusters.
    """
    filtered_clusters = []
    for cluster in clusters:
      new_cluster = [span for span in cluster if span[1] <= adjusted_char_limit]
      if new_cluster:
        filtered_clusters.append(new_cluster)

    return filtered_clusters

  def _find_containing_word_span(
      self,
      char_index,
      word_spans
  ):
    """Finds the word span containing a given character index using binary search.

    Args:
      char_index: The character index to search for.
      word_spans: A sorted list of all word spans in the text.

    Returns:
      The `(start, end)` span of the word containing the character, or None if
      the character is not within any word (e.g., it's whitespace).
    """
    # bisect_left finds an insertion point which comes after (to the right of)
    # any existing entries of char_index in word_spans.
    # We search for a tuple where the start is our char_index.
    insertion_point = bisect.bisect_left(word_spans, (char_index, 0))

    # The index is the *exact start* of the span at the insertion point.
    # If insertion_point is valid and the span at that point starts exactly
    # at char_index, we found our word.
    if insertion_point < len(word_spans):
      candidate_span = word_spans[insertion_point]
      if candidate_span[0] == char_index:
        return candidate_span

    # The index is *inside* the span *before* the insertion point.
    # If insertion_point > 0, the span at [insertion_point - 1] must start
    # before char_index. We just need to check if it also *ends* after
    # char_index (i.e., it contains it).
    if insertion_point > 0:
      candidate_span = word_spans[insertion_point - 1]
      # A character is contained if start <= char_index < end.
      if candidate_span[0] <= char_index < candidate_span[1]:
        return candidate_span
    return None

  def _get_annotation_map(
      self,
      all_mentions,
      word_spans,
  ):
    """Creates a map from an original mention to its final annotation span.

    This method maps any mention to the full span of words it "touches".
    If a mention starts *inside* a word, the annotation is expanded to the
    start of that word.
    If a mention ends *inside* a word, the annotation is expanded to the
    end of that word.
    This correctly handles both single-word sub-spans and multi-word mentions
    that involve sub-word spans (e.g., "In[Tel Aviv]").

    Args:
      all_mentions: A sequence of all mention spans.
      word_spans: A sorted list of all word spans in the text.

    Returns:
      A mapping from each original mention span to the span that should be
      used for annotation in the text.
    """
    original_to_annotated_map = {}
    for mention in all_mentions:
      mention_start, mention_end = mention

      # Find the word containing the *first* character of the mention.
      start_word_span = self._find_containing_word_span(
          mention_start, word_spans
      )
      # Find the word containing the *last* character of the mention
      # (at index mention_end - 1). mention_end is exclusive, so the last char
      # is at mention_end - 1.
      end_word_span = self._find_containing_word_span(
          mention_end - 1, word_spans
      )

      # If both start and end are within valid word spans:
      if start_word_span and end_word_span:
        full_annotation_span = (start_word_span[0], end_word_span[1])
        original_to_annotated_map[mention] = full_annotation_span
      else:
        # If the mention starts or ends outside a word (e.g., it's just
        # whitespace or punctuation that was stripped), map it to itself.
        original_to_annotated_map[mention] = mention

    return original_to_annotated_map

  def _get_word_spans(self, text):
    """Returns a list of all word spans in the text."""
    punctuations_to_strip = '.,?!:;()[]{}"\'׳״־׃׀،؛؟٬٫'
    word_spans = []

    for match in re.finditer(r'\S+', text):
      token = match.group()
      start, end = match.span()

      # Strip leading punctuation
      while token and token[0] in punctuations_to_strip:
        token = token[1:]
        start += 1
      # Strip trailing punctuation
      while token and token[-1] in punctuations_to_strip:
        token = token[:-1]
        end -= 1

      if token:  # Add span if token is not empty after stripping
        word_spans.append((start, end))
    return word_spans

  def _process_mentions(
      self, text, all_mentions
  ):
    """Determines the final annotation spans based on the sub-word strategy.

    If `merge_subword_mentions` is True, this method identifies sub-word
    mentions and maps them to whole-word spans. Otherwise, it returns the
    original mentions, mapping each to itself.

    Args:
      text: The input text.
      all_mentions: A sequence of all unique mention spans.

    Returns:
      A tuple containing:
        A sequence of unique, sorted spans to be annotated in the text.
        A mapping from each original mention to its annotated span.
    """
    if not all_mentions:
      return [], {}

    if not self.merge_subword_mentions:
      mentions_for_annotation = all_mentions
      original_to_annotated_map = {m: m for m in all_mentions}
      return mentions_for_annotation, original_to_annotated_map

    word_spans = self._get_word_spans(text)
    original_to_annotated_map = self._get_annotation_map(
        all_mentions, word_spans
    )

    # The spans to be annotated are the unique values in the map.
    mentions_for_annotation = sorted(
        list(set(original_to_annotated_map.values())), key=lambda x: x[0]
    )

    return mentions_for_annotation, original_to_annotated_map

  def _prepare_data(
      self,
      text,
      clusters,
      char_limit,
  ):
    """Prepares data by adjusting text length and filtering mentions.

    Args:
      text: The original text.
      clusters: The original coreference clusters.
      char_limit: An optional character limit for the text.

    Returns:
      A tuple of the processed (text, clusters, all_mentions).
    """
    adjusted_char_limit = self.get_adjusted_char_limit(text, char_limit)

    text = text[:adjusted_char_limit]
    clusters = self._filter_exceeding_mentions(clusters, adjusted_char_limit)

    # Assign unique indices to all mentions and sort them by their start
    # position.
    all_mentions = []
    for cluster in clusters:
      for mention in cluster:
        all_mentions.append(mention)
    all_mentions = sorted(all_mentions, key=lambda x: x[0])

    return text, clusters, all_mentions

  def parse_example(
      self,
      text,
      clusters,
      char_limit = None,
  ):
    """Parses the text and clusters to produce the annotated text and target.

    This is the main public method of the class. It orchestrates the entire
    parsing pipeline:
    1. Prepares the data (truncates text, filters mentions).
    2. Processes mentions, merging sub-word mentions if required.
    3. Assigns IDs to the mentions that will be annotated.
    4. Generates the final annotated text and target string using the
       configured strategies.

    Args:
      text: The text to parse and annotate with gold mentions.
      clusters: The clusters to parse into the target string.
      char_limit: The character limit to apply (character limit of the
        non-indexed text).

    Returns:
      A CorefExample containing the annotated text and the target string.
    """
    text, clusters, all_mentions = self._prepare_data(
        text, clusters, char_limit
    )
    mentions_for_annotation, original_to_annotated_map = self._process_mentions(
        text, all_mentions
    )

    # Assign unique integer IDs to the final set of spans that will be marked up
    # in the text.
    annotated_mention_to_id_map = {
        mention: i + 1 for i, mention in enumerate(mentions_for_annotation)
    }
    # Use the markup strategy to generate the annotated source text.
    annotated_text = self.markup_strategy.generate_annotated_text(
        text, mentions_for_annotation, annotated_mention_to_id_map
    )

    # Build a map from each *original* mention to the ID of its
    # (possibly merged) annotated span. Thus, sub-word mentions in word would
    # share the same ID.
    target_mention_to_id_map = {
        original: annotated_mention_to_id_map[annotated]
        for original, annotated in original_to_annotated_map.items()
        if annotated in annotated_mention_to_id_map
    }
    # Generate the target string using the original clusters.
    target_string = self.target_format_strategy.generate_target_string(
        clusters, target_mention_to_id_map
    )

    return CorefExample(input=annotated_text, label=target_string)
